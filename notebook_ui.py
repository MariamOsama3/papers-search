"""Chat UI for the paper research agent that runs inside a Jupyter notebook in VS Code.

No web server and no port forwarding: the widgets talk to the kernel over the same
connection VS Code already uses. Open chat_ui.ipynb and run its cell.

Flow: 1) search a topic  2) tick papers and download/index them  3) chat about them.
All the real work is done by agents/orchestra.py; this file only adds the UI.
"""
import asyncio
import html
import io
import sys
import threading
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import ipywidgets as W
from IPython.display import display, Markdown

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import agents.orchestra as orc  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

orc.OPEN_IMAGES = False  # show images in the notebook, never pop viewers on the server

# ---------------------------------------------------------------------------
# A private event loop for the async LangGraph agent (Jupyter already runs its own loop)
# ---------------------------------------------------------------------------
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


class _LiveLog(io.TextIOBase):
    """stdout replacement used while the agent runs. Agent output comes from the background
    loop thread, so it is appended to the log widget directly (works while the kernel's main
    thread is blocked) and the last line is mirrored into the status bar."""

    def __init__(self, ui):
        self.ui = ui
        self._lock = threading.Lock()

    def write(self, s):
        if s:
            with self._lock:
                self.ui.log.append_stdout(s)
                last = s.strip().splitlines()[-1:] if s.strip() else []
                if last:
                    self.ui._set_status("⏳ " + html.escape(last[0][:140]))
        return len(s)

    def flush(self):
        pass


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


async def _ask_agent(graph, question, topic, thread_id):
    if orc._rag_dirty and orc._session_chunks:  # a paper was downloaded from the chat
        await asyncio.to_thread(orc.reindex_sync)
        orc._rag_dirty = False
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    answer, tools = "", []
    async for event in graph.astream(
        {"messages": [HumanMessage(content=question)], "papers": orc.REGISTRY, "research_topic": topic},
        config, stream_mode="values",
    ):
        msgs = event.get("messages") or []
        if not msgs or msgs[-1].type != "ai":
            continue
        if msgs[-1].tool_calls:
            tools += [f"{tc['name']}({tc['args']})" for tc in msgs[-1].tool_calls]
        else:
            answer = _text(msgs[-1])
    return answer, orc.show_answer_images(answer), tools


# ---------------------------------------------------------------------------
# The UI
# ---------------------------------------------------------------------------
class PaperChatUI:
    def __init__(self):
        self.state = {"topic": "", "papers": [], "graph": None, "thread": str(uuid.uuid4())}
        self.checks = {}

        # -- step 1: search
        self.topic = W.Text(placeholder="e.g. diffusion models for medical imaging",
                            layout=W.Layout(width="60%"))
        self.search_btn = W.Button(description="Search papers", button_style="primary", icon="search")
        self.llm_sum = W.Checkbox(value=orc.USE_LLM_SUMMARIES, description="LLM summaries (off = instant)",
                                  indent=False)
        self.describe = W.Checkbox(value=orc.DESCRIBE_IMAGES, indent=False,
                                   description="Describe figures with LLM (slow; enables figure Q&A)")
        self.max_papers = W.IntSlider(value=orc.TARGET_PAPERS, min=3, max=20, description="Max papers")
        self.min_score = W.FloatSlider(value=orc.MIN_SCORE, min=0.0, max=0.8, step=0.05,
                                       description="Min relevance")
        self.log = W.Output(layout=W.Layout(max_height="220px", overflow="auto",
                                            border="1px solid #ccc", padding="4px"))
        self.status = W.HTML()

        # -- step 2: papers
        self.papers_box = W.VBox()
        self.download_btn = W.Button(description="Download & index selected", button_style="success",
                                     icon="download", disabled=True)

        # -- step 3: chat
        self.chat_out = W.Output(layout=W.Layout(border="1px solid #ccc", padding="8px",
                                                 max_height="600px", overflow="auto"))
        self.question = W.Text(placeholder="Download papers first, then ask about them...",
                               layout=W.Layout(width="80%"), disabled=True)
        self.send_btn = W.Button(description="Send", button_style="primary", icon="paper-plane",
                                 disabled=True)

        self.search_btn.on_click(self._safe(self.on_search))
        self.download_btn.on_click(self._safe(self.on_download))
        self.send_btn.on_click(self._safe(self.on_send))
        self.question.continuous_update = False
        self.question.observe(self._on_enter, names="value")

    # widget callbacks swallow exceptions silently, so show them in the log
    def _safe(self, fn):
        def wrapped(*_):
            try:
                fn()
            except Exception as e:
                self._set_status(f"❌ {type(e).__name__}: {e}", "red")
                with self.log:
                    import traceback
                    traceback.print_exc()
                self._busy(False)
        return wrapped

    def _set_status(self, text, color="#444"):
        self.status.value = f"<b style='color:{color}'>{text}</b>"

    def _busy(self, busy: bool):
        self.search_btn.disabled = busy
        self.download_btn.disabled = busy or not self.state["papers"]
        ready = bool(orc._downloaded_ids) and not busy
        self.question.disabled = not ready
        self.send_btn.disabled = not ready

    # -- step 1 -------------------------------------------------------------
    def on_search(self):
        topic = self.topic.value.strip()
        if not topic:
            self._set_status("Type a topic first.", "orange")
            return
        orc.USE_LLM_SUMMARIES = self.llm_sum.value
        orc.TARGET_PAPERS = self.max_papers.value
        orc.MIN_SCORE = self.min_score.value

        # forget the previous topic
        orc.REGISTRY.clear()
        orc._downloaded_ids.clear()
        orc._session_chunks.clear()
        orc._rag_dirty = False
        self.state.update(topic=topic, papers=[], graph=None, thread=str(uuid.uuid4()))
        self.chat_out.clear_output()
        self.papers_box.children = []
        self.log.clear_output()

        self._busy(True)
        self._set_status(f"⏳ Searching papers on “{topic}”...")
        with self.log:
            papers = orc.find_papers(topic)
            trends = []
            if papers:
                if orc.USE_LLM_SUMMARIES:
                    self._set_status("⏳ Writing LLM summaries...")
                    print("\nWriting LLM summaries...")
                else:
                    print("\nLLM summaries off: using the first sentence of each abstract (instant).")
                papers = orc.summarize_papers(papers)
                trends = orc.generate_trends(papers)

        if not papers:
            self._set_status("No relevant papers found. Try other wording or lower Min relevance.", "red")
            self._busy(False)
            return

        orc.REGISTRY.extend(papers)
        self.state["papers"] = papers
        self._render_papers(papers, trends)
        self._set_status(f"✅ Found {len(papers)} papers. Tick the ones you want, then press "
                         "“Download & index selected”.", "green")
        self._busy(False)

    def _render_papers(self, papers, trends):
        self.checks = {}
        rows = []
        if trends:
            rows.append(W.HTML("<h4>Key trends</h4><ul>" + "".join(f"<li>{t}</li>" for t in trends)
                               + "</ul>"))
        for i, p in enumerate(papers, 1):
            cb = W.Checkbox(value=False, indent=False,
                            description=f"{i}. {p['title']}",
                            layout=W.Layout(width="100%"),
                            style={"description_width": "initial"})
            self.checks[p["id"]] = cb
            info = W.HTML(
                f"<div style='margin:0 0 10px 22px'>"
                f"<small>{p['authors']} ({p['year']}) · {p['source']} · "
                f"<a href='{p['url']}' target='_blank'>arXiv:{p['id']}</a> · "
                f"relevance {p['relevance']:.2f}</small><br>{p['summary']}</div>")
            rows.append(W.VBox([cb, info]))
        self.papers_box.children = rows

    # -- step 2 -------------------------------------------------------------
    def on_download(self):
        todo = [p for p in self.state["papers"]
                if self.checks[p["id"]].value and p["id"] not in orc._downloaded_ids]
        if not todo:
            self._set_status("Tick at least one new paper first.", "orange")
            return
        orc.DESCRIBE_IMAGES = self.describe.value
        self._busy(True)
        failed = []
        with self.log:
            for k, p in enumerate(todo, 1):
                self._set_status(f"⏳ Processing paper {k}/{len(todo)}: {p['title'][:70]}...")
                print(f"\nDownloading: {p['title']} (arXiv:{p['id']})")
                try:
                    orc.download_from_registry(p)
                    print("   done")
                except Exception as e:
                    print(f"   FAILED: {e}")
                    failed.append(p["title"])
            if orc._session_chunks:
                self._set_status("⏳ Building the search index...")
                orc.reindex_sync()
                orc._rag_dirty = False

        for pid, cb in self.checks.items():
            if pid in orc._downloaded_ids:
                cb.value, cb.disabled = True, True
                if not cb.description.endswith("✅"):
                    cb.description += "  ✅"
        if orc._session_chunks and self.state["graph"] is None:
            self.state["graph"] = orc.build_graph()

        msg = f"✅ {len(orc._downloaded_ids)} paper(s) ready. Ask your questions below."
        if failed:
            msg += " Failed: " + "; ".join(failed)
        self._set_status(msg, "green" if not failed else "orange")
        self._busy(False)

    # -- step 3 -------------------------------------------------------------
    def _on_enter(self, change):
        if change["new"].strip() and not self.send_btn.disabled:
            self._safe(self.on_send)()

    def on_send(self):
        q = self.question.value.strip()
        if not q:
            return
        self.question.unobserve(self._on_enter, names="value")
        self.question.value = ""
        self.question.observe(self._on_enter, names="value")
        if self.state["graph"] is None:
            self.state["graph"] = orc.build_graph()

        with self.chat_out:
            display(Markdown(f"**🧑 You:** {q}"))
        self._busy(True)
        self._set_status("⏳ Thinking...")

        orc.DESCRIBE_IMAGES = self.describe.value  # used if the agent downloads a paper
        with redirect_stdout(_LiveLog(self)):  # stream agent logs (e.g. a chat download) into the progress log
            answer, images, tools = run_async(_ask_agent(
                self.state["graph"], q, self.state["topic"], self.state["thread"]))

        with self.chat_out:
            display(Markdown("**🤖 Agent:**\n\n" + (orc.IMG_TAG.sub("", answer).strip() or "_(empty)_")))
            for path in images:
                display(W.Image(value=Path(path).read_bytes(), format=Path(path).suffix.lstrip("."),
                                layout=W.Layout(max_width="600px")))
                display(Markdown(f"<small>{Path(path).name}</small>"))
            if tools:
                display(Markdown(f"<small>Tools used: {', '.join(tools)}</small>"))
            display(Markdown("---"))

        if any(t.startswith("download_paper") for t in tools):
            for pid, cb in self.checks.items():
                if pid in orc._downloaded_ids and not cb.disabled:
                    cb.value, cb.disabled = True, True
                    if not cb.description.endswith("✅"):
                        cb.description += "  ✅"
        self._set_status("✅ Ready.", "green")
        self._busy(False)

    # -- layout -------------------------------------------------------------
    def widget(self):
        return W.VBox([
            W.HTML("<h2>📄 Paper Research Agent</h2>"),
            W.HTML("<h3>1. Search</h3>"),
            W.HBox([self.topic, self.search_btn]),
            W.HBox([self.llm_sum, self.max_papers, self.min_score]),
            self.describe,
            self.status,
            W.Accordion(children=[self.log], titles=("Progress log",)),
            W.HTML("<h3>2. Choose papers</h3>"),
            self.papers_box,
            self.download_btn,
            W.HTML("<h3>3. Chat</h3>"),
            self.chat_out,
            W.HBox([self.question, self.send_btn]),
        ])


def show():
    ui = PaperChatUI()
    root = ui.widget()
    for child in root.children:  # open the progress log
        if isinstance(child, W.Accordion):
            child.selected_index = 0
    display(root)
    return ui
