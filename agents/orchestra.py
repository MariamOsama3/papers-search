import sys
import re
import json
import asyncio
import operator
import os
import time
from pathlib import Path
from typing import Annotated, TypedDict, List, Dict

import feedparser          # installed together with the `arxiv` package
import numpy as np
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")
if hasattr(sys.stdout, "reconfigure"):  # Jupyter's OutStream lacks it
    sys.stdout.reconfigure(encoding="utf-8")

from agents.Paper_RAG import (
    initialize_rag,
    retrieve_passages,
    get_paper_images,
    process_pdf_to_chunks,
    save_paper_meta,
    set_allowed_papers,
)

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_PAPERS = 10
MIN_SCORE = 0.40          # relevance threshold; tune using the printed scores
PAPERS_DIR = project_root / "downloaded_papers"
# Put a real contact email here - arXiv is more tolerant of clients that identify themselves.
UA = {"User-Agent": "paper-research-agent/1.0 (mailto:your_email@example.com)"}
ARXIV_DELAY = 3.5         # seconds between arXiv requests (arXiv asks for ~1 request / 3 s)
ARXIV_MAX_RETRIES = 5
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_TIMEOUT = 30        # seconds - a stuck request fails instead of hanging forever
OPEN_IMAGES = True        # open images used in an answer in the default image viewer

llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
    num_ctx=8192,
    base_url="http://localhost:11434",
)
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Summaries / trends: capped output + timeout, so a runaway JSON generation cannot hang the run.
USE_LLM_SUMMARIES = True  # False = instant summaries (first sentence of each abstract)
DESCRIBE_IMAGES = True    # False = skip LLM figure descriptions when processing a PDF (much faster)
SUMMARY_TIMEOUT = 180     # seconds per LLM call before falling back to the abstract
summary_llm = ChatOllama(
    model="qwen2.5:7b-instruct",  # 7B is plenty for 1-2 sentence summaries and fits on the GPU far better than 14B
    temperature=0,
    num_ctx=4096,          # prompts here are small; a smaller context loads and runs faster
    num_predict=600,       # hard cap on output tokens (5 short summaries fit easily)
    keep_alive="30m",      # keep the model loaded between runs instead of reloading it each time
    base_url="http://localhost:11434",
    client_kwargs={"timeout": SUMMARY_TIMEOUT},
)


# ---------------------------------------------------------------------------
# Types and the single source of truth (the registry)
# ---------------------------------------------------------------------------
class Paper(TypedDict):
    id: str
    title: str
    authors: str
    year: int
    url: str
    source: str
    abstract: str
    relevance: float
    summary: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    research_topic: str
    papers: list[Paper]


REGISTRY: List[Paper] = []          # papers found by the search step
_downloaded_ids: set = set()        # registry papers that are downloaded + chunked (= SELECTED)
_session_chunks: set = set()        # chunk files created this session
_rag_dirty = False                  # True when RAG must be rebuilt
_last_images: Dict[str, str] = {}   # image_id -> path, from the latest retrieval only

IMG_TAG = re.compile(r"\[IMG:\s*([^\]]+?)\s*\]")


# ---------------------------------------------------------------------------
# 1. Search (done in code, not by the model)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "about", "using", "into",
    "on", "of", "in", "to", "an", "a", "or", "by", "via", "based", "new", "recent",
}


_last_arxiv_call = 0.0


def _arxiv_wait() -> None:
    """Respect arXiv's rate limit across the whole session (search + downloads)."""
    global _last_arxiv_call
    gap = time.time() - _last_arxiv_call
    if gap < ARXIV_DELAY:
        time.sleep(ARXIV_DELAY - gap)
    _last_arxiv_call = time.time()


def clean_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", raw.strip())


def keywords(topic: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9\-]+", topic.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def search_arxiv(query: str, n: int = 25, max_attempts: int = 3) -> List[Paper]:
    """Query the arXiv API directly, with a timeout, rate-limit handling and progress output."""
    print(f"   Searching arXiv: {query}")
    for attempt in range(max_attempts):
        _arxiv_wait()
        try:
            r = requests.get(
                ARXIV_API,
                params={"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"},
                headers=UA,
                timeout=ARXIV_TIMEOUT,
            )
        except requests.RequestException as e:
            print(f"   [arXiv request failed: {e}] (attempt {attempt + 1}/{max_attempts})")
            continue

        if r.status_code in (429, 503):
            retry_after = r.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else min(10 * 2 ** attempt, 60)
            print(f"   arXiv rate limited ({r.status_code}), waiting {wait}s... "
                  f"(attempt {attempt + 1}/{max_attempts})")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"   [arXiv returned {r.status_code}, skipping this query]")
            return []

        out = []
        for e in feedparser.parse(r.content).entries:
            raw_id = e.get("id", "").rsplit("/abs/", 1)[-1]
            if not raw_id:
                continue
            names = [a.get("name", "") for a in e.get("authors", [])]
            authors = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
            published = e.get("published", "")
            out.append(Paper(
                id=clean_id(raw_id),
                title=" ".join(e.get("title", "").split()),
                authors=authors,
                year=int(published[:4]) if published[:4].isdigit() else 0,
                url=e.get("id", ""),
                source="arXiv",
                abstract=" ".join(e.get("summary", "").split()),
                relevance=0.0,
                summary="",
            ))
        print(f"   -> {len(out)} results")
        return out

    print(f"   [arXiv search gave up after {max_attempts} attempts]")
    return []


def search_semantic_scholar(query: str, n: int = 25) -> List[Paper]:
    headers = {}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    print(f"   Searching Semantic Scholar: {query}")
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": n,
                "fields": "title,abstract,year,authors,venue,externalIds",
            },
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            print(f"   [Semantic Scholar returned {r.status_code}, skipping]")
            return []
        out = []
        for p in r.json().get("data", []):
            aid = (p.get("externalIds") or {}).get("ArXiv")
            if not aid or not p.get("abstract") or not p.get("title"):
                continue
            authors = ", ".join(a["name"] for a in (p.get("authors") or [])[:3])
            out.append(Paper(
                id=clean_id(aid),
                title=" ".join(p["title"].split()),
                authors=authors,
                year=p.get("year") or 0,
                url=f"https://arxiv.org/abs/{clean_id(aid)}",
                source=p.get("venue") or "Semantic Scholar",
                abstract=" ".join(p["abstract"].split()),
                relevance=0.0,
                summary="",
            ))
        print(f"   -> {len(out)} results with an arXiv ID")
        return out
    except Exception as e:
        print(f"   [Semantic Scholar failed: {e}]")
        return []


def rank_by_relevance(topic: str, papers: List[Paper]) -> List[Paper]:
    if not papers:
        return []
    q = np.array(embedder.embed_query(topic))
    docs = np.array(embedder.embed_documents([f"{p['title']}. {p['abstract'][:1000]}" for p in papers]))
    sims = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))
    for p, s in zip(papers, sims):
        p["relevance"] = float(s)

    ranked = sorted(papers, key=lambda p: -p["relevance"])
    print("\n   Relevance scores (top 15):")
    for p in ranked[:15]:
        mark = "keep" if p["relevance"] >= MIN_SCORE else "drop"
        print(f"   {p['relevance']:.2f} [{mark}] {p['title'][:80]}")

    return [p for p in ranked if p["relevance"] >= MIN_SCORE]


def find_papers(topic: str) -> List[Paper]:
    kw = keywords(topic)
    arxiv_queries = [
        " AND ".join(f"all:{w}" for w in kw) if kw else topic,
        " ".join(kw) or topic,
        topic,
    ]
    s2_queries = [" ".join(kw) or topic]

    pool: List[Paper] = []
    seen_ids, seen_titles = set(), set()

    def add(papers: List[Paper]):
        for p in papers:
            tkey = re.sub(r"\W+", "", p["title"].lower())
            if p["id"] in seen_ids or tkey in seen_titles:
                continue
            seen_ids.add(p["id"])
            seen_titles.add(tkey)
            pool.append(p)

    total = len(arxiv_queries) + len(s2_queries)
    step = 0
    for q in arxiv_queries:
        step += 1
        print(f"\n   [Search {step}/{total}] arXiv (different wording each time, results are merged)")
        add(search_arxiv(q))
        print(f"   pool so far: {len(pool)} unique papers")
        if len(pool) >= 60:
            print("   enough candidates, skipping the remaining arXiv searches")
            break
    for q in s2_queries:
        step = max(step + 1, len(arxiv_queries) + 1)
        print(f"\n   [Search {step}/{total}] Semantic Scholar")
        add(search_semantic_scholar(q))
        print(f"   pool so far: {len(pool)} unique papers")

    print(f"\n   Collected {len(pool)} unique candidates. Ranking by relevance...")
    ranked = rank_by_relevance(topic, pool)

    if len(ranked) < TARGET_PAPERS:
        print(f"\n   NOTE: only {len(ranked)} papers passed the relevance threshold ({MIN_SCORE}). "
              f"Lower MIN_SCORE if you want more (at the cost of relevance).")
    return ranked[:TARGET_PAPERS]


# ---------------------------------------------------------------------------
# 2. Summaries (model writes text only, code controls structure)
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    id: str
    key_contribution: str


class BatchSummaries(BaseModel):
    items: List[PaperSummary]


class TrendList(BaseModel):
    key_trends: List[str]


def first_sentence(text: str, limit: int = 300) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip())[0][:limit]


def summarize_papers(papers: List[Paper]) -> List[Paper]:
    if not USE_LLM_SUMMARIES:
        for p in papers:
            p["summary"] = first_sentence(p["abstract"])
        return papers

    summarizer = summary_llm.with_structured_output(BatchSummaries)
    total_batches = (len(papers) - 1) // 5 + 1

    for start in range(0, len(papers), 5):
        batch = papers[start:start + 5]
        n = start // 5 + 1
        print(f"   Summarizing batch {n}/{total_batches} ({len(batch)} papers)... "
              "(the first call also loads the model)")
        t0 = time.time()
        listing = "\n\n".join(
            f"id: {p['id']}\ntitle: {p['title']}\nabstract: {p['abstract'][:700]}" for p in batch
        )
        prompt = (
            "Summarize each paper's core method or finding in 1-2 sentences, using ONLY its abstract. "
            "Return exactly one item per paper and copy each id exactly as given.\n\n" + listing
        )
        by_id = {}
        try:
            result = summarizer.invoke(prompt)
            by_id = {it.id.strip(): it.key_contribution.strip() for it in result.items}
            print(f"   -> done in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"   [summary batch failed after {time.time() - t0:.0f}s, "
                  f"using abstract fallback: {type(e).__name__}: {e}]")

        for p in batch:
            p["summary"] = by_id.get(p["id"]) or first_sentence(p["abstract"])

    return papers


def generate_trends(papers: List[Paper]) -> List[str]:
    if not USE_LLM_SUMMARIES:
        return []
    listing = "\n".join(f"- {p['title']}: {p['summary']}" for p in papers)
    print("   Writing key trends...")
    t0 = time.time()
    try:
        result = summary_llm.with_structured_output(TrendList).invoke(
            "Based ONLY on these papers, write 3-4 short key trends (one sentence each).\n\n" + listing
        )
        print(f"   -> done in {time.time() - t0:.0f}s")
        return [t.strip() for t in result.key_trends if t.strip()][:4]
    except Exception as e:
        print(f"   [trends generation failed after {time.time() - t0:.0f}s: {type(e).__name__}: {e}]")
        return []


# ---------------------------------------------------------------------------
# 3. Rendering (done in Python so the format is always exact)
# ---------------------------------------------------------------------------
def render_papers(papers: List[Paper], trends: List[str]) -> str:
    blocks = [
        f"### {i}. **[{p['title']}]({p['url']})**\n"
        f"- **Source/Venue**: {p['source']}\n"
        f"- **Authors & Year**: {p['authors']} ({p['year']})\n"
        f"- **arXiv ID**: {p['id']}\n"
        f"- **Key Contribution**: {p['summary']}\n"
        for i, p in enumerate(papers, 1)
    ]
    out = "\n---\n\n".join(blocks)
    if trends:
        out += "\n---\n\n### Key Trends\n" + "\n".join(f"- {t}" for t in trends)
    return out


# ---------------------------------------------------------------------------
# 4. Download (by registry index only - no invented IDs possible)
# ---------------------------------------------------------------------------
def fetch_arxiv_pdf(arxiv_id: str, max_retries: int = ARXIV_MAX_RETRIES) -> bytes:
    """Download a PDF with rate limiting, Retry-After support, exponential
    backoff, and a fallback between arXiv's two hosts."""
    urls = [
        f"https://export.arxiv.org/pdf/{arxiv_id}",   # arXiv's host for programmatic access
        f"https://arxiv.org/pdf/{arxiv_id}",
    ]
    last_err = None
    for attempt in range(max_retries):
        url = urls[attempt % len(urls)]
        _arxiv_wait()
        try:
            r = requests.get(url, headers=UA, timeout=90)
            if r.status_code in (429, 503):
                retry_after = r.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else min(10 * 2 ** attempt, 120)
                print(f"   rate limited ({r.status_code}), waiting {wait}s... "
                      f"(attempt {attempt + 1}/{max_retries})")
                last_err = f"HTTP {r.status_code}"
                time.sleep(wait)
                continue
            if r.status_code == 404:
                raise ValueError(f"arXiv has no PDF for {arxiv_id} (404).")
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                last_err = "response was not a PDF"
                time.sleep(min(5 * 2 ** attempt, 60))
                continue
            return r.content
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(5 * 2 ** attempt, 60))
    raise RuntimeError(f"Could not download {arxiv_id} after {max_retries} attempts: {last_err}")


def download_from_registry(p: Paper) -> Path:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPERS_DIR / f"{p['id'].replace('/', '_')}.pdf"

    if not path.exists():   # already-downloaded PDFs are reused, so test runs don't re-hit arXiv
        path.write_bytes(fetch_arxiv_pdf(p["id"]))

    save_paper_meta(str(path), p["title"], p["id"], p["authors"], p["year"])
    print(f"   processing PDF (figure descriptions {'on - slow' if DESCRIBE_IMAGES else 'off'})...")
    chunks_path = process_pdf_to_chunks(str(path), extract_images=True,
                                        describe_images=DESCRIBE_IMAGES, force=True)
    if not chunks_path:
        raise ValueError("No text could be extracted from the PDF.")

    _session_chunks.add(chunks_path)
    _downloaded_ids.add(p["id"])      # only mark as selected when it was really processed
    return path


def reindex_sync() -> None:
    """Rebuild the vector store from this session's chunk files and lock
    retrieval to the selected (downloaded) papers only."""
    initialize_rag(sorted(_session_chunks))
    set_allowed_papers(_downloaded_ids)


def parse_selection(text: str, papers: List[Paper]) -> List[Paper]:
    """Turn '1, 3, 5-7' or part of a title into registry papers."""
    chosen: List[int] = []

    for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", text):
        chosen.extend(range(int(a), int(b) + 1))
    text_wo_ranges = re.sub(r"\d+\s*-\s*\d+", " ", text)
    chosen.extend(int(n) for n in re.findall(r"\d+", text_wo_ranges))

    picked = [papers[i - 1] for i in dict.fromkeys(chosen) if 1 <= i <= len(papers)]
    if picked:
        return picked

    for part in (s.strip().lower() for s in text.split(",") if s.strip()):
        for p in papers:
            if part in p["title"].lower() and p not in picked:
                picked.append(p)
    return picked


# ---------------------------------------------------------------------------
# 5. Tools exposed to the model (small, focused list)
#    NOTE: retrieval is NOT a model tool - it runs in code before every answer.
# ---------------------------------------------------------------------------
def registry_block(papers: List[Paper]) -> str:
    """Selected papers (with summaries) are the only ones the model may discuss.
    Non-selected papers are listed by title only, flagged as off-limits."""
    active = [p for p in papers if p["id"] in _downloaded_ids]
    others = [(i, p) for i, p in enumerate(papers, 1) if p["id"] not in _downloaded_ids]

    if not active:
        out = "SELECTED PAPERS: (none yet)\n"
    else:
        out = "SELECTED PAPERS (the ONLY papers you may discuss):\n"
        for p in active:
            out += f"- {p['title']} (arXiv:{p['id']}, {p['year']}): {p['summary']}\n"

    if others:
        out += (
            "\nNOT SELECTED (you have NO content from these; never answer about them, "
            "only offer to download them by number):\n"
        )
        out += "\n".join(f"[{i}] {p['title']}" for i, p in others)
    return out


@tool
def list_registry_papers() -> str:
    """List the papers found for the current topic, showing which are selected (downloaded) and which are not."""
    if not REGISTRY:
        return "The registry is empty."
    return registry_block(REGISTRY)


@tool
def download_paper(index: int) -> str:
    """Download a paper by its number in the NOT SELECTED list (1-based registry number) and add it to the selected papers."""
    global _rag_dirty
    if not 1 <= index <= len(REGISTRY):
        return f"Invalid index. Valid: 1-{len(REGISTRY)}"
    p = REGISTRY[index - 1]
    if p["id"] in _downloaded_ids:
        return f"\"{p['title']}\" (arXiv:{p['id']}) is already downloaded and selected."
    try:
        print(f"\nDownloading from chat: {p['title']} (arXiv:{p['id']})")
        path = download_from_registry(p)
    except Exception as e:
        return f"Failed to download \"{p['title']}\": {e}"
    try:
        # Rebuild now, so the answer that follows this tool call can already search the new paper.
        print("   rebuilding the search index...")
        reindex_sync()
        _rag_dirty = False
    except Exception as e:
        _rag_dirty = True  # retry on the next question
        return (
            f"Downloaded \"{p['title']}\" (arXiv:{p['id']}) to {path}, but indexing failed ({e}). "
            "It will be retried on the user's next question."
        )
    return (
        f"Downloaded and indexed \"{p['title']}\" (arXiv:{p['id']}). "
        "It is now selected; its passages are available for answering."
    )


# ---------------------------------------------------------------------------
# 6. Q&A graph (scope + retrieval injected on every call)
# ---------------------------------------------------------------------------
SYSTEM = """You are an AI research assistant. You answer ONLY about the SELECTED PAPERS listed below.

RULES:
1. Use only the RETRIEVED PASSAGES and the SELECTED PAPERS summaries below. Never use outside knowledge.
2. Papers under NOT SELECTED are off-limits. If asked about them, say they are not downloaded and offer to download them.
3. If the passages do not contain the answer, say: "The selected papers do not contain information to answer this."
4. "The paper" / "this paper" means the selected paper(s). If one is selected, answer about it. Never ask the user which paper they mean.
5. Cite every claim as [Title, arXiv:ID, p.N]. Never invent IDs or titles.
6. To download a paper, call download_paper with its registry number.
7. Passages marked [FIGURE DESCRIPTION] describe a figure; passages marked [TABLE] hold a table's exact content.
   Use them for questions about figures, tables, charts, diagrams, numbers, or visual results.
   "Text inside the figure" and "Table content" are exact; prefer them over the generated "Description"
   for names and numbers. Copy numbers exactly as they appear in the table.
   Passages whose label contains "linked:" were added because they discuss a retrieved figure/table, or are
   the figure/table a retrieved paragraph mentions. Read them together to explain figures in the paper's context.
8. Do not guess the contents of an image from its filename or file path alone.
9. Format: one-sentence direct answer first, then bullet points, each ending with its citation.
10. If any part of your answer comes from a FIGURE DESCRIPTION, a TABLE, or an image, add that image's tag exactly
    as written in the passages at the end of that bullet, e.g. [IMG: 2404.01335/page_3_image_1.png].
    Only use tags that appear in the RETRIEVED PASSAGES; never invent one.
"""

_llm_with_tools = None


def trim_history(messages: List[BaseMessage], keep: int = 12) -> List[BaseMessage]:
    """Keep recent messages, but never start in the middle of a tool exchange."""
    msgs = [m for m in messages if m.type != "system"]
    tail = msgs[-keep:]
    while tail and tail[0].type != "human":
        tail = tail[1:]
    if not tail:
        idx = max((i for i, m in enumerate(msgs) if m.type == "human"), default=0)
        tail = msgs[idx:]
    return tail


async def researcher_node(state: AgentState):
    global _last_images
    # Retrieve in code, restricted to the selected papers, before the model answers.
    question = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    context, _last_images = "", {}
    if _downloaded_ids and isinstance(question, str) and question.strip():
        context, _last_images = await asyncio.to_thread(retrieve_passages, question, 5)

    system = SystemMessage(
        content=SYSTEM
        + "\n\n" + registry_block(state.get("papers", []))
        + ("\n\n" + context if context else "")
    )
    response = await _llm_with_tools.ainvoke([system] + trim_history(state["messages"]))
    return {"messages": [response]}


def build_graph():
    global _llm_with_tools
    tools = [list_registry_papers, download_paper, get_paper_images]
    _llm_with_tools = llm.bind_tools(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges("researcher", tools_condition)
    workflow.add_edge("tools", "researcher")
    return workflow.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Returning images that the answer actually used
# ---------------------------------------------------------------------------
def show_answer_images(text: str) -> List[str]:
    """Find [IMG: id] tags in the answer, keep only those that were really
    retrieved for this question, print their paths and optionally open them.
    Returns the list of image paths (useful for a Streamlit/Gradio UI)."""
    ids = list(dict.fromkeys(IMG_TAG.findall(text)))
    shown: List[str] = []
    if not ids:
        return shown

    print("\n [Images used in this answer]:")
    for image_id in ids:
        path = _last_images.get(image_id)
        if not path:  # model invented a tag or it was not retrieved -> reject
            print(f"   - {image_id}: ignored (not in retrieved passages)")
            continue
        print(f"   - {image_id}\n     {path}")
        shown.append(path)
        if OPEN_IMAGES:
            try:
                Image.open(path).show()
            except Exception as e:
                print(f"     [could not open: {e}]")
    return shown


def print_event(event: dict) -> None:
    msgs = event.get("messages") or []
    if not msgs:
        return
    last = msgs[-1]
    if last.type != "ai":
        return
    if last.tool_calls:
        for tc in last.tool_calls:
            print(f"\n [Agent] calling tool -> {tc['name']} {json.dumps(tc['args'], ensure_ascii=False)}")
    else:
        content = last.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        print(f"\n [Agent]:\n{content}")
        show_answer_images(content)


async def ask(app, query: str, topic: str, config: dict) -> None:
    global _rag_dirty
    if _rag_dirty and _session_chunks:
        print("\n--- Re-indexing papers ---")
        await asyncio.to_thread(reindex_sync)
        _rag_dirty = False

    async for event in app.astream(
        {"messages": [HumanMessage(content=query)], "papers": REGISTRY, "research_topic": topic},
        config,
        stream_mode="values",
    ):
        print_event(event)


# ---------------------------------------------------------------------------
# 7. Main flow
# ---------------------------------------------------------------------------
async def main():
    global _rag_dirty

    topic = ""
    while not topic:
        topic = input("\nEnter a research topic to search for: ").strip()

    print(f"\nSearching for papers on: {topic}")
    papers = await asyncio.to_thread(find_papers, topic)
    if not papers:
        print("No relevant papers found. Try different wording or lower MIN_SCORE.")
        return

    print("\nWriting summaries...")
    papers = await asyncio.to_thread(summarize_papers, papers)
    trends = await asyncio.to_thread(generate_trends, papers)

    REGISTRY.clear()
    REGISTRY.extend(papers)

    print("\n" + render_papers(papers, trends))

    choice = input("\nSelect papers to download (numbers like 1,3,5-7 or part of a title): ").strip()
    selected = parse_selection(choice, REGISTRY)
    if not selected:
        print("No valid selection. Exiting.")
        return

    for p in selected:
        print(f"\nDownloading: {p['title']} (arXiv:{p['id']})")
        try:
            await asyncio.to_thread(download_from_registry, p)
            print("   done")
        except Exception as e:
            print(f"   FAILED: {e}")

    if not _session_chunks:
        print("No papers were processed. Exiting.")
        return

    print("\n--- Initializing RAG ---")
    await asyncio.to_thread(reindex_sync)
    _rag_dirty = False

    app = build_graph()
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 15}

    print("\n=======================================================")
    print("Paper RAG is ready. Ask questions about the selected papers.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=======================================================")

    while True:
        query = input("\n[You]: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if query:
            await ask(app, query, topic, config)


if __name__ == "__main__":
    asyncio.run(main())