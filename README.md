# Paper Research Agent

Search arXiv and Semantic Scholar for a research topic, pick the papers you care about, and chat with them inside a Jupyter notebook. Every answer cites the paper and page it came from. When an answer is based on a **figure or table**, the chat **shows that image** under the answer.

Everything runs locally on [Ollama](https://ollama.com). No paid API key is needed.

---

## Part 1: Running and using the chat

### 1. Requirements

| What | Why |
|---|---|
| Python 3.10+ and Jupyter (or VS Code with the Jupyter extension) | runs the chat UI |
| Ollama, running on `http://localhost:11434` | all LLMs and embeddings |
| A GPU (recommended) | `qwen2.5:14b`, `llava`, and Docling run much faster on a GPU |
| Internet access | searching for papers and downloading PDFs |

### 2. Pull the Ollama models

```bash
ollama pull qwen2.5:14b           # chat agent + figure/table explanations
ollama pull qwen2.5:7b-instruct   # paper summaries and key trends
ollama pull nomic-embed-text      # embeddings (relevance ranking + search)
ollama pull llava:latest          # vision model that looks at the figures
```

Check that Ollama is up with `ollama list`.

### 3. Install the Python packages

```bash
cd "paper implemetation"
pip install -r requirements.txt
```

> `requirements.txt` includes `docling`, which gives much better figure and table detection but is a large install (it pulls in PyTorch). If you remove it, figures are found by a simpler PyMuPDF detector, and table cells are not extracted.

Optional: to get higher Semantic Scholar rate limits, add your key to a `.env` file in the project folder:

```
SEMANTIC_SCHOLAR_API_KEY=...
```

### 4. Start the chat

1. Make sure Ollama is running.
2. Open **`chat_ui.ipynb`** in VS Code or Jupyter.
3. Run its cell (Shift+Enter):

```python
from notebook_ui import show
ui = show()
```

The chat opens inside the notebook. There is no web server and nothing to port-forward. If the notebook is on a remote server, it works over the same connection VS Code already uses.

### 5. Using the chat

The chat UI has three sections, top to bottom.

**1. Search**

- Type a topic (for example *"diffusion models for medical imaging"*) and press **Search papers**.
- You can change these options before searching:

| Option | Effect |
|---|---|
| LLM summaries | On: an LLM writes a 1–2 sentence summary of each paper, plus key trends. Off: uses each abstract's first sentence (instant). |
| Max papers | How many papers to keep after ranking (3–20, default 10). |
| Min relevance | Similarity cut-off (default 0.40). Lower it if too few papers come back. |
| **Describe figures with LLM** | **Keep this on if you want to ask about figures.** Each figure is described by llava and explained by qwen2.5:14b. It is slow, but it is what makes figures searchable. |

- The **Progress log** shows each step live. The status line under the options tells you what to do next.

**2. Choose papers**

- Each result shows its title, authors, year, arXiv link, relevance score, and summary. The key trends appear above the list.
- Tick the papers you want and press **Download & index selected**. Each PDF is downloaded, its text, figures, and tables are extracted, and it is added to the search index. A ✅ appears next to each paper that is ready.

**3. Chat**

- Type a question and press **Enter** or **Send**. The chat box is enabled once at least one paper is indexed.
- Each answer shows:
  - a one-sentence direct answer, followed by bullet points;
  - a citation on every bullet, in the form `[Title, arXiv:ID, p.N]`;
  - **the figure or table images the answer used**, displayed under the text with their file name (for example `page_4_image_1.png`);
  - a *Tools used* line when the agent called a tool, for example to download a paper.

### 6. Asking about figures and tables

When a question is about something visual, the agent finds the matching figure or table and shows it together with the explanation.

Questions that return images:

- *"Explain the overview figure of the method."*
- *"What does Figure 3 show?"*
- *"Show me the architecture diagram."*
- *"What are the results in Table 2? Which method is best?"*
- *"Which figure compares the baselines?"*

Example of what you get:

```
🤖 Agent:
The overview figure shows a two-stage pipeline that first generates a monocular
video and then fills disoccluded regions in a frame matrix.

- A text prompt is turned into a monocular video, which is warped into fixed camera
  views using estimated depth [SVG: 3D Stereoscopic Video Generation…, arXiv:2407.00367, p.4]
- Disoccluded regions are inpainted by denoising along the frame matrix [… p.4]

   ┌──────────────────────────┐
   │   (the figure image)     │
   └──────────────────────────┘
   page_4_image_1.png
```

Tips for figure questions:

- Mention the figure or table number, or what it shows ("architecture", "results table", "ablation plot").
- Numbers in table answers are copied from the table's real cells, not read off the image by a model.
- An image is shown **only** when the answer actually used it. If no figure matches, you get a text answer without images.
- If **Describe figures** was off when a paper was indexed, its figures are still searchable by caption and the text inside them, but the answers are weaker. To fix this, turn it back on, run the cell again, and re-index the paper.

### 7. Other things you can ask

- *"What problem does this paper solve?"* or *"Compare the methods of the selected papers."*
- *"Which papers have I not selected?"*
- *"Download paper 4"*: downloads and indexes a paper you skipped, straight from the chat.

The agent only answers from the papers you **selected**. If you ask about another paper, it offers to download it instead. If the papers don't contain the answer, it says so rather than guessing.

A new search starts fresh: it clears the previous papers, index, and chat.

### 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to Ollama` | Start Ollama (`ollama serve`) and check that the models above are pulled. |
| The widgets don't render | Install `ipywidgets` in the notebook's kernel, then restart the kernel. |
| No papers found | Use broader wording, or lower *Min relevance*. |
| `arXiv rate limited (429/503)` | Normal. The app waits and retries automatically. |
| Indexing is very slow | Each figure goes through llava and then qwen2.5:14b. Untick *Describe figures* for speed, at the cost of weaker figure answers. |
| No images in answers | Ask explicitly about a figure or table, and check in the progress log that figures were found when the paper was indexed. |
| Docling fails or there is no GPU | Set `DOCLING_DEVICE = "cpu"` in `agents/Paper_RAG.py`. |

---

## Part 2: How it works (technical)

### Architecture

```
  chat_ui.ipynb  →  notebook_ui.py          ipywidgets chat UI (no logic)
                         │
                  agents/orchestra.py        search · summaries · registry · LangGraph agent
                         │
                  agents/Paper_RAG.py        PDF → text + figures/tables → chunks → FAISS → retrieval
                         │
                  Ollama (localhost:11434)
                  qwen2.5:14b · qwen2.5:7b-instruct · llava · nomic-embed-text
```

| File | Role |
|---|---|
| `chat_ui.ipynb` | Entry point. A single cell calls `notebook_ui.show()`. |
| `notebook_ui.py` | The widget UI. Runs the async agent on a private background event loop (Jupyter already runs its own loop), streams logs into the progress panel, and displays answer images with `ipywidgets.Image`. |
| `agents/orchestra.py` | Paper search, ranking, summaries, the paper registry, the agent tools, the LangGraph graph, and image-tag checking. |
| `agents/Paper_RAG.py` | PDF processing, figure/table detection and description, chunking, the FAISS vector store, and retrieval. |
| `downloaded_papers/` | Cache: `<id>.pdf`, `<id>.meta.json`, `<id>_chunks.json`, `<id>_extracted/images/page_N_image_M.png`. |

### Step 1: Search and ranking

Search is done in code, not by the LLM, so the model cannot invent papers.

1. The topic is split into keywords. Up to three arXiv queries and one Semantic Scholar query are sent (Semantic Scholar results without an arXiv ID are dropped). arXiv calls are spaced at least 3.5 s apart and retried on 429/503 responses.
2. Results are de-duplicated by arXiv ID and by title.
3. The topic and each `title + abstract` are embedded with `nomic-embed-text`. Papers are ranked by cosine similarity, those below `MIN_SCORE` are dropped, and the top `TARGET_PAPERS` are kept.
4. `qwen2.5:7b-instruct` uses structured output (Pydantic) to write the summaries (in batches of 5) and the key trends. If a call fails, the summary falls back to the abstract's first sentence.

The results are stored in `REGISTRY`. Papers are referenced only by their registry number, so the model can never make up an arXiv ID.

### Step 2: PDF processing

1. **Download** from `export.arxiv.org` (falling back to `arxiv.org`). The code checks that the response is a real PDF. PDFs already on disk are reused.
2. **Text.** `pymupdf4llm` converts each page to Markdown.
3. **Figures and tables** go through the image pipeline described in the next section.
4. **Chunking.** `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap) creates text chunks for each page. Every figure or table also gets **one dedicated image chunk**.
5. The chunks are saved to `<id>_chunks.json`, embedded with `nomic-embed-text` into an in-memory **FAISS** index, and retrieval is limited to the selected papers.

### The image pipeline (from PDF figure to image in the chat)

This is how a figure in a PDF ends up displayed under an answer.

```
 PDF page
   │  ① detect          Docling layout model (or the geometric PyMuPDF fallback)
   ▼
 figure/table region + caption + OCR text + table cells
   │  ② render          crop → PNG  (page_N_image_M.png, ≤1024 px)
   ▼
 image file in downloaded_papers/<id>_extracted/images/
   │  ③ describe        stage 1: llava "what do you see"
   │                    stage 2: qwen2.5:14b explains it using the paper's own text
   ▼
 image chunk  (description + image_id "<id>/page_N_image_M.png")
   │  ④ link            "as shown in Figure 2" ↔ Figure 2's chunk
   ▼
 FAISS index
   │  ⑤ retrieve        question → top passages + linked figures/text
   │                    each passage lists its [IMG: id] tags
   ▼
 LLM answer containing [IMG: 2407.00367/page_4_image_1.png]
   │  ⑥ verify          the tag is kept only if that image was retrieved for THIS question
   ▼
 image displayed under the answer
```

**① Detection.** Docling runs once per PDF, on the GPU, with OCR and table-structure extraction. It returns the box, caption, and kind (`figure` or `table`) of each item. For tables it also returns the cells as Markdown, and for figures the text inside them. Sub-panels without a caption that sit close to a captioned figure are merged into it. If Docling is missing or fails, a geometric detector merges nearby drawing and image rectangles on each page and searches for a "Figure N" or "Table N" caption.

**② Rendering.** Each whole figure (not its fragments) is cropped with a small margin and rendered to PNG, with the longest side at most 1024 px and at most ~216 dpi. Figures smaller than 120×25 pt are skipped. Images left over from earlier runs are deleted first.

**③ Two-stage description.** The two stages are kept separate because small vision models get confused by long prompts.
- *Stage 1 (llava).* llava gets only the image, the caption, and the OCR labels, and is asked to describe what it can **see**: the type of figure, its parts, arrows, axes, and labels. Tables skip this stage, because their exact cells are already known.
- *Stage 2 (qwen2.5:14b).* qwen first writes one ~120-word summary of the paper, from the start of the paper plus its conclusion. Then, for each figure, it combines the **exact facts** (caption, text inside the figure, table cells, and up to 4 sentences of the paper that mention "Figure N") with llava's description into a 3–5 sentence explanation. It is told to prefer the paper's words over the vision model's.
- All llava calls run before any qwen calls, so Ollama doesn't keep swapping models.
- If a model fails, the chunk keeps the exact facts anyway, so the figure stays searchable.

**Image chunk example** (`2407.00367_chunks.json`):

```json
{
  "chunk_type": "image",
  "figure_kind": "figure",
  "image_id": "2407.00367/page_4_image_1.png",
  "page_number": 4,
  "caption": "Figure 1: Overview – Top: Given a text prompt, our method first uses a video generation model…",
  "text": "Figure from \"SVG: 3D Stereoscopic Video Generation…\", page 4 (image id: 2407.00367/page_4_image_1.png).\n[Image: page_4_image_1.png] Caption: …\nWhat the paper says about it: …\nExplanation: …",
  "linked_chunks": ["2407.00367_p2_c2", "2407.00367_p3_c7", "2407.00367_p4_c1"]
}
```

The image ID has the form `<pdf_name>/<filename>`, so it is unique across papers. Two papers can both have a `page_4_image_1.png` without being confused.

**④ Figure ↔ text links.** Every text chunk that mentions "Figure 2", "Fig. 2", or "Table 3" is linked to that figure's chunk, and the link is stored in both directions (`linked_images` on the text chunk, `linked_chunks` on the figure chunk). Each page's text also ends with a short list of the figures on that page, so plain text chunks know which images belong to them.

**⑤ Retrieval.** For every question, the code (not the model) searches FAISS for the top 5 passages from the selected papers. It then follows the links and adds up to 3 more passages:
- a retrieved **figure** brings in the paragraphs where the authors discuss it;
- a retrieved **paragraph** brings in the figure or table it mentions.

Each passage is labelled `[TEXT]`, `[FIGURE DESCRIPTION]`, or `[TABLE]` (with `linked:` when it was added through a link). Under each passage is a list of its images:

```
--- [FIGURE DESCRIPTION] Source: "SVG: …" (arXiv:2407.00367), page 4 ---
Figure from "SVG: …", page 4 …
IMAGES FOR THIS PASSAGE (if your answer uses one, write its tag exactly):
- [IMG: 2407.00367/page_4_image_1.png]
```

At the same time, the code records the map `{image_id → file path}` for **only** the images in these passages (`_last_images`).

**⑥ Verification and display.** The system prompt tells the model to put an image's tag at the end of every bullet that uses a figure or table, to copy table numbers exactly, and never to guess an image's content from its file name. After the model answers:
1. `show_answer_images()` finds every `[IMG: …]` tag in the answer.
2. A tag is kept only if it appears in `_last_images`, the images retrieved for **this** question. Invented tags, or tags from an earlier question, are dropped.
3. The tags are removed from the displayed text, and each verified image is shown under the answer with `ipywidgets.Image` (max width 600 px), followed by its file name.

Because the tags must come from the retrieved passages, an image in the chat always matches something in the cited paper and page.

### Step 3: The chat agent (LangGraph)

```
START → researcher ──(tool call?)──► tools ──┐
            ▲                                │
            └────────────────────────────────┘
            └──(no tool call)──► END
```

- **State:** `messages`, `research_topic`, and `papers`. A `MemorySaver` keeps the chat history for each session.
- **The `researcher` node** retrieves passages and images (step ⑤ above). It then builds the system prompt from the rules, the *SELECTED* papers with their summaries, the *NOT SELECTED* papers listed by title only, and the passages, and calls `qwen2.5:14b` (temperature 0, 8k context). History is trimmed to about the last 12 messages.
- **Tools:**
  - `download_paper(index)`: downloads, processes, and re-indexes a paper mid-chat.
  - `list_registry_papers()`: shows which papers are selected and which are not.
  - `get_paper_images(page_number)`: lists the extracted figures with their `[IMG: id]` tags.
- `recursion_limit = 15` stops runaway tool loops.

### Grounding guards

| Guard | How |
|---|---|
| No invented papers | Search and ranking are done in Python. Downloads happen only by registry number. |
| Answers only from selected papers | A FAISS metadata filter on arXiv ID, plus the prompt rules. |
| Citations | Every passage header includes the title, arXiv ID, and page. |
| No invented or wrong images | An `[IMG]` tag is shown only if its image was retrieved for the current question. |
| Exact table numbers | Table chunks hold the real cell values from Docling, and the prompt says to copy them. |
| Robustness | Timeouts, retries, and fallbacks throughout: abstract sentence for summaries, raw facts for figure descriptions, geometric detector instead of Docling. |

### Main configuration knobs

| Constant | File | Default |
|---|---|---|
| `TARGET_PAPERS`, `MIN_SCORE` | `orchestra.py` | `10`, `0.40` |
| `USE_LLM_SUMMARIES`, `DESCRIBE_IMAGES` | `orchestra.py` | `True`, `True` |
| Chat / summary / embedding models | `orchestra.py` | `qwen2.5:14b` / `qwen2.5:7b-instruct` / `nomic-embed-text` |
| `FIGURE_DETECTOR` | `Paper_RAG.py` | `"docling"` (or `"geometric"`) |
| `DOCLING_DEVICE`, `DOCLING_DO_OCR`, `DOCLING_INCLUDE_TABLES` | `Paper_RAG.py` | `"cuda"`, `True`, `True` |
| `USE_TWO_STAGE` (llava + qwen explanation) | `Paper_RAG.py` | `True` |
| `FIG_MAX_PX`, `FIG_MAX_ZOOM` | `Paper_RAG.py` | `1024`, `3.0` |
| `MAX_LINKED_PASSAGES` | `Paper_RAG.py` | `3` |
| Chunk size / overlap | `Paper_RAG.py` | `1000` / `200` |

### Current limitations

- One notebook kernel serves one user and one topic at a time, because state is kept in module globals.
- The FAISS index lives in memory and is rebuilt after each download. Chunk files and images are cached on disk.
- Only papers that have an arXiv ID can be downloaded.
- Figure answers depend on the quality of detection. Unusual layouts can split one figure into pieces or merge two figures into one.
