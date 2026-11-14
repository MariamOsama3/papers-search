import pymupdf4llm
import pymupdf as fitz
import base64
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import io
import re
import json
import time
import sys
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
IMG_NAME_RE = r"page_\d+_image_\d+\.[a-zA-Z0-9]+"

# ---------------------------------------------------------------------------
# Module-level globals - populated by initialize_rag()
# ---------------------------------------------------------------------------
vectorstore = None
_images_dirs: List[str] = []
_images_dirs_by_pdf: Dict[str, Path] = {}
_allowed_ids: set = set()   # arXiv ids the retriever is allowed to return (the selected papers)
_chunks_by_id: Dict[str, Dict] = {}        # chunk_id -> chunk (to follow figure <-> text links)
_image_chunks: Dict[str, Dict] = {}        # image_id -> its figure/table chunk
MAX_LINKED_PASSAGES = 3                    # extra passages added per question by following links


def set_allowed_papers(ids) -> None:
    """Restrict retrieval to these arXiv ids. Pass an empty set to disable the filter."""
    global _allowed_ids
    _allowed_ids = set(ids)


# ---------------------------------------------------------------------------
# Paper metadata (title / arXiv id) - the key to source awareness
# ---------------------------------------------------------------------------
def save_paper_meta(pdf_path: str, title: str, arxiv_id: str = "",
                    authors: str = "", year: int = 0) -> None:
    """Call this from your download tool so every PDF gets a <stem>.meta.json
    next to it. The RAG pipeline reads it to attach real titles to chunks."""
    meta_path = Path(pdf_path).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"title": title, "arxiv_id": arxiv_id, "authors": authors, "year": year},
            f, indent=2, ensure_ascii=False,
        )


def load_paper_meta(pdf_path: str, first_page_text: str = "") -> Dict[str, str]:
    """Resolve title + arXiv id for a PDF.
    Priority: .meta.json -> PDF embedded metadata -> first heading on page 1 -> filename."""
    stem = Path(pdf_path).stem
    meta_path = Path(pdf_path).with_suffix(".meta.json")

    title, arxiv_id = "", ""

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = (data.get("title") or "").strip()
            arxiv_id = (data.get("arxiv_id") or "").strip()
        except Exception:
            pass

    if not arxiv_id:
        m = re.match(r"(\d{4}\.\d{4,5})(?:v\d+)?$", stem)
        arxiv_id = m.group(1) if m else "unknown"

    if not title:
        try:
            with fitz.open(pdf_path) as doc:
                embedded = (doc.metadata or {}).get("title", "").strip()
            if len(embedded) > 5:
                title = embedded
        except Exception:
            pass

    if not title and first_page_text:
        for line in first_page_text.splitlines():
            clean = line.strip().lstrip("#").replace("**", "").strip()
            if len(clean) > 8:
                title = clean[:200]
                break

    if not title:
        title = stem

    return {"paper_title": title, "arxiv_id": arxiv_id}


# ---------------------------------------------------------------------------
# Image ids: "<pdf_name>/<filename>" is unique across papers
# ---------------------------------------------------------------------------
def make_image_id(pdf_name: str, filename: str) -> str:
    return f"{pdf_name}/{filename}"


def resolve_image_path(pdf_name: str, image_name: str) -> Optional[Path]:
    """Resolve an extracted image name to the image belonging to its PDF."""
    images_dir = _images_dirs_by_pdf.get(pdf_name)
    if not images_dir:
        return None
    image_path = images_dir / image_name
    return image_path if image_path.exists() else None


def resolve_image_by_id(image_id: str) -> Optional[Path]:
    pdf_name, _, fname = image_id.strip().partition("/")
    return resolve_image_path(pdf_name, fname)


def is_valid_description(desc: str) -> bool:
    if not desc or "description skipped" in desc:
        return False
    bad_markers = ("[Error", "[ERROR", "[Timeout", "[No description")
    return not any(b in desc for b in bad_markers)


# ---------------------------------------------------------------------------
# Ollama helpers (image description)
# ---------------------------------------------------------------------------
def check_ollama_status(model: str = "llava:latest", base_url: str = OLLAMA_URL) -> bool:
    try:
        print("Checking Ollama connection...")
        response = requests.get(f"{base_url}/api/tags", timeout=60)
        response.raise_for_status()

        available_models = [m.get("name") for m in response.json().get("models", [])]

        if any(model in m for m in available_models):
            print(f"   OK: Ollama is running and '{model}' is available.")
            return True
        print(f"   WARNING: model '{model}' not found.")
        print(f"   Available models: {', '.join(available_models)}")
        print(f"   Please run: ollama pull {model}")
        return False

    except requests.exceptions.RequestException:
        print("   ERROR: Could not connect to Ollama. Please start it first.")
        return False


def describe_image_with_ollama(
    image_bytes: bytes,
    model: str = "llava:latest",
    ollama_url: str = f"{OLLAMA_URL}/api/generate",
    timeout: int = 180,
    max_retries: int = 2,
    caption: str = "",
    kind: str = "figure",
    figure_text: str = "",
    table_md: str = "",
) -> str:
    """Stage 1 of the two-stage description: the VLM reports ONLY what it can see.
    The paper context (summary, referencing paragraphs) is added in stage 2 by the
    text model, so this prompt stays short - small VLMs get confused by long prompts."""
    try:
        prompt = (
            "Describe only what you can SEE in this research paper figure: its type "
            "(diagram, plot, chart, example images...), the main parts, how they are "
            "connected (arrows, order, grouping), axes/legends, and visible labels. "
            "Be factual and concise; do not guess what the paper claims."
        )
        if caption:
            prompt += f"\nCaption: \"{caption[:300]}\""
        if figure_text:
            prompt += f"\nLabels read from the figure (exact spelling): {figure_text[:400]}"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64.b64encode(image_bytes).decode("utf-8")],
            "stream": False,
            "options": {"num_ctx": 4096, "temperature": 0.1, "num_predict": 400},
        }

        for attempt in range(max_retries):
            try:
                print(f"\n Processing... (attempt {attempt + 1}/{max_retries})", end="\r")
                response = requests.post(ollama_url, json=payload, timeout=timeout)
                response.raise_for_status()
                description = response.json().get("response", "").strip()
                return description if description else "[No description generated]"
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "[Timeout: Image processing took too long.]"

        return "[Error: no response]"

    except requests.exceptions.ConnectionError:
        return "[ERROR: Ollama connection lost during processing]"
    except Exception as e:
        return f"[Error: {e}]"


# ---------------------------------------------------------------------------
# Stage 2 / paper context: text model (sees the whole paper context, no image)
# ---------------------------------------------------------------------------
TEXT_MODEL = "qwen2.5:14b"       # same model as the Q&A agent, so Ollama keeps one copy loaded
TEXT_NUM_CTX = 8192              # same context size as the agent -> no reload between them
TEXT_TIMEOUT = 300               # seconds per call
USE_TWO_STAGE = True             # False = keep only the VLM's visual description (faster)
MAX_REF_SENTENCES = 4            # sentences from the paper that mention a figure, per figure
MAX_REF_CHARS = 1500


def generate_text(prompt: str, num_predict: int = 400, model: str = TEXT_MODEL) -> str:
    """Call the local text model. Returns '' on any failure (callers fall back)."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"num_ctx": TEXT_NUM_CTX, "temperature": 0.1,
                              "num_predict": num_predict}},
            timeout=TEXT_TIMEOUT,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        print(f"   [text model call failed: {type(e).__name__}: {e}]")
        return ""


def _section(text: str, names: tuple, limit: int) -> str:
    """Return up to `limit` chars starting at the first heading whose name matches."""
    for m in re.finditer(r"(?im)^\s*(?:#+\s*|\**)?(?:\d+\.?\s*)?(" + "|".join(names) + r")\b.*$", text):
        return text[m.start(): m.start() + limit]
    return ""


def summarize_paper(pages_text: List[Tuple[int, str]], title: str) -> str:
    """One short summary of the whole paper, written once and reused for every figure.
    Uses the beginning (abstract + introduction) and the conclusion - enough to know
    the goal and the method without sending the whole paper."""
    full = "\n".join(t for _, t in pages_text)
    if not full.strip():
        return ""
    start = full[:6000]
    end = _section(full, ("conclusions?", "discussion", "summary"), 2500) or full[-2000:]
    prompt = (
        f"Paper title: {title}\n\nBEGINNING OF THE PAPER:\n{start}\n\nCONCLUSION:\n{end}\n\n"
        "Write a summary of this paper in at most 120 words: the problem, the proposed "
        "method and its main components, and the main result. Use only the text above."
    )
    print(" Writing a paper summary for figure context...")
    t0 = time.time()
    summary = generate_text(prompt, num_predict=300)
    print(f"   -> {'done' if summary else 'failed'} in {time.time() - t0:.0f}s")
    return summary


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# a caption line: label + number + ':' or '.'  (e.g. "Figure 2: ...", "Table 1. ...")
CAPTION_LINE_RE = re.compile(r"^\s*(fig\.?|figure|table|tab\.)\s*\d+\s*[:.]", re.IGNORECASE)


def _figure_number(caption: str) -> Optional[Tuple[str, int]]:
    """'Figure 3: ...' -> ('figure', 3); 'Table 2. ...' -> ('table', 2)."""
    m = re.match(r"^\s*(fig\.?|figure|table|tab\.)\s*(\d+)", caption or "", re.IGNORECASE)
    if not m:
        return None
    kind = "table" if m.group(1).lower().startswith("tab") else "figure"
    return kind, int(m.group(2))


def _mention_re(kind: str, num: int) -> "re.Pattern":
    words = r"(?:Tables?|Tabs?\.)" if kind == "table" else r"(?:Figures?|Figs?\.)"
    return re.compile(rf"\b{words}\s*{num}(?!\d)", re.IGNORECASE)


def find_references(caption: str, pages_text: List[Tuple[int, str]]) -> Tuple[str, List[int]]:
    """Sentences in the paper body that mention this figure/table (not the caption itself)."""
    label = _figure_number(caption)
    if not label:
        return "", []
    pattern = _mention_re(*label)
    found, pages = [], []
    for page_no, text in pages_text:
        # split into paragraphs first (blank lines / headings), so headings never glue onto sentences
        sentences = []
        for para in re.split(r"\n\s*\n|\n(?=\s*#)", text):
            para = re.sub(r"^\s*#+\s*.*$", "", para, flags=re.MULTILINE)   # drop heading lines
            flat = " ".join(para.replace("**", "").split())
            if flat:
                sentences.extend(_SENT_SPLIT.split(flat))
        for sent in sentences:
            # skip captions ("Figure 2: ...", "Table 1. ..."), but keep body sentences
            # that merely start with the label ("Table 1 shows that ...")
            if not pattern.search(sent) or CAPTION_LINE_RE.match(sent):
                continue
            if caption and sent[:60] in caption:          # the caption repeated in the text
                continue
            found.append(sent[:500])
            pages.append(page_no)
            if len(found) >= MAX_REF_SENTENCES:
                break
        if len(found) >= MAX_REF_SENTENCES:
            break
    refs = " ... ".join(found)[:MAX_REF_CHARS]
    return refs, sorted(set(pages))


def _build_facts(img: Dict) -> str:
    """Exact, non-generated information about a figure/table (kept even if models fail)."""
    kind = img.get("kind", "figure")
    facts = f"[{'Table' if kind == 'table' else 'Image'}: {img['filename']}]"
    if img.get("caption"):
        facts += f" Caption: {img['caption'].rstrip('.')}."
    if img.get("figure_text"):
        facts += f"\nText inside the figure: {img['figure_text']}"
    if img.get("table_md"):
        facts += f"\nTable content:\n{img['table_md']}"
    if img.get("references"):
        facts += f"\nWhat the paper says about it: {img['references']}"
    return facts


def enrich_figures(images: List[Dict], pages_text: List[Tuple[int, str]],
                   title: str, use_llm: bool) -> str:
    """Stage 2 for every figure/table of one paper:
    1. find the sentences that reference it (always - cheap, no model);
    2. write ONE paper summary;
    3. the text model fuses summary + caption + exact text/table + references +
       the VLM's visual description into the final explanation.
    All text-model calls run here, after all VLM calls, so Ollama doesn't swap
    llava <-> qwen for every figure. Returns the paper summary."""
    for img in images:
        img["references"], img["ref_pages"] = find_references(img.get("caption", ""), pages_text)

    summary = ""
    if use_llm and USE_TWO_STAGE and images:
        summary = summarize_paper(pages_text, title)

    for n, img in enumerate(images, 1):
        facts = _build_facts(img)
        img["facts"] = facts
        visual = img.get("visual_description", "")
        visual_ok = is_valid_description(visual)

        final = ""
        if use_llm and USE_TWO_STAGE:
            kind = img.get("kind", "figure")
            print(f" Explaining {kind} {n}/{len(images)} ({img['filename']}) with the paper context...")
            t0 = time.time()
            prompt = (
                (f"PAPER SUMMARY: {summary}\n\n" if summary else f"PAPER TITLE: {title}\n\n")
                + f"{'TABLE' if kind == 'table' else 'FIGURE'} INFORMATION (exact):\n{facts}\n\n"
                + (f"WHAT A VISION MODEL SEES IN THE IMAGE:\n{visual}\n\n" if visual_ok else "")
                + ("Explain this table in 3-5 sentences: what it compares, the key rows/columns, "
                   "and the main result. Copy numbers exactly from the table content."
                   if kind == "table" else
                   "Explain this figure in 3-5 sentences: what it shows, how its parts relate to "
                   "the paper's method, and its key message.")
                + " Prefer the exact information and the paper's own words over the vision "
                  "model's description. Do not invent anything that is not given above."
            )
            final = generate_text(prompt, num_predict=350)
            print(f"   -> {'done' if final else 'failed, using the visual description'} "
                  f"in {time.time() - t0:.0f}s")

        if final:
            img["description"] = f"{facts}\nExplanation: {final}"
        elif visual_ok:
            img["description"] = f"{facts}\nDescription: {visual}"
        else:
            img["description"] = f"{facts}\nDescription: (AI description skipped)"
    return summary


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
CAPTION_RE = re.compile(r"^\s*(fig\.?|figure|table|tab\.)\s*\d+", re.IGNORECASE)

FIG_MERGE_GAP = 18        # points: pieces closer than this belong to the same figure
FIG_MARGIN = 6            # points of padding around a rendered figure
FIG_MAX_PX = 1024         # longest side of the rendered figure image
FIG_MAX_ZOOM = 3.0        # never render above ~216 dpi

# Figure detector: "docling" (layout model, recommended) or "geometric" (PyMuPDF only).
# If Docling is not installed or fails on a PDF, the geometric detector is used automatically.
FIGURE_DETECTOR = "docling"
DOCLING_DEVICE = "cuda"          # "cuda", "cuda:N" (a specific GPU, e.g. "cuda:3"), "cpu" or "auto"
DOCLING_INCLUDE_TABLES = True    # detect tables, render them, and extract their cells as a Markdown table
DOCLING_DO_OCR = True            # OCR text inside figures (labels in raster images / scanned parts)
CAPTION_ATTACH_GAP = 40          # points: a caption-less panel this close to a captioned figure joins it
MAX_TABLE_CHARS = 4000           # cap on table Markdown stored per table (keeps chunks embeddable)
MAX_FIGURE_TEXT_CHARS = 1500     # cap on text-inside-figure stored per figure


def _grow(r: "fitz.Rect", d: float) -> "fitz.Rect":
    return fitz.Rect(r.x0 - d, r.y0 - d, r.x1 + d, r.y1 + d)


def _merge_rects(rects: List["fitz.Rect"], gap: float) -> List["fitz.Rect"]:
    """Union rectangles that touch or lie within `gap` points of each other."""
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        rects.sort(key=lambda r: (r.y0, r.x0))
        out: List[fitz.Rect] = []
        for r in rects:
            g = _grow(r, gap)
            for k, o in enumerate(out):
                if g.intersects(o):
                    out[k] = fitz.Rect(min(o.x0, r.x0), min(o.y0, r.y0),
                                       max(o.x1, r.x1), max(o.y1, r.y1))
                    changed = True
                    break
            else:
                out.append(fitz.Rect(r))
        rects = out
    return rects


def _find_caption(page, region: "fitz.Rect", blocks) -> str:
    """Return the 'Figure N: ...' / 'Table N: ...' text block just below (or above) a region."""
    best, best_dist = "", 1e9
    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if not CAPTION_RE.match(text or ""):
            continue
        if x1 < region.x0 or x0 > region.x1:          # must overlap horizontally
            continue
        if y0 >= region.y1 - 2:                        # below the figure
            dist = y0 - region.y1
        elif y1 <= region.y0 + 2:                      # above (tables usually)
            dist = region.y0 - y1
        else:
            continue
        if 0 <= dist < 60 and dist < best_dist:
            best, best_dist = " ".join(text.split()), dist
    return best[:400]


def find_figure_regions(page, min_width: float = 120, min_height: float = 25) -> List[Dict]:
    """Locate whole figures on a page by grouping embedded images AND vector
    drawings that sit close together. A diagram made of 20 small pictures,
    boxes and arrows comes back as ONE region instead of 20 fragments."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    image_rects = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page_rect
        if not r.is_empty:
            image_rects.append(r)

    drawing_rects = []
    for d in page.get_drawings():
        raw = fitz.Rect(d["rect"])
        if raw.width == 0 and raw.height == 0:          # a single point
            continue
        # horizontal/vertical lines (arrows, axes) have zero height/width - give them thickness
        r = _grow(raw, 0.5)
        if not r.intersects(page_rect):
            continue
        r = r & page_rect
        if r.width * r.height > 0.8 * page_area:        # page background / frame
            continue
        if r.width > 0.9 * page_rect.width and r.height < 3:   # full-width rules / header lines
            continue
        drawing_rects.append(r)

    if not image_rects and not drawing_rects:
        return []

    regions = []
    for region in _merge_rects(image_rects + drawing_rects, FIG_MERGE_GAP):
        n_img = sum(1 for r in image_rects if region.contains(r) or region.intersects(r))
        n_draw = sum(1 for r in drawing_rects if region.intersects(r))
        if region.width < min_width or region.height < min_height:
            continue
        if n_img == 0 and n_draw < 8:                 # a lone box or a few table rules
            continue
        regions.append({"rect": region, "n_images": n_img, "n_drawings": n_draw})

    regions.sort(key=lambda g: (g["rect"].y0, g["rect"].x0))
    return regions


# ---------------------------------------------------------------------------
# Docling figure detection (layout model) + caption-anchored grouping
# ---------------------------------------------------------------------------
_docling_converter = None


def _union(a: "fitz.Rect", b: "fitz.Rect") -> "fitz.Rect":
    return fitz.Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _rect_gap(a: "fitz.Rect", b: "fitz.Rect") -> float:
    """Distance between two rectangles (0 if they touch or overlap)."""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return max(dx, dy)


def _get_docling_converter():
    """Create the Docling converter once (loading its layout model is the slow part)."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions

        opts = PdfPipelineOptions()
        opts.do_ocr = DOCLING_DO_OCR                      # read text inside bitmap areas (figures)
        opts.do_table_structure = DOCLING_INCLUDE_TABLES  # extract table rows/columns/cells
        # accepts "cuda", "cuda:N", "cpu", "auto" (Docling validates the string)
        opts.accelerator_options = AcceleratorOptions(device=DOCLING_DEVICE)
        print(f"   Loading Docling (device={DOCLING_DEVICE}, ocr={DOCLING_DO_OCR}, "
              f"tables={DOCLING_INCLUDE_TABLES})...")
        _docling_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _docling_converter


def _group_by_caption(regs: List[Dict]) -> List[Dict]:
    """One caption = one figure.
    - pieces that share a caption are merged;
    - caption-less panels next to a captioned figure join it (sub-figures (a), (b), (c));
    - two figures with different captions are never merged, however close they are;
    - remaining caption-less pieces are grouped by distance, so nothing is lost."""
    def absorb(target: Dict, piece: Dict) -> None:
        target["rect"] = _union(target["rect"], piece["rect"])
        if piece.get("table_md"):
            target["table_md"] = "\n\n".join(x for x in (target.get("table_md"), piece["table_md"]) if x)

    anchors: Dict[Tuple[str, str], Dict] = {}
    loose: List[Dict] = []
    for r in regs:
        if r["caption"]:
            key = (r["kind"], r["caption"])
            if key in anchors:
                absorb(anchors[key], r)
            else:
                anchors[key] = dict(r)
        else:
            loose.append(r)

    figures = list(anchors.values())
    attached = True
    while attached and loose:           # repeat: a grown figure may now reach another panel
        attached = False
        remaining = []
        for r in loose:
            near = [f for f in figures if f["kind"] == r["kind"]
                    and _rect_gap(f["rect"], r["rect"]) <= CAPTION_ATTACH_GAP]
            if near:
                best = min(near, key=lambda f: _rect_gap(f["rect"], r["rect"]))
                absorb(best, r)
                attached = True
            else:
                remaining.append(r)
        loose = remaining

    # caption-less tables stay one-per-table; caption-less figure pieces are grouped by distance
    for r in loose:
        if r["kind"] == "table":
            figures.append(dict(r))
    for rect in _merge_rects([r["rect"] for r in loose if r["kind"] != "table"], FIG_MERGE_GAP):
        figures.append({"rect": rect, "caption": "", "kind": "figure", "table_md": ""})

    figures.sort(key=lambda f: (f["rect"].y0, f["rect"].x0))
    return figures


def detect_figures_docling(pdf_path: str) -> Optional[Dict[int, List[Dict]]]:
    """Run Docling's layout model on the whole PDF once.
    Returns {page_no (1-based): [{"rect", "caption", "kind"}, ...]}, or None if
    Docling is unavailable/fails (the caller then uses geometric detection)."""
    try:
        converter = _get_docling_converter()
    except ImportError:
        print("   Docling is not installed (pip install docling) - using geometric figure detection.")
        return None
    except Exception as e:
        print(f"   Docling could not start ({e}) - using geometric figure detection.")
        return None

    print(" Detecting figures with Docling...")
    start = time.time()
    try:
        doc = converter.convert(pdf_path).document
    except Exception as e:
        print(f"   Docling failed on this PDF ({e}) - using geometric figure detection.")
        return None

    items = [(p, "figure") for p in getattr(doc, "pictures", [])]
    if DOCLING_INCLUDE_TABLES:
        items += [(t, "table") for t in getattr(doc, "tables", [])]

    def to_rect(prov) -> Optional["fitz.Rect"]:
        page_item = doc.pages.get(prov.page_no)
        if page_item is None:
            return None
        # Docling uses a bottom-left origin; PyMuPDF uses top-left. Both are in PDF points.
        bb = prov.bbox.to_top_left_origin(page_height=page_item.size.height)
        rect = fitz.Rect(bb.l, bb.t, bb.r, bb.b)
        return None if rect.is_empty else rect

    by_page: Dict[int, List[Dict]] = {}
    for item, kind in items:
        try:
            caption = " ".join((item.caption_text(doc) or "").split())[:400]
        except Exception:
            caption = ""
        table_md = ""
        if kind == "table":
            try:
                table_md = (item.export_to_markdown(doc=doc) or "").strip()[:MAX_TABLE_CHARS]
            except Exception:
                table_md = ""
        for prov in getattr(item, "prov", None) or []:
            rect = to_rect(prov)
            if rect is None:
                continue
            by_page.setdefault(prov.page_no, []).append(
                {"rect": rect, "caption": caption, "kind": kind, "table_md": table_md}
            )

    # every text line Docling found (PDF text + OCR text), to collect the words inside each figure
    texts_by_page: Dict[int, List[Tuple["fitz.Rect", str]]] = {}
    for t in getattr(doc, "texts", []):
        text = " ".join((getattr(t, "text", "") or "").split())
        if not text:
            continue
        for prov in getattr(t, "prov", None) or []:
            rect = to_rect(prov)
            if rect is not None:
                texts_by_page.setdefault(prov.page_no, []).append((rect, text))

    result: Dict[int, List[Dict]] = {}
    for page_no, regs in by_page.items():
        figures = _group_by_caption(regs)
        for fig in figures:
            fig["figure_text"] = "" if fig["kind"] == "table" else _text_inside(
                fig["rect"], texts_by_page.get(page_no, []), skip=fig["caption"])
        result[page_no] = figures

    n_fig = sum(1 for v in result.values() for f in v if f["kind"] == "figure")
    n_tab = sum(1 for v in result.values() for f in v if f["kind"] == "table")
    print(f"   Docling found {n_fig} figure(s) and {n_tab} table(s) in {time.time() - start:.0f}s")
    return result


def _text_inside(region: "fitz.Rect", texts: List[Tuple["fitz.Rect", str]], skip: str = "") -> str:
    """Join the text lines whose centre lies inside a figure, in reading order."""
    area = _grow(region, 2)
    inside = []
    for rect, text in texts:
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        if area.x0 <= cx <= area.x1 and area.y0 <= cy <= area.y1 and text != skip \
                and not CAPTION_RE.match(text):
            inside.append((round(rect.y0), rect.x0, text))
    inside.sort()
    out = " | ".join(dict.fromkeys(t for _, _, t in inside))   # drop exact duplicates
    return out[:MAX_FIGURE_TEXT_CHARS]


def extract_images_from_page(
    pdf_document,
    page_number: int,
    images_dir: Path,
    skip_small_images: bool = True,
    min_width: int = 120,       # points (1/72 inch) on the page, not pixels
    min_height: int = 25,
    use_ollama: bool = True,
    regions: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Render each whole figure on the page as ONE image (with its caption).
    `regions` comes from Docling when available; if None, figures are found
    geometrically on this page."""
    page = pdf_document[page_number]
    extracted_images = []

    if not skip_small_images:
        min_width, min_height = 20, 20

    if regions is None:
        try:
            regions = find_figure_regions(page, min_width, min_height)
        except Exception as e:
            print(f"Figure detection failed on page {page_number + 1}: {e}")
            return []
    else:
        regions = [r for r in regions
                   if r["rect"].width >= min_width and r["rect"].height >= min_height]

    blocks = page.get_text("blocks") if regions else []

    for fig_index, reg in enumerate(regions, 1):
        try:
            clip = _grow(reg["rect"], FIG_MARGIN) & page.rect
            zoom = max(1.0, min(FIG_MAX_ZOOM, FIG_MAX_PX / max(clip.width, clip.height)))
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
            image_bytes = pix.tobytes("png")
            width, height = pix.width, pix.height

            caption = reg.get("caption") or _find_caption(page, reg["rect"], blocks)
            kind = reg.get("kind", "figure")
            table_md = reg.get("table_md", "") or ""
            figure_text = reg.get("figure_text")
            if figure_text is None and kind == "figure":
                # geometric path: vector text labels inside the figure (no OCR available here)
                figure_text = " | ".join(
                    line.strip() for line in page.get_text("text", clip=reg["rect"]).splitlines()
                    if line.strip() and not CAPTION_RE.match(line))[:MAX_FIGURE_TEXT_CHARS]
            figure_text = figure_text or ""

            image_filename = f"page_{page_number + 1}_image_{fig_index}.png"
            image_path = images_dir / image_filename
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            info = {"filename": image_filename, "kind": kind, "caption": caption,
                    "figure_text": figure_text, "table_md": table_md}
            facts = _build_facts(info)

            # Stage 1 (VLM): visual description only. Tables skip it - their exact cell
            # content already comes from Docling and stage 2 explains it.
            elapsed = 0.0
            visual = ""
            if use_ollama and kind == "figure":
                print(f" Looking at figure {image_filename} ({width}x{height}px)"
                      f"{' - ' + caption[:60] if caption else ''}...")
                start = time.time()
                visual = describe_image_with_ollama(image_bytes, caption=caption,
                                                    figure_text=figure_text)
                elapsed = time.time() - start
            # provisional description; enrich_figures() replaces it with the final one
            if is_valid_description(visual):
                description = f"{facts}\nDescription: {visual}"
            else:
                description = f"{facts}\nDescription: (AI description skipped)"

            extracted_images.append({
                "image_index": fig_index,
                "filename": image_filename,
                "path": str(image_path),
                "width": width,
                "height": height,
                "format": "png",
                "kind": kind,
                "caption": caption,
                "figure_text": figure_text,
                "table_md": table_md,
                "bbox": [round(v, 1) for v in clip],
                "facts": facts,             # exact text (caption / OCR / table) - kept even if llava fails
                "visual_description": visual,
                "description": description,
                "processing_time": round(elapsed, 2),
            })

        except Exception as e:
            print(f"Failed to render figure {fig_index} on page {page_number + 1}: {e}")
            continue

    return extracted_images


def load_and_clean_pdf(
    pdf_path: str,
    extract_images: bool = False,
    describe_images: bool = False,
    skip_small_images: bool = True,
) -> Dict:
    if extract_images and describe_images:
        if not check_ollama_status():
            print("\n Proceeding WITHOUT AI image descriptions because Ollama is not ready.")
            describe_images = False

    try:
        print("\n Extracting text from PDF...")
        # For math-heavy papers consider `marker-pdf` instead of pymupdf4llm.
        pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        print(f"   Extracted {len(pages_data)} pages")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return {"pages": [], "images": [], "metadata": {}}

    cleaned_pages: List[Dict] = []
    all_images: List[Dict] = []

    pdf_name = Path(pdf_path).stem
    base_dir = Path(pdf_path).parent / f"{pdf_name}_extracted"
    images_dir = base_dir / "images"

    pdf_document = None
    if extract_images:
        images_dir.mkdir(parents=True, exist_ok=True)
        # remove fragments left by older runs so they don't mix with the new whole-figure images
        for old in images_dir.glob("page_*_image_*.*"):
            try:
                old.unlink()
            except OSError:
                pass
        pdf_document = fitz.open(pdf_path)

    # Docling looks at the whole PDF once; pages then just pick up their figures.
    docling_regions = None
    if extract_images and FIGURE_DETECTOR == "docling":
        docling_regions = detect_figures_docling(pdf_path)

    total_processing_time = 0.0
    pages_text = [(i + 1, (p.get("text") or "").strip()) for i, p in enumerate(pages_data)]
    paper_meta = load_paper_meta(pdf_path, pages_text[0][1] if pages_text else "")

    # Pass 1 - every page: render figures/tables and run the VLM (stage 1, visual only)
    for i, page in enumerate(pages_data):
        text = page.get("text", "")

        if not text.strip() and not extract_images:
            continue

        page_dict = {"page_number": i + 1, "text": text.strip(), "images": []}

        if extract_images:
            page_regions = None if docling_regions is None else docling_regions.get(i + 1, [])
            page_images = extract_images_from_page(
                pdf_document, i, images_dir, skip_small_images,
                use_ollama=describe_images, regions=page_regions,
            )
            page_dict["images"] = page_images
            all_images.extend(page_images)
            total_processing_time += sum(img.get("processing_time", 0) for img in page_images)

        cleaned_pages.append(page_dict)

    if pdf_document is not None:
        pdf_document.close()

    # Pass 2 - whole paper: references + paper summary + text-model explanation (stage 2).
    # Runs after ALL VLM calls, so Ollama does not swap llava <-> qwen for every figure.
    paper_summary = ""
    if all_images:
        t0 = time.time()
        paper_summary = enrich_figures(all_images, pages_text, paper_meta["paper_title"],
                                       use_llm=describe_images)
        total_processing_time += time.time() - t0

    # Short pointer in the page text so text chunks still name the figures on their page
    for page_dict in cleaned_pages:
        if page_dict["images"]:
            refs = f"\n\n---\n{FIG_LIST_HEADER}\n"
            for img in page_dict["images"]:
                refs += f"- {img['filename']}"
                refs += f": {img['caption']}\n" if img.get("caption") else "\n"
            page_dict["text"] += refs

    metadata = {
        "pdf_path": pdf_path,
        "pdf_name": pdf_name,
        "paper_title": paper_meta["paper_title"],
        "arxiv_id": paper_meta["arxiv_id"],
        "total_pages": len(cleaned_pages),
        "total_images": len(all_images),
        "images_directory": str(images_dir) if extract_images else None,
        "descriptions_generated": describe_images,
        "paper_summary": paper_summary,
    }

    if describe_images and all_images:
        metadata["total_processing_time_seconds"] = round(total_processing_time, 2)
        metadata["average_time_per_image"] = round(total_processing_time / len(all_images), 2)

    return {"pages": cleaned_pages, "images": all_images, "metadata": metadata}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_into_chunks(
    extracted_data: Dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    """Split pages into overlapping text chunks, plus one dedicated chunk per
    described image. Every chunk carries the paper's title and arXiv id; image
    chunks also carry a unique image_id ("<pdf_name>/<filename>")."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )

    meta = extracted_data.get("metadata", {})
    pdf_name = meta.get("pdf_name", "unknown_paper")
    pdf_path = meta.get("pdf_path", "")
    paper_title = meta.get("paper_title", pdf_name)
    arxiv_id = meta.get("arxiv_id", "unknown")

    chunks: List[Dict] = []
    chunk_counter = 0
    image_chunk_count = 0

    for page in extracted_data.get("pages", []):
        page_num = page.get("page_number", 0)
        page_text = page.get("text", "").strip()

        # --- text chunks ---
        if page_text:
            for split_idx, split_text in enumerate(splitter.split_text(page_text)):
                chunk_counter += 1
                img_matches = sorted(set(re.findall(IMG_NAME_RE, split_text)))

                chunks.append({
                    "chunk_id": f"{pdf_name}_p{page_num}_c{split_idx + 1}",
                    "chunk_index": chunk_counter,
                    "page_number": page_num,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "paper_title": paper_title,
                    "arxiv_id": arxiv_id,
                    "referenced_images": img_matches,
                    "has_images": bool(img_matches),
                    "chunk_type": "text",
                    "image_id": "",
                    "text": split_text,
                    "char_count": len(split_text),
                })

        # --- one dedicated chunk per figure / table ---
        for img in page.get("images", []):
            desc = img.get("description", "")
            facts = img.get("facts", "")
            # keep the llava description when it is good; otherwise keep the exact facts
            # (caption, OCR text, table cells) so tables/figures stay searchable anyway
            body = desc if is_valid_description(desc) else facts
            if not body or not (is_valid_description(desc) or img.get("caption")
                                or img.get("figure_text") or img.get("table_md")):
                continue
            chunk_counter += 1
            image_chunk_count += 1
            image_id = make_image_id(pdf_name, img["filename"])
            kind_label = "Table" if img.get("kind") == "table" else "Figure"
            text = (
                f"{kind_label} from \"{paper_title}\", page {page_num} (image id: {image_id}).\n"
                f"{body}"
            )
            chunks.append({
                "chunk_id": f"{pdf_name}_p{page_num}_{img['filename']}",
                "chunk_index": chunk_counter,
                "page_number": page_num,
                "pdf_name": pdf_name,
                "pdf_path": pdf_path,
                "paper_title": paper_title,
                "arxiv_id": arxiv_id,
                "referenced_images": [img["filename"]],
                "has_images": True,
                "chunk_type": "image",
                "figure_kind": img.get("kind", "figure"),   # "figure" or "table"
                "image_id": image_id,
                "caption": img.get("caption", ""),
                "text": text,
                "char_count": len(text),
            })

    link_figures_and_text(chunks)

    print(f"\n Created {len(chunks)} chunks ({image_chunk_count} image chunks) from "
          f"{len(extracted_data.get('pages', []))} pages for \"{paper_title}\".")
    return chunks


MENTION_RE = re.compile(r"\b(Figures?|Figs?\.|Tables?|Tabs?\.)\s*(\d+)", re.IGNORECASE)
FIG_LIST_HEADER = "**Figures and tables on this page:**"
MAX_LINKED_CHUNKS = 5


def link_figures_and_text(chunks: List[Dict]) -> None:
    """Two-way links, stored on the chunks:
    - text chunk  -> "linked_images": figures/tables it mentions ("as shown in Figure 2");
    - image chunk -> "linked_chunks": text chunks that mention it.
    Retrieval follows these links, so a found figure brings the authors' explanation
    with it, and a found paragraph brings the figure it talks about."""
    by_label: Dict[Tuple[str, int], Dict] = {}
    for c in chunks:
        if c.get("chunk_type") == "image":
            label = _figure_number(c.get("caption", ""))
            if label and label not in by_label:
                by_label[label] = c
            c["linked_chunks"] = []

    for c in chunks:
        if c.get("chunk_type") == "image":
            continue
        linked = []
        body = c["text"].split(FIG_LIST_HEADER)[0]   # ignore the "figures on this page" list
        for word, num in MENTION_RE.findall(body):
            kind = "table" if word.lower().startswith("tab") else "figure"
            target = by_label.get((kind, int(num)))
            if target and target["image_id"] not in linked:
                linked.append(target["image_id"])
                if len(target["linked_chunks"]) < MAX_LINKED_CHUNKS:
                    target["linked_chunks"].append(c["chunk_id"])
        c["linked_images"] = linked


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def build_vector_store(chunks: list, batch_size: int = 100, delay_seconds: float = 0.0):
    """Build a FAISS store using local Ollama embeddings (no API key needed)."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    docs = []
    for chunk in chunks:
        pdf_name = chunk.get("pdf_name", "unknown_paper")
        docs.append(Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk.get("chunk_id", ""),
                "page_number": chunk.get("page_number", 0),
                "pdf_name": pdf_name,
                # .get() fallbacks keep old *_chunks.json files (without titles) working
                "paper_title": chunk.get("paper_title") or pdf_name,
                "arxiv_id": chunk.get("arxiv_id") or "unknown",
                "referenced_images": chunk.get("referenced_images", []),
                "chunk_type": chunk.get("chunk_type", "text"),
                "figure_kind": chunk.get("figure_kind", ""),
                "image_id": chunk.get("image_id", ""),
                "linked_images": chunk.get("linked_images", []),
                "linked_chunks": chunk.get("linked_chunks", []),
            },
        ))

    if not docs:
        raise ValueError("No chunks to embed.")

    store = None
    total_batches = (len(docs) - 1) // batch_size + 1

    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        n = i // batch_size + 1
        print(f" Embedding batch {n}/{total_batches} ({len(batch)} chunks)...")

        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)

        if delay_seconds and n < total_batches:
            time.sleep(delay_seconds)

    return store


def initialize_rag(chunks_json_paths: Optional[list] = None) -> None:
    """Load *_chunks.json files (or find them in downloaded_papers/) and build the FAISS store."""
    global vectorstore, _images_dirs, _images_dirs_by_pdf
    _images_dirs = []
    _images_dirs_by_pdf = {}

    if isinstance(chunks_json_paths, str):
        chunks_json_paths = [chunks_json_paths]

    if not chunks_json_paths:
        papers_dir = Path(__file__).resolve().parent.parent / "downloaded_papers"
        candidates = sorted(papers_dir.glob("*_chunks.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_chunks.json found in {papers_dir}. "
                "Download a paper and run process_pdf_to_chunks() first."
            )
        chunks_json_paths = [str(p) for p in candidates]

    all_chunks = []
    for path in chunks_json_paths:
        print(f"Loading RAG chunks from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and "paper_title" not in data[0]:
            print("   WARNING: this chunks file has no paper titles. "
                  "Delete it and re-run process_pdf_to_chunks() to regenerate it.")
        if data and not any(c.get("chunk_type") == "image" for c in data):
            print("   NOTE: no image chunks in this file. Re-run process_pdf_to_chunks("
                  "describe_images=True, force=True) to answer from figures.")
        all_chunks.extend(data)

        chunks_path = Path(path)
        pdf_stem = chunks_path.stem.replace("_chunks", "")
        images_dir = chunks_path.parent / f"{pdf_stem}_extracted" / "images"
        _images_dirs.append(str(images_dir))
        _images_dirs_by_pdf[pdf_stem] = images_dir

    # lookup tables for following figure <-> text links at query time
    _chunks_by_id.clear()
    _image_chunks.clear()
    for c in all_chunks:
        _chunks_by_id[c.get("chunk_id", "")] = c
        if c.get("chunk_type") == "image" and c.get("image_id"):
            _image_chunks[c["image_id"]] = c
    if all_chunks and not any("linked_images" in c or "linked_chunks" in c for c in all_chunks):
        print("   NOTE: these chunks have no figure-text links. Re-process the papers to enable them.")

    print(f"Embedding {len(all_chunks)} chunks - this may take a moment...")
    vectorstore = build_vector_store(all_chunks)

    print(f"Images directories set: {len(_images_dirs)} loaded.")
    print("RAG vector store ready.")


def list_loaded_papers() -> List[Dict[str, str]]:
    """Return unique papers currently in the vector store (for the registry / list tool)."""
    if vectorstore is None:
        return []
    seen, papers = set(), []
    for doc in vectorstore.docstore._dict.values():
        key = doc.metadata.get("arxiv_id", "") + doc.metadata.get("pdf_name", "")
        if key not in seen:
            seen.add(key)
            papers.append({
                "title": doc.metadata.get("paper_title", ""),
                "arxiv_id": doc.metadata.get("arxiv_id", ""),
            })
    return papers


# ---------------------------------------------------------------------------
# Retrieval (used directly by the orchestrator, returns text + image map)
# ---------------------------------------------------------------------------
def retrieve_passages(query: str, top_k: int = 5) -> Tuple[str, Dict[str, str]]:
    """Return (context_text, {image_id: absolute_path}) for the selected papers.
    The image map holds ONLY images attached to the retrieved passages, so the
    caller can verify any [IMG: ...] tag the model writes."""
    if vectorstore is None:
        return "[ERROR] RAG vector store is not initialized. Call initialize_rag() first.", {}

    top_k = max(1, min(int(top_k), 8))

    # Hard scope: only chunks from the selected papers can ever be returned.
    kwargs = {}
    if _allowed_ids:
        kwargs = dict(
            fetch_k=max(50, top_k * 10),
            filter=lambda md: md.get("arxiv_id") in _allowed_ids,
        )
    results = vectorstore.similarity_search(query, k=top_k, **kwargs)

    if not results:
        return "No relevant passages found in the selected papers.", {}

    # (metadata, text, note) for every passage; start with the search hits
    passages = [(doc.metadata, doc.page_content.strip(), "") for doc in results]
    seen = {md.get("chunk_id") for md, _, _ in passages}

    # Follow figure <-> text links: a found figure brings the paragraphs that discuss it,
    # a found paragraph brings the figure/table it mentions.
    def allowed(c: Dict) -> bool:
        return not _allowed_ids or c.get("arxiv_id") in _allowed_ids

    extra = []
    for md, _, _ in passages:
        if len(extra) >= MAX_LINKED_PASSAGES:
            break
        if md.get("chunk_type") == "image":
            for cid in (md.get("linked_chunks") or [])[:2]:
                c = _chunks_by_id.get(cid)
                if c and cid not in seen and allowed(c) and len(extra) < MAX_LINKED_PASSAGES:
                    seen.add(cid)
                    extra.append((c, c["text"].strip(),
                                  f"paper text that discusses {md.get('image_id', 'this figure')}"))
        else:
            for image_id in md.get("linked_images") or []:
                c = _image_chunks.get(image_id)
                if c and c.get("chunk_id") not in seen and allowed(c) and len(extra) < MAX_LINKED_PASSAGES:
                    seen.add(c.get("chunk_id"))
                    extra.append((c, c["text"].strip(), "figure/table mentioned in a passage above"))
    passages += extra

    parts = ["=== RETRIEVED PASSAGES (cite as [Title, arXiv:ID, p.N]) ==="]
    images: Dict[str, str] = {}

    for md, text, note in passages:
        pdf_name = md.get("pdf_name", "")
        if md.get("chunk_type") == "image":
            kind = "TABLE" if md.get("figure_kind") == "table" else "FIGURE DESCRIPTION"
        else:
            kind = "TEXT"
        if note:
            kind += f" - linked: {note}"

        parts.append(
            f"\n--- [{kind}] Source: \"{md.get('paper_title', 'Unknown')}\" "
            f"(arXiv:{md.get('arxiv_id', 'unknown')}), page {md.get('page_number', '?')} ---\n"
            f"{text}"
        )

        names = md.get("referenced_images") or []
        if isinstance(names, str):
            names = [names]
        if not names:  # fallback for older chunks without metadata
            names = sorted(set(re.findall(IMG_NAME_RE, text)))

        tags = []
        for name in names:
            path = resolve_image_path(pdf_name, name)
            if not path:
                continue
            image_id = make_image_id(pdf_name, name)
            images[image_id] = str(path.resolve())
            tags.append(f"- [IMG: {image_id}]")

        if tags:
            parts.append(
                "IMAGES FOR THIS PASSAGE (if your answer uses one, write its tag exactly):\n"
                + "\n".join(tags)
            )

    return "\n".join(parts), images


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------
@tool
def search_paper_rag(query: str, top_k: int = 4) -> str:
    """Search the selected papers (text passages and figure descriptions).
    Passages that relate to an image include an [IMG: id] tag for that image."""
    return retrieve_passages(query, top_k)[0]


@tool
def get_paper_images(page_number: int = 0) -> str:
    """List extracted figure images for a page of the selected papers.
    Pass page_number=0 to list all images. Each image is labelled with its [IMG: id]."""
    if not _images_dirs:
        return "[ERROR] Images directories are not set. Call initialize_rag() first."

    all_images: List[Path] = []
    for dir_path in _images_dirs:
        images_dir = Path(dir_path)
        if images_dir.exists():
            for pattern in ("png", "jpg", "jpeg"):
                all_images.extend(images_dir.glob(f"page_*_image_*.{pattern}"))

    if not all_images:
        return "No extracted images found for the selected papers."

    if page_number == 0:
        matched, header = all_images, "All extracted figures"
    else:
        matched = [p for p in all_images if p.name.startswith(f"page_{page_number}_")]
        header = f"Figures on page {page_number}"

    if not matched:
        pages = sorted({int(p.name.split("_")[1]) for p in all_images})
        return f"No images found for page {page_number}. Pages with images: {pages}"

    lines = [f"{header} ({len(matched)} image(s)):\n"]
    for img_path in sorted(matched):
        pdf_name = img_path.parent.parent.name.replace("_extracted", "")
        size_kb = round(img_path.stat().st_size / 1024, 1)
        lines.append(f"  - [IMG: {make_image_id(pdf_name, img_path.name)}]")
        lines.append(f"    Path : {img_path.resolve()}")
        lines.append(f"    Size : {size_kb} KB")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def process_pdf_to_chunks(pdf_file: str, extract_images: bool = True,
                          describe_images: bool = False, force: bool = False):
    """Process one PDF into chunks and save <stem>_chunks.json. Skips if it exists unless force=True."""
    if not Path(pdf_file).exists():
        print(f"Could not find PDF at: {pdf_file}")
        return None

    chunks_file = Path(pdf_file).parent / f"{Path(pdf_file).stem}_chunks.json"
    if chunks_file.exists() and not force:
        return str(chunks_file)

    print(f"\n Processing PDF to chunks: {pdf_file}")
    extracted = load_and_clean_pdf(pdf_file, extract_images=extract_images,
                                   describe_images=describe_images)

    if extracted and extracted.get("pages"):
        chunks = split_into_chunks(extracted, chunk_size=1000, chunk_overlap=200)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved chunks to: {chunks_file}")
        return str(chunks_file)

    print("No text could be extracted from PDF.")
    return None


if __name__ == "__main__":
    pdf_file = str(Path(__file__).parent.parent / "downloaded_papers" / "2404.01335.pdf")

    if not Path(pdf_file).exists():
        print(f"Could not find PDF at: {pdf_file}")
        sys.exit(1)

    chunks_path = process_pdf_to_chunks(pdf_file, extract_images=True,
                                        describe_images=True, force=True)
    if chunks_path:
        initialize_rag(chunks_path)

        test_query = "What does the architecture figure show?"
        print(f"\n Test query: '{test_query}'")
        context, imgs = retrieve_passages(test_query)
        print(context)
        print("\n Images available for this query:", imgs)
        print("\n Loaded papers:", list_loaded_papers())