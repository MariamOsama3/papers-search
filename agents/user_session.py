"""Per-user state for the paper research agent (Phase 1).

Everything that used to be a module-level global in orchestra.py / Paper_RAG.py
lives in a UserSession, one per user. Each user gets their own folder:

    users/<user_id>/
        state.json      registry, selected papers, chunk files, topic
        papers/         PDFs, *_chunks.json, *_extracted/images/
        index/          saved FAISS index + lookup tables   (filled in step 2)
        chats.sqlite    saved chat history                  (used in step 3)
"""
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_ROOT = PROJECT_ROOT / "users"


def safe_user_id(raw: str) -> str:
    """Turn a login name / email into a folder name that cannot escape USERS_ROOT.
    Readable part (letters, digits, - and _ only) + a short hash so different
    emails never collide, e.g. 'mariam@uni.edu' -> 'mariam-uni-edu_3f9a2c1b'."""
    raw = (raw or "").strip().lower()
    if not raw:
        raise ValueError("user id is empty")
    readable = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")[:40] or "user"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


@dataclass
class UserSession:
    user_id: str

    # --- was in orchestra.py ---
    topic: str = ""
    registry: List[dict] = field(default_factory=list)       # papers found by the last search
    downloaded_ids: set = field(default_factory=set)         # selected (downloaded + processed) papers
    session_chunks: set = field(default_factory=set)         # this user's *_chunks.json files
    rag_dirty: bool = False                                  # index must be rebuilt before next question
    last_images: Dict[str, str] = field(default_factory=dict)  # image_id -> path, from the last retrieval

    # --- was in Paper_RAG.py (filled by the retrieval code in step 2) ---
    vectorstore: object = None
    images_dirs_by_pdf: Dict[str, Path] = field(default_factory=dict)
    chunks_by_id: Dict[str, dict] = field(default_factory=dict)
    image_chunks: Dict[str, dict] = field(default_factory=dict)

    # one user's requests run one at a time (two tabs can't corrupt their state)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ---------------- folders ----------------
    @property
    def root(self) -> Path:
        return USERS_ROOT / self.user_id

    @property
    def papers_dir(self) -> Path:
        return self.root / "papers"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def chats_db(self) -> Path:
        return self.root / "chats.sqlite"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.papers_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)

    def owns(self, path) -> bool:
        """True only if `path` is inside this user's folder (use before opening any file)."""
        try:
            Path(path).resolve().relative_to(self.root.resolve())
            return True
        except (ValueError, OSError):
            return False

    # ---------------- save / load ----------------
    def save(self) -> None:
        """Write the small state to state.json (the index is saved separately in step 2)."""
        with self.lock:
            self.ensure_dirs()
            data = {
                "user_id": self.user_id,
                "topic": self.topic,
                "registry": self.registry,
                "downloaded_ids": sorted(self.downloaded_ids),
                # stored relative to the user folder, so the folder can be moved
                "session_chunks": sorted(
                    str(Path(p).resolve().relative_to(self.root.resolve()))
                    for p in self.session_chunks if self.owns(p)
                ),
            }
            tmp = self.state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.state_file)          # atomic: never a half-written state.json

    @classmethod
    def load_or_create(cls, user_id: str) -> "UserSession":
        s = cls(user_id=user_id)
        s.ensure_dirs()
        if s.state_file.exists():
            try:
                data = json.loads(s.state_file.read_text(encoding="utf-8"))
                s.topic = data.get("topic", "")
                s.registry = data.get("registry", [])
                s.downloaded_ids = set(data.get("downloaded_ids", []))
                s.session_chunks = {
                    str(s.root / rel) for rel in data.get("session_chunks", [])
                    if (s.root / rel).exists()
                }
                # the index is rebuilt (or loaded, in step 2) before the next question
                s.rag_dirty = bool(s.session_chunks)
            except Exception as e:
                print(f"   [could not read {s.state_file}: {e} - starting fresh]")
        return s


class SessionManager:
    """One UserSession per user, shared by all requests/tabs of the app."""

    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()

    def get(self, login: str) -> UserSession:
        uid = safe_user_id(login)
        with self._lock:                      # two logins can't create the same session twice
            if uid not in self._sessions:
                self._sessions[uid] = UserSession.load_or_create(uid)
            return self._sessions[uid]

    def active_users(self) -> List[str]:
        with self._lock:
            return list(self._sessions)