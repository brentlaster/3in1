
#!/usr/bin/env python3
"""
index_pdfs.py
────────────────────────────────────────────────────────────────────
Create a **fresh** ChromaDB vector-index from the contents of every PDF
inside `./data/`, embedding **each non-blank line** with the
*all-MiniLM-L6-v2* Sentence-BERT model.

High-level flow
---------------
1. **Reset DB** – delete any existing `./chroma_db/` folder so we never mix
   embeddings from previous runs.
2. **Collect PDFs** – scan `./data/*.pdf`.
3. **Extract lines** – use *pdfplumber* to pull plain text from each page,
   split on newlines, drop blank lines.
4. **Embed** – convert each line to a 384-dimensional vector
   (MiniLM-L6-v2).
5. **Store** – write `(vector, raw line, metadata)` into a persistent
   Chroma collection called `"codebase"`.

After it finishes you can query the vectors with any Chroma-compatible
client or the companion RAG script.
"""

# ───────────────────── standard-library imports ────────────────────
import shutil
import re
from pathlib import Path
from typing import List

# ───────────────────── 3rd-party imports ───────────────────────────
import pdfplumber                               # PDF text extractor
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ╔════════════════════════════════════════════════════════════════╗
# 1.  Configuration / constants                                    ║
# ╚════════════════════════════════════════════════════════════════╝
PDF_DIR          = Path("./data")              # where to look for *.pdf
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"          # SBERT model on HF Hub
CHROMA_PATH      = Path("./chroma_db")         # output folder (wiped each run)
COLLECTION_NAME  = "codebase"                  # logical collection inside DB

# ╔════════════════════════════════════════════════════════════════╗
# 2.  Regex helper: split lines & trim whitespace                  ║
# ╚════════════════════════════════════════════════════════════════╝
#   • `\r?\n`  = Windows or Unix newline
#   • `[^\S\r\n]*` = optional leading/trailing spaces or tabs
LINE_RE = re.compile(r"[^\S\r\n]*\r?\n[^\S\r\n]*")

def extract_lines(path: Path) -> List[str]:
    """
    Read a PDF and return *every* non-blank line, preserving order.

    Parameters
    ----------
    path : Path
        Full path to a .pdf file.

    Returns
    -------
    List[str]
        One entry per non-empty line (page order kept).
    """
    lines: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in LINE_RE.split(text):
                line = raw_line.strip()
                if line:                      # skip blanks
                    lines.append(line)
    return lines

def reset_chroma(db_path: Path) -> None:
    """
    Delete any existing `db_path` directory so we always start clean.
    """
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

# ╔════════════════════════════════════════════════════════════════╗
# 3.  Main routine                                                 ║
# ╚════════════════════════════════════════════════════════════════╝
def index_pdfs() -> None:
    """
    Walk `PDF_DIR`, embed every line of every PDF, and store everything
    into a *new* ChromaDB at `CHROMA_PATH`.
    """
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR.resolve()}")
        return

    # ── 1. Load embedding model (one-off) ─────────────────────────
    print(f"Embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # ── 2. Fresh DB on disk ───────────────────────────────────────
    reset_chroma(CHROMA_PATH)

    # ── 3. Connect to persistent Chroma client ────────────────────
    client = PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),                  # defaults are fine
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    coll = client.get_or_create_collection(COLLECTION_NAME)

    # ── 4. Iterate over every PDF ─────────────────────────────────
    for pdf_path in pdf_files:
        print(f"→ Indexing {pdf_path.name}")
        try:
            lines = extract_lines(pdf_path)
        except Exception as err:
            print(f"[WARN] Could not read {pdf_path}: {err}")
            continue

        # Embed and write each line
        for idx, line in enumerate(lines):
            vector = embed_model.encode(line).tolist()

            coll.add(
                ids        =[f"{pdf_path}-{idx}"],            # unique ID
                embeddings =[vector],                         # the vector
                documents  =[line],                           # raw text
                metadatas  =[{"path": str(pdf_path),
                              "chunk_index": idx}],           # extra info
            )

    print("Indexing complete — new DB stored in ./chroma_db")

# ╔════════════════════════════════════════════════════════════════╗
# 4.  Script entry-point                                           ║
# ╚════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    index_pdfs()
