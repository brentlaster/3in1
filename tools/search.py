#!/usr/bin/env python3
# search.py — colourised, similarity-aware search with numbered, clearly-
#             separated results and explicit cosine-similarity labels.

import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ── ANSI colours (works on most POSIX terminals) ─────────────────────────
GREEN = "\033[92m"   # best match
BLUE  = "\033[94m"   # other matches
RED   = "\033[91m"   # similarity label / value
RESET = "\033[0m"

# ── Connect to on-disk Chroma database ───────────────────────────────────
db_client = PersistentClient(
    path="./chroma_db",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # same model as indexers

# ── Utility: exact cosine similarity ─────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ── Core search routine ──────────────────────────────────────────────────
def search(query: str, top_k: int = 3) -> None:
    coll = db_client.get_or_create_collection(name="codebase")

    total_chunks = len(coll.get().get("documents", []))
    if total_chunks == 0:
        print("Collection is empty — nothing to search.")
        return
    print(f"Collection contains {total_chunks} chunks.\n")

    query_vec = embed_model.encode(query)

    results = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"],
    )

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    embeds = results["embeddings"][0]

    if not docs:
        print("No matches found.")
        return

    sims = [cosine_sim(query_vec, np.array(e)) for e in embeds]
    best_idx = int(np.argmax(sims))

    for i, (doc, meta, sim) in enumerate(zip(docs, metas, sims), start=1):
        colour = GREEN if i-1 == best_idx else BLUE
        separator = "-" * 80
        print(
            f"{colour}{separator}\n"
            f"Result {i}/{len(sims)}\n"
            f"{separator}{RESET}\n"
            f"{doc}\n\n"
            f"{RED}Cosine similarity: {sim:.4f}{RESET}\n"
            f"Source: {meta['path']}  (chunk {meta['chunk_index']})\n"
        )

# ── Simple REPL ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Enter your search query (type 'exit' to quit):")
    while True:
        user_input = input("🔍 Search: ").strip()
        if user_input.lower() == "exit":
            print("Exiting search.")
            break
        if user_input:
            search(user_input)
        else:
            print("Please enter a valid query.")
