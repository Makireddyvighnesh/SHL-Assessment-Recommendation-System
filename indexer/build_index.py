"""
Build LlamaIndex vector index from scraped SHL assessments.
Uses HuggingFace embeddings (free, no API key needed).
Persists index to indexer/storage/ for reuse.

Run once after scraping:
    python indexer/build_index.py
"""

import json
import os
import sys
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("scraper/shl_assessments.json")
STORAGE_PATH = Path("indexer/storage")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # fast, free, 384-dim


def load_assessments(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assessment_to_document(a: dict) -> Document:
    """
    Convert an assessment dict into a LlamaIndex Document.
    Rich text = better semantic search.
    Metadata = returned in API response.
    """
    test_types_str  = ", ".join(a.get("test_type", []))  or "Not specified"
    job_levels_str  = ", ".join(a.get("job_levels", [])) or "Not specified"
    languages_str   = ", ".join(a.get("languages", []))  or "Not specified"
    duration        = a.get("duration")
    duration_str    = f"{duration} minutes" if duration else "Not specified"

    # ── Rich text for embedding ───────────────────────────────────────────────
    # Everything here is searchable — more context = better retrieval
    text = f"""
Assessment Name: {a['name']}

Description: {a.get('description', 'No description available.')}

Test Types: {test_types_str}

Job Levels: {job_levels_str}

Languages: {languages_str}

Duration: {duration_str}

Remote Support: {a.get('remote_support', 'No')}

Adaptive Support: {a.get('adaptive_support', 'No')}

URL: {a.get('url', '')}
""".strip()

    # ── Metadata (returned in API response) ───────────────────────────────────
    metadata = {
        "name":             a["name"],
        "url":              a.get("url", ""),
        "test_type":        json.dumps(a.get("test_type", [])),
        "remote_support":   a.get("remote_support", "No"),
        "adaptive_support": a.get("adaptive_support", "No"),
        "duration":         duration if duration else 0,
        "description":      (a.get("description", "") or "")[:500],
        "job_levels":       json.dumps(a.get("job_levels", [])),
        "languages":        json.dumps(a.get("languages", [])),
    }

    return Document(text=text, metadata=metadata, id_=a.get("url", a["name"]))


def build_index() -> VectorStoreIndex:
    print(f"Loading assessments from {DATA_PATH}...")
    assessments = load_assessments(DATA_PATH)
    print(f"  Loaded {len(assessments)} assessments")

    print(f"Setting up embedding model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = None  # no LLM needed at index time

    print("Converting assessments to Documents...")
    documents = [assessment_to_document(a) for a in assessments]

    print("Building vector index (this may take a minute)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    print(f"Persisting index to {STORAGE_PATH}...")
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_PATH))

    print(f"✅ Index built with {len(documents)} documents")
    return index


def load_index() -> VectorStoreIndex:
    """Load persisted index (faster than rebuilding)."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = None

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_PATH))
    index = load_index_from_storage(storage_context)
    return index


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run scraper/scrape_shl.py first.")
        sys.exit(1)

    build_index()