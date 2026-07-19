"""ChromaDB discovery, semantic retrieval, and LLM context construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class OpenAIQueryCollection:
    """Pair a raw persisted collection with an explicit OpenAI query embedder.

    Chroma persists the default embedding-function identity even when all document
    vectors were supplied explicitly. Attaching a different function while reopening
    then conflicts. This wrapper avoids that ambiguity by sending query embeddings.
    """

    def __init__(self, collection: Any, embedding_function: Any) -> None:
        self.collection = collection
        self.embedding_function = embedding_function

    def count(self) -> int:
        return self.collection.count()

    def query(self, **kwargs: Any) -> Dict[str, Any]:
        return self.collection.query(**kwargs)

    def embed_query(self, query: str) -> List[float]:
        embeddings = self.embedding_function([query])
        return list(embeddings[0])


def discover_chroma_backends(search_root: str = ".") -> Dict[str, Dict[str, Any]]:
    """Discover persistent ChromaDB collections below ``search_root``.

    Only directories containing Chroma's SQLite file are opened, which avoids creating
    empty databases during discovery.
    """
    root = Path(search_root).resolve()
    candidates = {path.parent for path in root.glob("chroma_db*/chroma.sqlite3")}
    candidates.update(path.parent for path in root.glob("*/chroma_db*/chroma.sqlite3"))
    backends: Dict[str, Dict[str, Any]] = {}
    for directory in sorted(candidates):
        try:
            client = chromadb.PersistentClient(path=str(directory))
            for collection in client.list_collections():
                name = collection.name
                key = f"{directory}::{name}"
                backends[key] = {
                    "directory": str(directory),
                    "collection_name": name,
                    "display_name": f"{name} — {directory.name} ({collection.count()} chunks)",
                    "document_count": collection.count(),
                }
        except Exception as exc:
            key = f"{directory}::error"
            backends[key] = {
                "directory": str(directory),
                "collection_name": "",
                "display_name": f"{directory.name} (unavailable: {str(exc)[:80]})",
                "document_count": 0,
                "error": str(exc),
            }
    return backends


def initialize_rag_system(
    chroma_dir: str,
    collection_name: str,
    openai_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
) -> Tuple[Any, bool, Optional[str]]:
    """Open a persisted collection and attach the matching query embedder."""
    try:
        directory = Path(chroma_dir).expanduser().resolve()
        if not (directory / "chroma.sqlite3").exists():
            raise FileNotFoundError(f"No ChromaDB database found at {directory}")
        client = chromadb.PersistentClient(path=str(directory))
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY") or os.getenv(
            "CHROMA_OPENAI_API_KEY"
        )
        if not api_key:
            raise ValueError("An OpenAI API key is required for semantic query embeddings")
        embedding_function = OpenAIEmbeddingFunction(api_key=api_key, model_name=embedding_model)
        collection = client.get_collection(name=collection_name)
        return OpenAIQueryCollection(collection, embedding_function), True, None
    except Exception as exc:
        return None, False, str(exc)


def retrieve_documents(
    collection: Any,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Embed ``query`` and return the nearest unique chunks from ChromaDB."""
    if collection is None:
        raise ValueError("collection is required")
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if n_results <= 0:
        raise ValueError("n_results must be greater than zero")
    if collection.count() == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    where = None
    if mission_filter and mission_filter.lower() not in {"all", "all missions"}:
        where = {"mission": mission_filter}
    kwargs: Dict[str, Any] = {
        # Ask for a few extras so near-identical OCR snippets can be removed.
        "n_results": min(max(n_results * 2, n_results), collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if hasattr(collection, "embed_query"):
        kwargs["query_embeddings"] = [collection.embed_query(query.strip())]
    else:
        # Supports Chroma collections configured with an embedding function and test doubles.
        kwargs["query_texts"] = [query.strip()]
    if where:
        kwargs["where"] = where
    raw = collection.query(**kwargs)

    rows: List[Tuple[float, str, Dict[str, Any], str]] = []
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]
    seen = set()
    for index, document in enumerate(documents):
        normalized = " ".join((document or "").lower().split())
        fingerprint = normalized[:300]
        if not document or fingerprint in seen:
            continue
        seen.add(fingerprint)
        distance = float(distances[index]) if index < len(distances) else float("inf")
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        doc_id = ids[index] if index < len(ids) else ""
        rows.append((distance, document, metadata, doc_id))
    rows.sort(key=lambda row: row[0])
    rows = rows[:n_results]
    return {
        "ids": [[row[3] for row in rows]],
        "documents": [[row[1] for row in rows]],
        "metadatas": [[row[2] for row in rows]],
        "distances": [[row[0] for row in rows]],
    }


def _humanize(value: Any) -> str:
    return str(value or "Unknown").replace("_", " ").strip().title()


def format_context(
    documents: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    distances: Optional[Sequence[float]] = None,
    max_chunk_chars: int = 1600,
) -> str:
    """Create a source-attributed context block sorted by similarity distance."""
    if not documents:
        return ""
    rows = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = float(distances[index]) if distances and index < len(distances) else float(index)
        rows.append((distance, index, document, metadata))
    rows.sort(key=lambda row: row[0])

    context_parts = [
        "RETRIEVED NASA ARCHIVE CONTEXT",
        "Use only these excerpts for factual mission claims. Cite their source labels.",
    ]
    seen = set()
    source_number = 0
    for distance, _, document, metadata in rows:
        normalized = " ".join((document or "").split())
        fingerprint = normalized.lower()[:300]
        if not normalized or fingerprint in seen:
            continue
        seen.add(fingerprint)
        source_number += 1
        source = str(metadata.get("source", "unknown source"))
        mission = _humanize(metadata.get("mission"))
        category = _humanize(metadata.get("document_category"))
        chunk_index = metadata.get("chunk_index")
        chunk_label = f", chunk {int(chunk_index) + 1}" if isinstance(chunk_index, int) else ""
        score_label = f", distance {distance:.4f}" if distances is not None else ""
        context_parts.append(
            f"--- [Source {source_number}: {source} | {mission} | {category}{chunk_label}{score_label}] ---"
        )
        clipped = normalized[:max_chunk_chars]
        if len(normalized) > max_chunk_chars:
            clipped += " …"
        context_parts.append(clipped)
    return "\n\n".join(context_parts)
