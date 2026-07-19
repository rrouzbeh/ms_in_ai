#!/usr/bin/env python3
"""Build and inspect a persistent ChromaDB index of NASA mission documents."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chroma_embedding_text_only.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Chunk NASA text files, embed the chunks, and persist them in ChromaDB."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chroma_persist_directory: str = "./chroma_db_openai",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_persist_directory = str(Path(chroma_persist_directory).resolve())
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "NASA Apollo 11, Apollo 13, and Challenger text chunks",
                "embedding_model": embedding_model,
                "hnsw:space": "cosine",
            },
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse OCR whitespace while retaining all words and punctuation."""
        return re.sub(r"\s+", " ", text).strip()

    def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into character-bounded, consistently overlapping chunks.

        Every non-final chunk is at most ``chunk_size`` characters. Whenever possible,
        its end is moved to a nearby sentence boundary. The next chunk starts exactly
        ``chunk_overlap`` characters before the previous chunk ended.
        """
        cleaned = self._normalize_text(text)
        if not cleaned:
            return []

        spans: List[Tuple[int, int]] = []
        start = 0
        text_length = len(cleaned)
        while start < text_length:
            hard_end = min(start + self.chunk_size, text_length)
            end = hard_end
            if hard_end < text_length:
                # Avoid tiny chunks: only prefer a natural boundary in the final 40%.
                boundary_floor = start + max(self.chunk_overlap + 1, int(self.chunk_size * 0.6))
                window = cleaned[boundary_floor:hard_end]
                matches = list(re.finditer(r"(?:[.!?](?=\s)|\n)", window))
                if matches:
                    end = boundary_floor + matches[-1].end()

            if end <= start:
                end = hard_end
            spans.append((start, end))
            if end == text_length:
                break
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        total_chunks = len(spans)
        for index, (char_start, char_end) in enumerate(spans):
            chunk = cleaned[char_start:char_end]
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_size": len(chunk),
                    "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                }
            )
            chunks.append((chunk, chunk_metadata))
        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """Return whether ``doc_id`` is already present in the collection."""
        return bool(self.collection.get(ids=[doc_id]).get("ids", []))

    def get_embeddings(self, texts: Sequence[str], max_attempts: int = 3) -> List[List[float]]:
        """Create OpenAI embeddings for a batch, retrying transient failures."""
        if not texts:
            return []
        if self.openai_client is None:
            raise ValueError(
                "An OpenAI API key is required to create embeddings. Set OPENAI_API_KEY "
                "or pass --openai-key."
            )

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=list(texts),
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in ordered]
            except Exception:
                if attempt == max_attempts:
                    raise
                delay = 2 ** (attempt - 1)
                logger.warning("Embedding request failed; retrying in %s second(s)", delay)
                time.sleep(delay)
        raise RuntimeError("Embedding creation failed")

    def get_embedding(self, text: str) -> List[float]:
        """Create one OpenAI embedding."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return self.get_embeddings([text])[0]

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing collection item and its embedding."""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[self.get_embedding(text)],
            )
            return True
        except Exception as exc:
            logger.error("Error updating document %s: %s", doc_id, exc)
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete chunks whose source contains ``source_pattern``."""
        all_docs = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(all_docs.get("ids", []), all_docs.get("metadatas", []))
            if source_pattern in str((metadata or {}).get("source", ""))
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Return all stored chunk IDs associated with a source file."""
        result = self.collection.get(
            where={
                "$and": [
                    {"source": file_path.stem},
                    {"mission": self.extract_mission_from_path(file_path)},
                ]
            },
            include=[],
        )
        return list(result.get("ids", []))

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Generate a stable mission/source/chunk ID safe for ChromaDB."""
        mission = re.sub(r"[^a-z0-9_]+", "_", str(metadata.get("mission", "unknown")).lower())
        source = re.sub(r"[^a-z0-9_]+", "_", file_path.stem.lower()).strip("_")
        chunk_index = int(metadata.get("chunk_index", 0))
        return f"{mission}_{source}_chunk_{chunk_index:05d}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """Read and chunk one UTF-8 NASA text file with source metadata."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []

        resolved_path = file_path.resolve()
        metadata: Dict[str, Any] = {
            "source": file_path.stem,
            "file_path": str(resolved_path),
            "file_type": "text",
            "content_type": "full_text",
            "mission": self.extract_mission_from_path(file_path),
            "data_type": self.extract_data_type_from_path(file_path),
            "document_category": self.extract_document_category_from_filename(file_path.name),
            "file_size": len(content),
            "source_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self.chunk_text(content, metadata)

    @staticmethod
    def extract_mission_from_path(file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        if "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        if "challenger" in path_str or "sts-51l" in path_str:
            return "challenger"
        return "unknown"

    @staticmethod
    def extract_data_type_from_path(file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "flight_plan" in path_str:
            return "flight_plan"
        if "mission_audio" in path_str or "audio" in path_str:
            return "audio_transcript"
        if "transcript" in path_str or "transscript" in path_str:
            return "transcript"
        if "textract" in path_str:
            return "textract_extracted"
        return "document"

    @staticmethod
    def extract_document_category_from_filename(filename: str) -> str:
        lowered = filename.lower()
        if "pao" in lowered:
            return "public_affairs_officer"
        if "_cm" in lowered or "transscript_cm" in lowered:
            return "command_module"
        if "tec" in lowered:
            return "technical"
        if "flight_plan" in lowered:
            return "flight_plan"
        if "mission_audio" in lowered:
            return "mission_audio"
        if "ntrs" in lowered:
            return "nasa_archive"
        if "19900066485" in lowered:
            return "technical_report"
        if "19710015566" in lowered:
            return "mission_report"
        return "general_document"

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Find all supported NASA ``.txt`` sources in deterministic order."""
        root = Path(base_path)
        files: List[Path] = []
        for mission_dir in ("apollo11", "apollo13", "challenger"):
            directory = root / mission_dir
            if directory.is_dir():
                files.extend(directory.rglob("*.txt"))
        return sorted(
            path
            for path in files
            if not path.name.startswith(".") and "summary" not in path.name.lower()
        )

    @staticmethod
    def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
        if size <= 0:
            raise ValueError("batch_size must be greater than zero")
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """Add, update, or replace chunks from one file in embedding batches."""
        if update_mode not in {"skip", "update", "replace"}:
            raise ValueError("update_mode must be one of: skip, update, replace")
        stats = {"added": 0, "updated": 0, "skipped": 0}
        if not documents:
            return stats

        existing_ids = set(self.get_file_documents(file_path))
        desired_ids = {self.generate_document_id(file_path, meta) for _, meta in documents}
        if update_mode == "replace" and existing_ids:
            self.collection.delete(ids=sorted(existing_ids))
            existing_ids.clear()
        elif update_mode == "update":
            stale_ids = existing_ids - desired_ids
            if stale_ids:
                self.collection.delete(ids=sorted(stale_ids))
                existing_ids -= stale_ids

        pending: List[Tuple[str, str, Dict[str, Any], str]] = []
        for text, metadata in documents:
            doc_id = self.generate_document_id(file_path, metadata)
            if doc_id in existing_ids:
                if update_mode == "skip":
                    stats["skipped"] += 1
                    continue
                action = "update"
            else:
                action = "add"
            pending.append((doc_id, text, metadata, action))

        for batch in self._batched(pending, batch_size):
            texts = [item[1] for item in batch]
            embeddings = self.get_embeddings(texts)
            additions = [(item, emb) for item, emb in zip(batch, embeddings) if item[3] == "add"]
            updates = [(item, emb) for item, emb in zip(batch, embeddings) if item[3] == "update"]
            if additions:
                self.collection.add(
                    ids=[item[0] for item, _ in additions],
                    documents=[item[1] for item, _ in additions],
                    metadatas=[item[2] for item, _ in additions],
                    embeddings=[embedding for _, embedding in additions],
                )
                stats["added"] += len(additions)
            if updates:
                self.collection.update(
                    ids=[item[0] for item, _ in updates],
                    documents=[item[1] for item, _ in updates],
                    metadatas=[item[2] for item, _ in updates],
                    embeddings=[embedding for _, embedding in updates],
                )
                stats["updated"] += len(updates)
        return stats

    def process_all_text_data(
        self, base_path: str, update_mode: str = "skip", batch_size: int = 50
    ) -> Dict[str, Any]:
        """Process every supported source and return detailed statistics."""
        stats: Dict[str, Any] = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }
        files = self.scan_text_files_only(base_path)
        if not files:
            raise FileNotFoundError(
                f"No NASA .txt files found under {Path(base_path).resolve()}; expected "
                "apollo11/, apollo13/, and/or challenger/."
            )

        for file_path in files:
            mission = self.extract_mission_from_path(file_path)
            mission_stats = stats["missions"].setdefault(
                mission, {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0}
            )
            try:
                chunks = self.process_text_file(file_path)
                result = self.add_documents_to_collection(
                    chunks, file_path, batch_size=batch_size, update_mode=update_mode
                )
                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["documents_added"] += result["added"]
                stats["documents_updated"] += result["updated"]
                stats["documents_skipped"] += result["skipped"]
                mission_stats["files"] += 1
                mission_stats["chunks"] += len(chunks)
                for key in ("added", "updated", "skipped"):
                    mission_stats[key] += result[key]
                logger.info("Processed %s (%d chunks)", file_path.name, len(chunks))
            except Exception as exc:
                stats["errors"] += 1
                logger.exception("Failed to process %s: %s", file_path, exc)
        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata or {},
            "persist_directory": self.chroma_persist_directory,
        }

    def query_collection(
        self, query_text: str, n_results: int = 5, mission_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text must not be empty")
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        where = None
        if mission_filter and mission_filter.lower() != "all":
            where = {"mission": mission_filter}
        kwargs: Dict[str, Any] = {
            "query_embeddings": [self.get_embedding(query_text)],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get_collection_stats(self) -> Dict[str, Any]:
        all_docs = self.collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        stats: Dict[str, Any] = {
            "total_documents": len(metadatas),
            "unique_sources": 0,
            "missions": {},
            "data_types": {},
            "document_categories": {},
            "file_types": {},
        }
        sources = set()
        for metadata in metadatas:
            metadata = metadata or {}
            sources.add((metadata.get("mission"), metadata.get("source")))
            for output_key, metadata_key in (
                ("missions", "mission"),
                ("data_types", "data_type"),
                ("document_categories", "document_category"),
                ("file_types", "file_type"),
            ):
                value = str(metadata.get(metadata_key, "unknown"))
                stats[output_key][value] = stats[output_key].get(value, 0) + 1
        stats["unique_sources"] = len(sources)
        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index NASA mission text in ChromaDB")
    parser.add_argument("--data-path", default="./data_text", help="Directory containing mission folders")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip")
    parser.add_argument("--test-query")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--delete-source")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key=args.openai_key,
            chroma_persist_directory=args.chroma_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if args.delete_source:
            logger.info("Deleted %d chunks", pipeline.delete_documents_by_source(args.delete_source))
            return 0
        if args.stats_only:
            for key, value in pipeline.get_collection_stats().items():
                logger.info("%s: %s", key, value)
            return 0
        if not args.openai_key:
            raise ValueError("OPENAI_API_KEY or --openai-key is required when indexing/querying")

        started = time.monotonic()
        stats = pipeline.process_all_text_data(
            args.data_path, update_mode=args.update_mode, batch_size=args.batch_size
        )
        logger.info("Processing complete in %.2fs: %s", time.monotonic() - started, stats)
        logger.info("Collection info: %s", pipeline.get_collection_info())
        logger.info("Collection statistics: %s", pipeline.get_collection_stats())
        if args.test_query:
            logger.info("Test query results: %s", pipeline.query_collection(args.test_query))
        return 0 if stats["errors"] == 0 else 1
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
