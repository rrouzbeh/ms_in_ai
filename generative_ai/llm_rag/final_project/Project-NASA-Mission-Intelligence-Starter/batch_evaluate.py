#!/usr/bin/env python3
"""Run the NASA RAG application and RAGAS metrics over evaluation_dataset.txt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import llm_client
import rag_client
import ragas_evaluator


def run_batch_evaluation(
    records: Sequence[Dict[str, str]],
    collection: Any,
    openai_key: str,
    *,
    answer_model: str = "gpt-4.1-mini",
    evaluator_model: str = "gpt-4.1-mini",
    top_k: int = 4,
) -> Dict[str, Any]:
    """Retrieve, answer, and evaluate every dataset record end to end."""
    results: List[Dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        row: Dict[str, Any] = {
            "index": index,
            "category": record["category"],
            "mission": record["mission"],
            "question": record["question"],
        }
        try:
            retrieval = rag_client.retrieve_documents(
                collection,
                record["question"],
                n_results=top_k,
                mission_filter=record["mission"],
            )
            documents = (retrieval or {}).get("documents", [[]])[0]
            metadatas = (retrieval or {}).get("metadatas", [[]])[0]
            distances = (retrieval or {}).get("distances", [[]])[0]
            context = rag_client.format_context(documents, metadatas, distances)
            answer = llm_client.generate_response(
                openai_key, record["question"], context, [], model=answer_model
            )
            scores = ragas_evaluator.evaluate_response_quality(
                record["question"],
                answer,
                list(documents),
                reference_answer=record["expected_answer"],
                openai_api_key=openai_key,
                evaluator_model=evaluator_model,
            )
            row.update(
                {
                    "answer": answer,
                    "expected_answer": record["expected_answer"],
                    "retrieved_sources": [metadata.get("source", "unknown") for metadata in metadatas],
                    "scores": scores,
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
        results.append(row)
    return {
        "questions": results,
        "aggregate": ragas_evaluator.aggregate_metric_scores(results),
        "successful": sum(1 for row in results if "error" not in row and "error" not in row.get("scores", {})),
        "total": len(results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-evaluate NASA Mission Intelligence")
    parser.add_argument("--dataset", default="./evaluation_dataset.txt")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--answer-model", default="gpt-4.1-mini")
    parser.add_argument("--evaluator-model", default="gpt-4.1-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", help="Optional JSON result path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.openai_key:
        print("ERROR: OPENAI_API_KEY or --openai-key is required")
        return 2
    records = ragas_evaluator.load_evaluation_dataset(args.dataset)
    collection, success, error = rag_client.initialize_rag_system(
        args.chroma_dir,
        args.collection_name,
        openai_api_key=args.openai_key,
        embedding_model=args.embedding_model,
    )
    if not success:
        print(f"ERROR: {error}")
        return 2
    report = run_batch_evaluation(
        records,
        collection,
        args.openai_key,
        answer_model=args.answer_model,
        evaluator_model=args.evaluator_model,
        top_k=args.top_k,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["successful"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
