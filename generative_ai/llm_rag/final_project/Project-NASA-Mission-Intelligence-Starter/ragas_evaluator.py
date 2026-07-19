"""RAGAS response evaluation and evaluation-dataset utilities."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

try:
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, BleuScore, Faithfulness, RougeScore

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _validate_inputs(question: str, answer: str, contexts: Sequence[str]) -> Optional[str]:
    if not isinstance(question, str) or not question.strip():
        return "question must be a non-empty string"
    if not isinstance(answer, str) or not answer.strip():
        return "answer must be a non-empty string"
    if not isinstance(contexts, (list, tuple)) or not contexts:
        return "contexts must contain at least one retrieved text excerpt"
    if any(not isinstance(context, str) or not context.strip() for context in contexts):
        return "every context must be a non-empty string"
    return None


def _as_float(value: Any) -> float:
    raw_value = value.value if hasattr(value, "value") else value
    score = float(raw_value)
    if math.isnan(score) or math.isinf(score):
        raise ValueError("metric returned a non-finite value")
    return max(0.0, min(1.0, score))


def lexical_context_coverage(answer: str, contexts: Sequence[str]) -> float:
    """Return the share of meaningful answer terms found in retrieved context.

    This inexpensive diagnostic is supplementary; the rubric metrics remain RAGAS
    Response Relevancy and Faithfulness.
    """
    stopwords = {
        "about", "after", "again", "also", "been", "being", "could", "from", "have",
        "into", "more", "that", "their", "there", "these", "they", "this", "through",
        "were", "what", "when", "where", "which", "with", "would", "your",
    }
    answer_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", answer.lower())
        if len(token) > 3 and token not in stopwords
    }
    if not answer_terms:
        return 0.0
    context_terms = set(re.findall(r"[a-z0-9]+", " ".join(contexts).lower()))
    return len(answer_terms & context_terms) / len(answer_terms)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    *,
    openai_api_key: Optional[str] = None,
    evaluator_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small",
) -> Dict[str, Any]:
    """Evaluate one question/context/answer triple with RAGAS.

    Response Relevancy and Faithfulness are always computed. BLEU and ROUGE-L are
    additionally computed when a reference answer is supplied.
    """
    validation_error = _validate_inputs(question, answer, contexts)
    if validation_error:
        return {"error": validation_error}
    if reference_answer is not None and (not isinstance(reference_answer, str) or not reference_answer.strip()):
        return {"error": "reference_answer must be a non-empty string when provided"}
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS is unavailable; install dependencies from requirements.txt"}
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY is required for RAGAS LLM-based metrics"}

    try:
        openai_client = OpenAI(api_key=api_key)
        evaluator_llm = llm_factory(
            evaluator_model, client=openai_client, temperature=0
        )
        evaluator_embeddings = RagasOpenAIEmbeddings(
            client=openai_client, model=embedding_model
        )
        metrics = {
            "response_relevancy": AnswerRelevancy(
                llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1
            ),
            "faithfulness": Faithfulness(llm=evaluator_llm),
        }
        if reference_answer:
            metrics.update({"bleu": BleuScore(), "rouge_l": RougeScore(rouge_type="rougeL")})

        scores: Dict[str, Any] = {}
        cleaned_contexts = [context.strip() for context in contexts]
        scores["response_relevancy"] = _as_float(
            metrics["response_relevancy"].score(
                user_input=question.strip(), response=answer.strip()
            )
        )
        scores["faithfulness"] = _as_float(
            metrics["faithfulness"].score(
                user_input=question.strip(),
                response=answer.strip(),
                retrieved_contexts=cleaned_contexts,
            )
        )
        if reference_answer:
            scores["bleu"] = _as_float(
                metrics["bleu"].score(reference=reference_answer.strip(), response=answer.strip())
            )
            scores["rouge_l"] = _as_float(
                metrics["rouge_l"].score(
                    reference=reference_answer.strip(), response=answer.strip()
                )
            )
        scores["lexical_context_coverage"] = lexical_context_coverage(answer, contexts)
        return scores
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}


def load_evaluation_dataset(path: str) -> List[Dict[str, str]]:
    """Load the project's JSON Lines evaluation dataset from a ``.txt`` file."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    records: List[Dict[str, str]] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        required = {"category", "mission", "question", "expected_answer"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Line {line_number} is missing fields: {', '.join(sorted(missing))}")
        if any(not isinstance(record[field], str) or not record[field].strip() for field in required):
            raise ValueError(f"Line {line_number} contains an empty required field")
        records.append({field: str(record[field]).strip() for field in required})
    if not records:
        raise ValueError("Evaluation dataset contains no records")
    return records


def aggregate_metric_scores(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute a mean for every numeric metric across successful batch rows."""
    values: Dict[str, List[float]] = {}
    for result in results:
        scores = result.get("scores", result)
        if not isinstance(scores, dict) or "error" in scores:
            continue
        for name, value in scores.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(name, []).append(float(value))
    return {name: mean(metric_values) for name, metric_values in sorted(values.items())}
