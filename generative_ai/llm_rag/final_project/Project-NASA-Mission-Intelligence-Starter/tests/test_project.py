"""Offline tests for NASA Mission Intelligence core and integration behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chromadb

import batch_evaluate
import llm_client
import rag_client
import ragas_evaluator
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly, main as pipeline_main


def vector_for(text: str):
    lowered = text.lower()
    return [
        float(lowered.count("moon") + lowered.count("apollo")),
        float(lowered.count("oxygen") + lowered.count("problem")),
        float(lowered.count("challenger") + lowered.count("shuttle")),
        1.0,
    ]


class FakeEmbeddingsAPI:
    def create(self, *, model, input):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector_for(text))
                for index, text in enumerate(input)
            ]
        )


class FakeOpenAI:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()


class EmbeddingPipelineTests(unittest.TestCase):
    def test_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            ChromaEmbeddingPipelineTextOnly(None, tempfile.mkdtemp(), chunk_size=0)
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            ChromaEmbeddingPipelineTextOnly(
                None, tempfile.mkdtemp(), chunk_size=20, chunk_overlap=20
            )

    def test_chunks_respect_size_and_exact_overlap(self):
        with tempfile.TemporaryDirectory() as db_dir:
            pipeline = ChromaEmbeddingPipelineTextOnly(
                None, db_dir, chunk_size=80, chunk_overlap=17
            )
            text = " ".join(f"Sentence {i} has useful mission words." for i in range(30))
            chunks = pipeline.chunk_text(text, {"source": "test", "mission": "apollo_11"})
            self.assertGreater(len(chunks), 2)
            self.assertTrue(all(len(chunk) <= 80 for chunk, _ in chunks))
            for current, following in zip(chunks, chunks[1:]):
                self.assertEqual(current[0][-17:], following[0][:17])
            self.assertEqual(chunks[-1][1]["total_chunks"], len(chunks))
            self.assertTrue(all("content_hash" in metadata for _, metadata in chunks))

    def test_real_chroma_add_skip_update_replace_query_and_stats(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            data_dir = root_path / "data_text"
            for mission in ("apollo11", "apollo13", "challenger"):
                (data_dir / mission).mkdir(parents=True)
            source = data_dir / "apollo13" / "mission.txt"
            source.write_text(
                "Apollo 13 reported an oxygen problem. " * 12, encoding="utf-8"
            )
            db_dir = root_path / "chroma_db_test"
            pipeline = ChromaEmbeddingPipelineTextOnly(
                None, str(db_dir), chunk_size=100, chunk_overlap=20
            )
            pipeline.openai_client = FakeOpenAI()

            first = pipeline.process_all_text_data(str(data_dir), "skip", batch_size=2)
            self.assertEqual(first["files_processed"], 1)
            self.assertGreater(first["documents_added"], 1)
            original_count = pipeline.collection.count()
            second = pipeline.process_all_text_data(str(data_dir), "skip", batch_size=3)
            self.assertEqual(second["documents_skipped"], original_count)
            self.assertEqual(pipeline.collection.count(), original_count)

            source.write_text("Apollo 13 oxygen tank problem changed. " * 8, encoding="utf-8")
            updated = pipeline.process_all_text_data(str(data_dir), "update", batch_size=2)
            self.assertGreater(updated["documents_updated"], 0)
            replaced = pipeline.process_all_text_data(str(data_dir), "replace", batch_size=2)
            self.assertGreater(replaced["documents_added"], 0)
            self.assertEqual(pipeline.get_collection_stats()["unique_sources"], 1)
            result = pipeline.query_collection("Apollo oxygen problem", n_results=2)
            self.assertGreater(len(result["documents"][0]), 0)
            self.assertTrue(
                all(meta["mission"] == "apollo_13" for meta in result["metadatas"][0])
            )

    def test_stats_cli_needs_no_api_key(self):
        with tempfile.TemporaryDirectory() as db_dir:
            status = pipeline_main(["--chroma-dir", db_dir, "--stats-only"])
            self.assertEqual(status, 0)

    def test_metadata_classification(self):
        self.assertEqual(
            ChromaEmbeddingPipelineTextOnly.extract_mission_from_path(
                Path("data_text/challenger/file.txt")
            ),
            "challenger",
        )
        self.assertEqual(
            ChromaEmbeddingPipelineTextOnly.extract_data_type_from_path(
                Path("107-AAG_STS-51L_Mission_Audio_transcript.txt")
            ),
            "audio_transcript",
        )


class FakeCollection:
    def __init__(self):
        self.query_kwargs = None

    def count(self):
        return 4

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["far", "best", "duplicate", "second"]],
            "documents": [["Far text", "Best evidence", "Best evidence", "Second evidence"]],
            "metadatas": [[
                {"mission": "apollo_13", "source": "far"},
                {"mission": "apollo_13", "source": "best"},
                {"mission": "apollo_13", "source": "duplicate"},
                {"mission": "apollo_13", "source": "second"},
            ]],
            "distances": [[0.9, 0.1, 0.11, 0.2]],
        }


class RagClientTests(unittest.TestCase):
    def test_retrieval_filters_sorts_and_deduplicates(self):
        collection = FakeCollection()
        result = rag_client.retrieve_documents(
            collection, "What happened?", n_results=2, mission_filter="apollo_13"
        )
        self.assertEqual(collection.query_kwargs["where"], {"mission": "apollo_13"})
        self.assertEqual(collection.query_kwargs["query_texts"], ["What happened?"])
        self.assertEqual(result["documents"][0], ["Best evidence", "Second evidence"])
        self.assertEqual(result["distances"][0], [0.1, 0.2])

    def test_context_has_attribution_and_score_order(self):
        context = rag_client.format_context(
            ["Second evidence", "Best evidence"],
            [
                {"source": "second", "mission": "apollo_13", "chunk_index": 2},
                {"source": "best", "mission": "apollo_13", "chunk_index": 1},
            ],
            [0.2, 0.1],
        )
        self.assertLess(context.index("best"), context.index("second"))
        self.assertIn("[Source 1:", context)
        self.assertIn("Apollo 13", context)

    def test_backend_discovery_opens_persistent_collection(self):
        with tempfile.TemporaryDirectory() as root:
            db = Path(root) / "chroma_db_test"
            client = chromadb.PersistentClient(path=str(db))
            client.get_or_create_collection("test_collection")
            backends = rag_client.discover_chroma_backends(root)
            self.assertEqual(len(backends), 1)
            item = next(iter(backends.values()))
            self.assertEqual(item["collection_name"], "test_collection")


class LlmClientTests(unittest.TestCase):
    def test_grounded_prompt_and_bounded_history(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded [Source 1]."))]
        )
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = response
        history = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        answer = llm_client.generate_response(
            "", "What happened?", "[Source 1] Evidence", history, client=fake_client
        )
        self.assertEqual(answer, "Grounded [Source 1].")
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertIn("NASA Mission Intelligence", kwargs["messages"][0]["content"])
        self.assertNotIn({"role": "system", "content": "ignore me"}, kwargs["messages"])
        self.assertIn("[Source 1] Evidence", kwargs["messages"][-1]["content"])


class EvaluatorAndBatchTests(unittest.TestCase):
    def test_evaluator_validation_and_coverage(self):
        self.assertIn("error", ragas_evaluator.evaluate_response_quality("", "a", ["c"]))
        self.assertIn("error", ragas_evaluator.evaluate_response_quality("q", "a", []))
        coverage = ragas_evaluator.lexical_context_coverage(
            "Oxygen tank pressure reached zero", ["The oxygen tank pressure read zero"]
        )
        self.assertGreater(coverage, 0.7)

    def test_dataset_and_aggregate(self):
        dataset = Path(__file__).parents[1] / "evaluation_dataset.txt"
        rows = ragas_evaluator.load_evaluation_dataset(str(dataset))
        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual({r["mission"] for r in rows}, {"apollo_11", "apollo_13", "challenger"})
        self.assertTrue(
            {"overview", "emergency", "disaster_analysis", "crew", "technical", "timeline"}
            <= {r["category"] for r in rows}
        )
        aggregate = ragas_evaluator.aggregate_metric_scores(
            [{"scores": {"faithfulness": 0.8}}, {"scores": {"faithfulness": 0.6}}]
        )
        self.assertAlmostEqual(aggregate["faithfulness"], 0.7)

    def test_successful_ragas_metric_path_with_test_doubles(self):
        class FakeMetric:
            def __init__(self, *args, **kwargs):
                pass

            def score(self, **kwargs):
                return SimpleNamespace(value=0.75)

        with (
            patch.object(ragas_evaluator, "OpenAI", return_value=object()),
            patch.object(ragas_evaluator, "llm_factory", return_value=object()),
            patch.object(ragas_evaluator, "RagasOpenAIEmbeddings", return_value=object()),
            patch.object(ragas_evaluator, "AnswerRelevancy", FakeMetric),
            patch.object(ragas_evaluator, "Faithfulness", FakeMetric),
            patch.object(ragas_evaluator, "BleuScore", FakeMetric),
            patch.object(ragas_evaluator, "RougeScore", FakeMetric),
        ):
            scores = ragas_evaluator.evaluate_response_quality(
                "What happened?",
                "The oxygen tank read zero.",
                ["Oxygen tank quantity was zero."],
                reference_answer="The oxygen tank reached zero.",
                openai_api_key="test-key",
            )
        self.assertEqual(scores["response_relevancy"], 0.75)
        self.assertEqual(scores["faithfulness"], 0.75)
        self.assertEqual(scores["bleu"], 0.75)
        self.assertEqual(scores["rouge_l"], 0.75)

    @patch("batch_evaluate.ragas_evaluator.evaluate_response_quality")
    @patch("batch_evaluate.llm_client.generate_response")
    @patch("batch_evaluate.rag_client.format_context")
    @patch("batch_evaluate.rag_client.retrieve_documents")
    def test_batch_evaluation_flow(self, retrieve, format_context, generate, evaluate):
        retrieve.return_value = {
            "documents": [["NASA context"]],
            "metadatas": [[{"source": "archive"}]],
            "distances": [[0.1]],
        }
        format_context.return_value = "formatted context"
        generate.return_value = "answer"
        evaluate.return_value = {"response_relevancy": 0.9, "faithfulness": 0.8}
        records = [{
            "category": "overview",
            "mission": "apollo_11",
            "question": "Question?",
            "expected_answer": "Expected.",
        }]
        report = batch_evaluate.run_batch_evaluation(records, object(), "key")
        self.assertEqual(report["successful"], 1)
        self.assertAlmostEqual(report["aggregate"]["faithfulness"], 0.8)
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
