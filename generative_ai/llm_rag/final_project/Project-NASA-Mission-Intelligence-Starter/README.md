# NASA Mission Intelligence

A complete retrieval-augmented chat system over NASA archive material for Apollo 11,
Apollo 13, and Space Shuttle Challenger (STS-51L). It indexes the bundled text sources
with OpenAI embeddings in persistent ChromaDB, retrieves mission-filtered evidence,
generates source-cited answers, and evaluates answers with RAGAS.

## Included data

The repository contains 12 UTF-8 text sources (about 6.3 MB):

- Apollo 11: flight plan, mission and Saturn V evaluation reports, public-affairs
  commentary, technical air-to-ground transcript, and command-module onboard transcript.
- Apollo 13: public-affairs commentary, technical air-to-ground transcript, and
  command-module onboard transcript.
- Challenger: three speaker-segmented STS-51L mission-audio transcripts.

## Project structure

```text
.
├── data_text/                 NASA source documents
├── tests/test_project.py      offline unit and integration tests
├── embedding_pipeline.py      chunking, embeddings, Chroma persistence, CLI
├── rag_client.py              backend discovery, retrieval, context formatting
├── llm_client.py              grounded answer generation and conversation history
├── ragas_evaluator.py         RAGAS metrics and dataset utilities
├── batch_evaluate.py          end-to-end evaluation-set runner
├── chat.py                    Streamlit chat interface
├── evaluation_dataset.txt     9 JSONL questions and expected answers
├── PROJECT_REPORT.md          implementation and verification report
└── requirements.txt           pinned runtime dependencies
```

## Setup

Python 3.10 or newer and an OpenAI API key are required for live indexing, answering,
and RAGAS metrics. Collection statistics and the offline test suite do not require a key.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

## Build the index

From this directory:

```bash
python embedding_pipeline.py \
  --data-path ./data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --batch-size 50 \
  --update-mode skip \
  --test-query "What happened to Apollo 13?"
```

`chunk_size` and `chunk_overlap` are character counts. Chunks never exceed the configured
size, and adjacent chunks share exactly the configured overlap. Natural sentence endings
are preferred when they occur late enough in a chunk.

Update modes:

- `skip`: retain every existing stable chunk ID and embed only new chunks.
- `update`: re-embed existing IDs, add new IDs, and remove stale IDs if a file shrank.
- `replace`: remove every stored chunk for a source file and rebuild it.

The embedding request is batched, while each returned vector remains associated with its
own document and metadata. Metadata includes source/file path, mission, document category,
chunk index and boundaries, timestamps, and source/chunk SHA-256 hashes.

Inspect an existing collection without an API key:

```bash
python embedding_pipeline.py \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --stats-only
```

The output includes total chunks, unique source count, and aggregates by mission, data
type, document category, and file type.

## Run the chat app

```bash
streamlit run chat.py
```

The sidebar supports collection, answer-model, mission, top-k, and optional RAGAS settings.
Every answer receives a context made from score-sorted, deduplicated excerpts with source
labels. The system prompt requires those labels as inline citations and requires an explicit
insufficiency statement when the retrieved evidence cannot support an answer. Conversation
history is bounded and is treated as dialogue context, never as factual evidence.

RAGAS evaluation is off by default because it makes additional model calls. When enabled,
the UI displays Response Relevancy, Faithfulness, and supplementary lexical context coverage.

## Batch evaluation

`evaluation_dataset.txt` is JSON Lines despite its `.txt` extension. It contains nine
questions spanning overview, emergency, disaster analysis, crew, technical, and timeline
categories across all three missions.

```bash
python batch_evaluate.py \
  --dataset ./evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --output ./evaluation_results/report.json
```

For every record, this performs retrieval, answer generation, and RAGAS evaluation.
Response Relevancy and Faithfulness are always computed. Because the dataset supplies a
reference answer, BLEU and ROUGE-L are also computed. The JSON report includes per-question
results and the mean for every numeric metric.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite is offline and deterministic: fake OpenAI embeddings are injected while a real
temporary ChromaDB verifies persistence and retrieval behavior. It also tests chunking,
all update modes, metadata, context construction, prompts/history, malformed evaluator
inputs, dataset coverage, metric aggregation, and batch orchestration.

## Error handling and security

- API keys come from `OPENAI_API_KEY`, CLI arguments, or the Streamlit password field and
  are never written to project files.
- Indexing/query commands validate missing keys, paths, empty input, batch sizes, and chunk
  settings with clear errors.
- Individual source failures are logged and counted without corrupting successful files.
- Generated databases, logs, environments, and evaluation outputs are excluded by
  `.gitignore`.
