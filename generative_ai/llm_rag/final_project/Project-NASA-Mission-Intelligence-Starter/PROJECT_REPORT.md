# Project Report

## Implementation summary

The starter TODOs were replaced with a complete five-component RAG application plus an
end-to-end batch evaluator and offline test suite.

The embedding pipeline normalizes OCR whitespace, creates sentence-aware character chunks
with exact overlap, calls `text-embedding-3-small` in batches, and writes explicit vectors
and rich metadata to persistent ChromaDB. Stable mission/source/chunk IDs make `skip`,
`update`, and `replace` deterministic. Update also removes stale tail chunks when source
content becomes shorter.

Retrieval connects to a selected persistent collection with the matching OpenAI embedding
function. It applies an optional `mission` metadata filter, requests extra candidates,
removes duplicate OCR excerpts, sorts by cosine distance, and returns the configured top-k.
The context formatter emits distinct source headers containing mission, source, category,
chunk, and similarity distance.

The LLM client supplies a NASA archive specialist system prompt, a bounded valid chat
history, the current question, and only the current retrieved context. It explicitly treats
history as conversational—not evidentiary—and requires inline source labels or an
insufficient-context response.

The RAGAS evaluator calculates Response Relevancy and Faithfulness for every valid triple.
When the batch dataset's expected answer is present, it also calculates BLEU and ROUGE-L.
A deterministic lexical context coverage score is included as a low-cost diagnostic. Empty
or malformed input returns a clear structured error.

The Streamlit interface exposes collection, model, top-k, mission filter, and RAGAS controls;
preserves conversation history; shows retrieved evidence; and displays evaluation metrics.

## Dataset audit

All 12 bundled source files were scanned successfully as UTF-8 text: six Apollo 11 files,
three Apollo 13 files, and three Challenger files. Together they contain 6,223,821 characters
and 239,265 lines. The evaluation set contains nine valid JSONL records across all missions and
the six rubric categories: overview, emergency, disaster analysis, crew, technical, and
timeline.

## Verification

The automated suite uses a real temporary ChromaDB and injected deterministic embedding
responses. It verifies:

- maximum chunk size and exact consecutive overlap;
- metadata extraction and stable identifiers;
- persisted add, skip, update, replace, query, and aggregate statistics;
- optional mission filtering, score sorting, and duplicate removal;
- source-attributed context formatting;
- grounded prompt composition and conversation-history filtering;
- evaluator validation, dataset parsing, aggregation, and batch orchestration;
- statistics CLI operation without an API key.

Live OpenAI indexing and RAGAS calls require the project owner's API key and incur API usage.
They are therefore kept as explicit commands rather than embedded secrets or automatic paid
test steps.

The complete real corpus was also indexed once into a disposable ChromaDB with deterministic
offline test embeddings. All 12 files completed with zero errors and produced 17,467 chunks:
9,490 Apollo 11, 7,026 Apollo 13, and 951 Challenger. Mission-filtered queries returned only
the selected mission, and the database reported all 12 unique sources. The disposable index
was removed automatically after the check.
