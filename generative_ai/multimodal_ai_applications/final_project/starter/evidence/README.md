# Verification Evidence

## Automated tests

- `pytest-moderation-results.txt`: moderation result tests, including flagged and unflagged `is_flagged` cases for every modality. Result: 27 passed.
- `pytest-full-suite.txt`: complete project test suite, including live Gemini connectivity. Result: 58 passed.

## Moderation evaluations

- `text-evals.txt`: text moderation evaluation. Result: 91.7%.
- `image-evals.txt`: image moderation evaluation. Result: 100.0%.
- `audio-evals.txt`: audio moderation evaluation. Result: 90.0%.
- `video-evals.txt`: video moderation evaluation. Result: 100.0%.

Evaluation scores may vary between runs because model responses are nondeterministic. Every evaluation suite completed without runtime errors.

## Runtime verification

- `runtime-startup.txt`: startup output from `uv run multimodal-moderation`.
- `runtime-smoke-test.txt`: successful HTTP checks for FastAPI, Gradio, and Phoenix.

## Screenshots to add before submission

- `gradio-conversation.png`: a complete conversation with context-aware customer responses.
- `gradio-flagged-message.png`: an unfriendly message blocked by moderation with feedback visible.
- `phoenix-traces.png`: Phoenix showing `conversation`, `chat_turn`, moderation, and customer-agent spans.
- `pytest-passed.png`: terminal showing the complete passing test suite.
