# SentimentScope submission

This folder collects the final project artifacts.

## Verified result

- Best validation accuracy: **78.12%**
- Held-out test accuracy: **76.76%**
- Required threshold: **greater than 75%**
- Training device: Apple M4 GPU through PyTorch MPS
- Training time: approximately 1.7 minutes for three epochs

## Files

- `SentimentScope.ipynb` — completed notebook with all cell outputs
- `sentimentscope_model.pt` — trained checkpoint used for the reported result
- `results_summary.json` — metrics and training history
- `requirements.txt` — pinned dependencies
- `aclImdb_v1.tar.gz` — original IMDB dataset archive

The notebook also contains a reusable `SentimentPredictor` class that loads the
checkpoint and performs sentiment inference on a batch of raw text inputs.

## Run locally

From the submission directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/jupyter notebook SentimentScope.ipynb
```
