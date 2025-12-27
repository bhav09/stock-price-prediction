
# Stock Price Prediction (RNN/LSTM, MLflow, Streamlit)

A teaching-first, reproducible project structure that covers:
- EDA, baselines, RNN/LSTM training with MLflow tracking.
- Time-aware splits, leakage guardrails, and walk-forward validation.
- Batch and single-point inference via a Streamlit app.

## Data source (this repo)

- This project is configured to run on the included CSV: `RELI Historical Data.csv` (Reliance Industries).

## quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or conda
pip install -r requirements.txt

# run app
streamlit run app.py
```

## Docker (optional)

```bash
docker build -t stock-price-prediction .
docker run --rm -p 8501:8501 stock-price-prediction
```
