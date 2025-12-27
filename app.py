import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = "RELI Historical Data.csv"
MODELS_DIR = Path("models")
STATE_PATH = MODELS_DIR / "reliance_lstm_state.pt"
META_PATH = MODELS_DIR / "reliance_lstm_meta.json"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS – CLEAN LIGHT THEME WITH IMPROVED READABILITY
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global light background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    background-color: #fafbfc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main container padding */
.block-container {
    padding-top: 2rem !important;
    max-width: 1100px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * {
    color: #374151 !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    color: #fff;
    text-align: center;
}
.hero-banner h1 {
    font-size: 2.25rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 1.1rem;
    opacity: 0.92;
    margin: 0;
    font-weight: 400;
}

/* Section titles */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1f2937;
    margin: 2.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #e5e7eb 0%, transparent 100%);
    margin-left: 1rem;
}

/* Info box */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #1e40af;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Feature cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.5rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}
.feature-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}
.feature-icon.blue { background: #dbeafe; }
.feature-icon.purple { background: #ede9fe; }
.feature-icon.green { background: #d1fae5; }
.feature-icon.amber { background: #fef3c7; }
.feature-icon.rose { background: #ffe4e6; }
.feature-icon.cyan { background: #cffafe; }
.feature-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.4rem;
}
.feature-desc {
    font-size: 0.9rem;
    color: #6b7280;
    line-height: 1.55;
}

/* How it works steps */
.steps-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin: 1.5rem 0;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem;
    position: relative;
}
.step-item.done {
    border-left: 3px solid #22c55e;
}
.step-item.active {
    border-left: 3px solid #3b82f6;
    background: #f8fafc;
}
.step-number {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-content h4 {
    margin: 0 0 0.25rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #1f2937;
}
.step-content p {
    margin: 0;
    font-size: 0.9rem;
    color: #6b7280;
    line-height: 1.5;
}

/* Tech stack pills */
.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
}
.tech-pill {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 50px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #374151;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.tech-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.tech-pill .dot.blue { background: #3b82f6; }
.tech-pill .dot.purple { background: #8b5cf6; }
.tech-pill .dot.green { background: #10b981; }
.tech-pill .dot.amber { background: #f59e0b; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem;
}
div[data-testid="stMetric"] label {
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1f2937 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

/* Highlight metric (for important values) */
.highlight-metric {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
}
.highlight-metric label, .highlight-metric [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Chart container */
div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem;
}

/* Expander */
details {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
}
details summary {
    font-weight: 600 !important;
    color: #374151 !important;
}

/* CTA button style info box */
.cta-box {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 2rem 0;
}
.cta-box p {
    color: #fff;
    font-size: 1rem;
    margin: 0;
}
.cta-box strong {
    font-weight: 600;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #9ca3af !important;
    margin: 1.5rem 0 0.75rem 0;
}

/* Legend for chart */
.chart-legend {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #6b7280;
}
.chart-legend span {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 3px;
}
.legend-dot.historical { background: #3b82f6; }
.legend-dot.clean { background: #22c55e; }
.legend-dot.noisy { background: #f59e0b; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    """Simple LSTM regressor matching the training notebook."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def parse_volume(vol_str):
    """Parse volume strings like '8.96M' to numeric values."""
    if pd.isna(vol_str) or vol_str == '-':
        return 0.0
    vol_str = str(vol_str).strip()
    if vol_str.endswith('M'):
        return float(vol_str[:-1].replace(',', '')) * 1_000_000
    elif vol_str.endswith('K'):
        return float(vol_str[:-1].replace(',', '')) * 1_000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1].replace(',', '')) * 1_000_000_000
    else:
        try:
            return float(vol_str.replace(',', ''))
        except:
            return 0.0

def parse_price(price_str):
    """Parse price strings like '1,567.50' to float."""
    if pd.isna(price_str) or price_str == '-':
        return np.nan
    return float(str(price_str).replace(',', ''))

def load_reliance_ohlcv(path: Path | None = None) -> pd.DataFrame:
    """Load the Reliance CSV and return an OHLCV DataFrame."""
    if path is None:
        project_root = Path.cwd()
        csv_root = project_root / DATA_FILE
        csv_notebooks = project_root / "notebooks" / DATA_FILE
        path = csv_notebooks if csv_notebooks.exists() else csv_root

    df_raw = pd.read_csv(path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    
    # Parse prices (remove commas)
    for col in ['Price', 'Open', 'High', 'Low']:
        df_raw[col] = df_raw[col].apply(parse_price)
    
    # Parse volume (e.g., "8.96M")
    df_raw['Vol.'] = df_raw['Vol.'].apply(parse_volume)
    
    df_raw = df_raw.rename(
        columns={
            "Date": "date",
            "Price": "close",  # Price is the closing price
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Vol.": "volume",
        }
    )
    df_raw["date"] = pd.to_datetime(df_raw["date"], format="%d-%m-%Y")
    df_raw = df_raw.sort_values("date").set_index("date")

    df = df_raw[["open", "high", "low", "close", "volume"]].copy()
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = RSIIndicator(close=out["close"].squeeze(), window=14).rsi()
    macd = MACD(close=out["close"].squeeze())
    out["macd"] = macd.macd()
    out["ema_10"] = EMAIndicator(close=out["close"].squeeze(), window=10).ema_indicator()
    out["ema_20"] = EMAIndicator(close=out["close"].squeeze(), window=20).ema_indicator()
    bb = BollingerBands(close=out["close"].squeeze(), window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out.index.dayofweek
    out["dom"] = out.index.day
    out["is_month_end"] = out.index.is_month_end.astype(int)
    return out


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Build next-step log-return target y."""
    out = df.copy()
    price = out["close"]
    out["y"] = (np.log(price) - np.log(price.shift(1))).shift(-1)
    return out


def to_supervised(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, lookback: int
):
    X, y, idx = [], [], []
    values = df[feature_cols + [target_col]].values
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i, :-1])
        y.append(values[i, -1])
        idx.append(df.index[i])
    return np.array(X), np.array(y), pd.Index(idx)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    """Load the trained PyTorch model and metadata."""
    if not STATE_PATH.exists() or not META_PATH.exists():
        st.error(
            "Saved model not found. Run the LSTM training notebook first to create "
            "'models/reliance_lstm_state.pt' and 'models/reliance_lstm_meta.json'."
        )
        st.stop()

    with META_PATH.open() as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    model = LSTMRegressor(
        input_size=len(feature_cols),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
    )
    state = torch.load(STATE_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


@st.cache_data(show_spinner=False)
def build_reliance_dataset():
    """End-to-end feature pipeline on Reliance CSV."""
    df = load_reliance_ohlcv()
    df = add_technical_indicators(df)
    df = add_calendar(df)
    df = build_target(df)
    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reliance Stock Forecast", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 📊 Reliance Forecast")
    st.markdown("---")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📈 Predictions"],
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Reliance Industries Price Forecaster</h1>
            <p>A neural network's best guess at where the market goes next</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # What is this app?
    st.markdown('<div class="section-title">🎯 What is this app?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Ever wondered where Reliance stock might be heading tomorrow? This app takes a crack at that question.
            <br><br>
            We trained a neural network on a year of market data – it learned patterns in how prices move day-to-day. 
            Now it makes predictions you can explore, tweak, and (hopefully) find useful.
            <br><br>
            <strong>A word of caution:</strong> Markets are messy. This is a learning project, not financial advice. 
            Use it to explore, but always do your own homework before putting real money anywhere.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown('<div class="section-title">🔄 How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box" style="background: #f0fdf4; border-color: #22c55e; color: #166534;">
            ✅ <strong>Ready to go!</strong> The model is already trained and loaded. No waiting around – 
            head to Predictions and start exploring right away.
        </div>
        <div class="steps-container">
            <div class="step-item done">
                <div class="step-number">1</div>
                <div class="step-content">
                    <h4>Data collection ✓</h4>
                    <p>We grabbed a year of Reliance stock prices – opens, closes, highs, lows, volumes. The boring but essential stuff.</p>
                </div>
            </div>
            <div class="step-item done">
                <div class="step-number">2</div>
                <div class="step-content">
                    <h4>Feature engineering ✓</h4>
                    <p>Calculated RSI, MACD, EMAs, Bollinger Bands – the same indicators traders look at. Gives the model something meaningful to chew on.</p>
                </div>
            </div>
            <div class="step-item done">
                <div class="step-number">3</div>
                <div class="step-content">
                    <h4>Model training ✓</h4>
                    <p>An LSTM neural network learned from this data. It figured out patterns between today's indicators and tomorrow's price move. Took a while, but it's done.</p>
                </div>
            </div>
            <div class="step-item active">
                <div class="step-number">4</div>
                <div class="step-content">
                    <h4>Your turn</h4>
                    <p>Pick dates, set your forecast horizon, play with uncertainty. The model runs predictions in real-time as you adjust settings.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # What you can do
    st.markdown('<div class="section-title">✨ Things you can try</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon blue">📅</div>
                <div class="feature-title">Pick your horizon</div>
                <div class="feature-desc">Forecast 5 days out, or 60. Shorter horizons tend to be more reliable, but it's fun to see where longer trends might go.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon purple">🔍</div>
                <div class="feature-title">Dig into history</div>
                <div class="feature-desc">Scroll back through past prices. See how the market actually moved vs. what patterns looked like at the time.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon green">📊</div>
                <div class="feature-title">Watch the chart</div>
                <div class="feature-desc">The main chart overlays predictions on real prices. You'll see exactly where the model's forecast begins.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon amber">🎲</div>
                <div class="feature-title">Add some chaos</div>
                <div class="feature-desc">Crank up the "uncertainty" slider to simulate market randomness. It shows how small daily surprises can compound over time.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tech stack (simplified)
    st.markdown('<div class="section-title">🛠️ Built with</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tech-stack">
            <div class="tech-pill"><span class="dot blue"></span>Python</div>
            <div class="tech-pill"><span class="dot purple"></span>PyTorch (AI/ML)</div>
            <div class="tech-pill"><span class="dot green"></span>Pandas (Data)</div>
            <div class="tech-pill"><span class="dot amber"></span>Streamlit (Interface)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA
    st.markdown(
        """
        <div class="cta-box">
            <p>👈 Alright, enough reading – click <strong>"📈 Predictions"</strong> in the sidebar and see what the model thinks!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
else:
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Price Predictions</h1>
            <p>See where Reliance stock might be heading based on AI analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, meta = load_model_and_meta()
    df = build_reliance_dataset()

    feature_cols: list[str] = meta["feature_cols"]
    lookback: int = meta["lookback"]

    # ── SIDEBAR CONTROLS ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-section">Forecast Settings</p>', unsafe_allow_html=True)

        max_forecast_days = st.number_input(
            "Days to predict ahead",
            min_value=1,
            max_value=7,
            value=5,
            step=1,
            help="How many business days into the future should we forecast? (Max 7 days)",
        )

        st.markdown('<p class="sidebar-section">Date Range to View</p>', unsafe_allow_html=True)

        hist_start = df.index.min().date()
        hist_end = df.index.max().date()

        view_start = st.date_input(
            "From",
            value=max(hist_start, hist_end - pd.Timedelta(days=14)),
            min_value=hist_start,
            max_value=hist_end,
            help="Start date for the chart (you can pick any date; chart shows up to ~2 weeks of history plus forecast).",
        )

        st.markdown('<p class="sidebar-section">Uncertainty</p>', unsafe_allow_html=True)

        noise_level = st.slider(
            "Add randomness",
            min_value=0.0,
            max_value=0.05,
            value=0.0,
            step=0.005,
            format="%.3f",
            help="Simulate market uncertainty by adding random variation to predictions. Higher = more variation.",
        )

    # ── SCALING & SEQUENCES ──────────────────────────────────────────────────
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    X, y, idx = to_supervised(df_scaled, feature_cols, "y", lookback)

    # ── NEXT-DAY PREDICTION ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Tomorrow\'s Prediction</div>', unsafe_allow_html=True)

    if len(X) == 0:
        st.warning("Not enough data to make predictions.")
    else:
        last_seq = torch.from_numpy(X[-1:]).float()
        with torch.no_grad():
            pred_return = model(last_seq).cpu().numpy().ravel()[0]

        last_close = float(df["close"].iloc[-1])
        implied_next_close = last_close * float(np.exp(pred_return))
        change_pct = (implied_next_close - last_close) / last_close * 100
        direction = "📈 Up" if change_pct > 0 else "📉 Down" if change_pct < 0 else "➡️ Flat"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"₹{last_close:,.2f}")
        with col2:
            st.metric("Predicted Next Close", f"₹{implied_next_close:,.2f}", delta=f"{change_pct:+.2f}%")
        with col3:
            st.metric("Direction", direction)

        st.markdown(
            """
            <div class="info-box">
                <strong>What does this mean?</strong> Based on recent market patterns, our AI predicts the Reliance
                closing price for the next trading day. The percentage shows the expected change from today's close.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── MULTI-DAY FORECAST ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Extended Forecast</div>', unsafe_allow_html=True)

    # We treat everything in df as historical. We then forecast *ahead* of the last date.
    hist_end_ts = df.index.max()
    hist_end = hist_end_ts.date()
    n_future = min(max_forecast_days, 7)

    df_ext = df.copy()
    df_scaled_ext = df_scaled.copy()
    future_dates: list[pd.Timestamp] = []
    future_pred_returns: list[float] = []
    future_pred_closes: list[float] = []
    future_ma_closes: list[float] = []

    if n_future > 0 and len(df_ext) >= lookback:
        future_idx_full = pd.bdate_range(
            start=hist_end_ts + pd.Timedelta(days=1), periods=n_future, freq="B"
        )
        # EMA(20) parameters
        alpha_ema20 = 2.0 / (20 + 1)
        last_ema20 = float(df_ext["ema_20"].iloc[-1])

        for future_date in future_idx_full:
            last_close = float(df_ext["close"].iloc[-1])

            # LSTM prediction based on last lookback window
            seq_features = df_scaled_ext[feature_cols].values[-lookback:]
            X_last = torch.from_numpy(seq_features[None, :, :]).float()
            with torch.no_grad():
                pred_ret = float(model(X_last).cpu().numpy().ravel()[0])

            new_close = last_close * float(np.exp(pred_ret))

            # Update EMA20 as a baseline prediction series
            last_ema20 = alpha_ema20 * new_close + (1.0 - alpha_ema20) * last_ema20

            new_row = df_ext.iloc[-1].copy()
            new_row.name = future_date
            new_row["close"] = new_close
            new_row["ema_20"] = last_ema20

            df_ext = pd.concat([df_ext, new_row.to_frame().T])

            new_feat_df = new_row[feature_cols].to_frame().T
            new_feat_scaled = scaler.transform(new_feat_df)
            new_scaled_row = new_row.copy()
            new_scaled_row[feature_cols] = new_feat_scaled[0]
            df_scaled_ext = pd.concat([df_scaled_ext, new_scaled_row.to_frame().T])

            future_dates.append(future_date)
            future_pred_returns.append(pred_ret)
            future_pred_closes.append(new_close)
            future_ma_closes.append(last_ema20)

        forecast_df = pd.DataFrame(
            {
                "pred_log_return": future_pred_returns,
                "pred_close": future_pred_closes,
                "ma_pred": future_ma_closes,
            },
            index=pd.DatetimeIndex(future_dates, name="date"),
        )

        if noise_level > 0.0:
            rng = np.random.default_rng()
            noise = rng.normal(loc=0.0, scale=noise_level, size=len(future_pred_returns))
            noisy_pred_closes = []
            last_clean_close = float(df["close"].iloc[-1])
            for i, (clean_ret, eps) in enumerate(zip(future_pred_returns, noise)):
                prev_close = last_clean_close if i == 0 else noisy_pred_closes[-1]
                noisy_pred_closes.append(prev_close * float(np.exp(clean_ret + eps)))
            forecast_df["pred_close_noisy"] = noisy_pred_closes
        else:
            forecast_df["pred_close_noisy"] = np.nan
    else:
        forecast_df = pd.DataFrame(
            columns=["pred_log_return", "pred_close", "ma_pred", "pred_close_noisy"]
        )
        future_idx_full = pd.DatetimeIndex([])

    # Extend view to include forecast horizon (after the last historical day)
    if n_future > 0 and len(future_idx_full) > 0:
        last_future_date = future_idx_full[-1].date()
        effective_end = last_future_date
    else:
        effective_end = hist_end

    view_mask = (df_ext.index.date >= view_start) & (df_ext.index.date <= effective_end)
    df_view = df_ext.loc[view_mask].copy()

    is_future = df_view.index > hist_end_ts

    forecast_clean_series = pd.Series(index=df_view.index, dtype=float)
    forecast_noisy_series = pd.Series(index=df_view.index, dtype=float)
    ma_series = pd.Series(index=df_view.index, dtype=float)
    if not forecast_df.empty:
        for ts, row in forecast_df.iterrows():
            if ts in forecast_clean_series.index:
                forecast_clean_series.loc[ts] = row["pred_close"]
                ma_series.loc[ts] = row["ma_pred"]
                if not np.isnan(row["pred_close_noisy"]):
                    forecast_noisy_series.loc[ts] = row["pred_close_noisy"]

    # Build plot dataframe:
    # - Actual Price: historical only
    # - LSTM / MA / Noise: only in forecast region (future)
    plot_df = pd.DataFrame(
        {
            "Actual Price": df_view["close"].where(~is_future),
            "LSTM Prediction": forecast_clean_series,
            "MA Prediction": ma_series,
            "With Noise": forecast_noisy_series,
        },
        index=df_view.index,
    )

    # Y-axis range:
    # - min based on LSTM predictions (so deviations in forecast are visible)
    # - max based on actual prices (anchor to real observed highs)
    lstm_values = plot_df["LSTM Prediction"].values
    lstm_values = lstm_values[~np.isnan(lstm_values)]

    actual_values = plot_df["Actual Price"].values
    actual_values = actual_values[~np.isnan(actual_values)]

    if len(lstm_values) > 0 and len(actual_values) > 0:
        y_min = float(np.min(lstm_values)) - 25
        y_max = float(np.max(actual_values)) + 0  # no extra padding on top, per request
    else:
        y_min, y_max = 0, 100

    # Melt dataframe for Altair
    plot_df_reset = plot_df.reset_index()
    plot_df_reset.columns = ["Date", "Actual Price", "LSTM Prediction", "MA Prediction", "With Noise"]
    plot_melted = plot_df_reset.melt(id_vars=["Date"], var_name="Series", value_name="Price")

    # Define color mapping
    color_scale = alt.Scale(
        domain=["Actual Price", "LSTM Prediction", "MA Prediction", "With Noise"],
        range=["#3b82f6", "#22c55e", "#8b5cf6", "#f59e0b"],
    )

    # Create Altair chart with custom Y-axis
    chart = alt.Chart(plot_melted).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Price:Q", title="Price (₹)", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="Legend", orient="bottom")),
        strokeDash=alt.condition(
            alt.FieldOneOfPredicate(field="Series", oneOf=["MA Prediction"]),
            alt.value([5, 5]),  # Dashed line for MA prediction (baseline)
            alt.value([0]),  # Solid line for others
        ),
    ).properties(
        height=400
    ).configure_axis(
        grid=True,
        gridOpacity=0.3,
    )

    st.altair_chart(chart, use_container_width=True)

    # Explanation
    st.markdown(
        """
        <div class="info-box">
            <strong>Reading this chart:</strong> All prices shown in INR (₹).
            <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
                <li><strong style="color: #3b82f6;">Blue</strong> = Actual historical prices (up to the latest date)</li>
                <li><strong style="color: #22c55e;">Green</strong> = LSTM model's prediction for future days</li>
                <li><strong style="color: #8b5cf6;">Purple (dashed)</strong> = Moving Average baseline prediction for the same future days</li>
                <li><strong style="color: #f59e0b;">Orange</strong> = LSTM prediction with added randomness (noise)</li>
            </ul>
            <p style="margin-top: 0.5rem;"><em>All lines are plotted on the same timeline so you can see how the model and MA baseline extend beyond the historical data.</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Forecast data table
    if not forecast_df.empty:
        with st.expander("📋 View forecast data"):
            display_df = forecast_df[["pred_close", "ma_pred", "pred_close_noisy"]].copy()
            display_df.columns = ["LSTM Prediction", "MA Prediction", "With Noise"]
            display_df.index.name = "Date"
            st.dataframe(display_df.style.format("₹{:,.2f}"), use_container_width=True)

    # Data preview
    with st.expander("📊 View raw market data"):
        st.dataframe(df.tail(15), use_container_width=True)
