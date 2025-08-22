# app.py
import os, math, requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime, timezone, timedelta
import numpy as np

# ======================
# CONFIG
# ======================
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")  # replace with your key
BASE_URL = "https://www.alphavantage.co/query"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")          # optional
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "you@example.com")  # SEC asks for a UA

COEFS = {
    "rev_growth_1y": 0.9,
    "ebit_margin": 1.1,
    "debt_to_equity": -0.8,
    "interest_coverage": 0.7,
    "price_momentum_3m": 0.6,
    "volatility_30d": -0.5,
    "news_sentiment_7d": 0.5,
}
BIAS = -0.2

# basic CIK map for SEC companyfacts (US tickers only). Add more as needed.
CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
}

# ======================
# HELPERS / MODEL
# ======================
def clamp(x, lo=-1, hi=1):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(lo, min(hi, x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if -700 < x < 700 else (0 if x < 0 else 1)

def score_with_attribution(features: dict):
    linear = BIAS
    attrs = []
    for k, w in COEFS.items():
        v = float(features.get(k, 0.0))
        c = v * w
        linear += c
        attrs.append({"feature": k, "value": v, "weight": w, "contribution": c})
    return sigmoid(linear), sorted(attrs, key=lambda a: abs(a["contribution"]), reverse=True)

def score_to_band(s: float) -> str:
    if s >= 0.8: return "AAA"
    if s >= 0.7: return "AA"
    if s >= 0.6: return "A"
    if s >= 0.5: return "BBB"
    if s >= 0.4: return "BB"
    if s >= 0.3: return "B"
    return "CCC"

# ======================
# DATA FETCHERS (fault tolerant)
# ======================
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_price_df(symbol: str, period: str = "1y"):
    """Return a cleaned daily DataFrame from yfinance; flatten columns if necessary."""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df["Close_adj"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    return df.sort_index()

@st.cache_data(ttl=60*60, show_spinner=False)
def get_fundamentals_alpha(symbol: str):
    # Primary: Alpha Vantage
    try:
        url = f"{BASE_URL}?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}"
        d = requests.get(url, timeout=20).json()
        if isinstance(d, dict) and d.get("Symbol"):
            rev = float(d.get("QuarterlyRevenueGrowthYOY", 0) or 0)
            margin = float(d.get("ProfitMargin", 0) or 0)
            de = float(d.get("DebtToEquity", 0) or 0)
            ebitda = float(d.get("EBITDA", 0) or 0)
            int_exp = float(d.get("InterestExpense", 1) or 1)
            intcov = ebitda / int_exp if int_exp else 0
            return {
                "revenue_growth_1y": rev,
                "ebit_margin": margin,
                "de_ratio": de,
                "int_cov": intcov,
                "_sources": ["AlphaVantage"],
            }
    except Exception:
        pass
    # Fallback: yfinance info
    try:
        info = yf.Ticker(symbol).info
        rev = info.get("revenueGrowth", 0) or 0
        margin = info.get("profitMargins", 0) or 0
        de = info.get("debtToEquity", 0) or 0
        ebitda = float(info.get("ebitda", 0) or 0)
        int_exp = float(info.get("interestExpense", 1) or 1)
        intcov = ebitda / int_exp if int_exp else 0
        return {
            "revenue_growth_1y": rev,
            "ebit_margin": margin,
            "de_ratio": de,
            "int_cov": intcov,
            "_sources": ["yfinance"],
        }
    except Exception:
        return {
            "revenue_growth_1y": 0, "ebit_margin": 0, "de_ratio": 0, "int_cov": 0,
            "_sources": ["funds-error"]
        }

@st.cache_data(ttl=60*60, show_spinner=False)
def get_news_sentiment_alpha(symbol: str):
    # Unstructured (free/public) via Alpha Vantage News API
    try:
        url = f"{BASE_URL}?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
        d = requests.get(url, timeout=20).json()
        if isinstance(d, dict) and d.get("feed"):
            headlines = [f.get("title", "") for f in d.get("feed")[:10]]
            # some responses contain 'overall_sentiment_score'; be defensive
            overall = d.get("overall_sentiment_score", 0.0)
            try:
                sentiment = float(overall)
            except Exception:
                sentiment = 0.0
            return {"sentiment": sentiment, "headlines": headlines, "_sources": ["AlphaVantage-News"]}
    except Exception:
        pass
    return {"sentiment": 0.0, "headlines": [f"{symbol}: no recent news"], "_sources": ["news-fallback"]}

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_fred_series(series_id: str, start: str = "2018-01-01"):
    # Optional macro series from FRED
    if not FRED_API_KEY:
        return pd.DataFrame()
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
        r = requests.get(url, timeout=20)
        js = r.json()
        obs = js.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty: return df
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date").sort_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_sec_companyfacts(symbol: str):
    # Optional SEC companyfacts for US tickers (requires CIK)
    cik = CIK_MAP.get(symbol.upper())
    if not cik:
        return {}
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def sec_companyfacts_features(cf: dict) -> dict:
    try:
        usgaap = cf.get("facts", {}).get("us-gaap", {})
        def last_val(tag):
            units = usgaap.get(tag, {}).get("units", {})
            for k in ("USD", "USD/shares", "pure"):
                if k in units and units[k]:
                    vals = sorted(units[k], key=lambda x: x.get("end", ""))
                    return float(vals[-1].get("val", 0.0))
            return 0.0
        net_income = last_val("NetIncomeLoss")
        revenue = last_val("Revenues") or last_val("SalesRevenueNet") or last_val("RevenueFromContractWithCustomerExcludingAssessedTax")
        assets = last_val("Assets")
        liabilities = last_val("Liabilities")
        margin = (net_income / revenue) if revenue else 0.0
        leverage = (liabilities / assets) if assets else 0.0
        return {
            "sec_net_income_margin": clamp(margin / 0.2),
            "sec_leverage": clamp(-leverage),
        }
    except Exception:
        return {}

# ======================
# FEATURE ENGINEERING
# ======================
def build_features_from_history(df_prices: pd.DataFrame, idx: int):
    close = df_prices["Close_adj"].iloc[:idx + 1]
    if idx >= 60:
        prev_60 = float(close.iloc[-61])
        mom = (float(close.iloc[-1]) - prev_60) / prev_60 if prev_60 != 0 else 0.0
    else:
        mom = 0.0
    if len(close) >= 30:
        vol = close.pct_change().dropna().rolling(30).std().iloc[-1] * (252 ** 0.5)
        vol = float(vol) if not pd.isna(vol) else 0.0
    else:
        vol = 0.0
    return {"mom_3m": mom, "vol_30d": vol}

def fundamentals_to_features(f: dict) -> dict:
    return {
        "rev_growth_1y": clamp(f.get("revenue_growth_1y", 0)),
        "ebit_margin": clamp(f.get("ebit_margin", 0) / 0.25),
        "debt_to_equity": clamp(-f.get("de_ratio", 0) / 2.0),
        "interest_coverage": clamp(f.get("int_cov", 0) / 10.0),
    }

# ======================
# SIMPLE ONLINE (INCREMENTAL) MODEL — no extra libs
# learns to predict a proxy risk target; adaptive score = 1 - tanh(pred)
# ======================
def compute_proxy_risk_target(close_series: pd.Series, horizon_days: int = 20):
    close = close_series.copy()
    fwd = close.shift(-horizon_days)
    ret_fwd = (fwd / close - 1.0)
    vol = close.pct_change().rolling(20).std()
    dd = (close.rolling(60).max() - close) / close
    risk = 0.5 * vol.fillna(0) + 0.5 * dd.fillna(0) - 0.5 * ret_fwd.fillna(0)
    r = (risk - risk.quantile(0.05)) / (risk.quantile(0.95) - risk.quantile(0.05) + 1e-9)
    return r.clip(0, 1)

def online_sgd_scores(feature_rows: list[dict], y_proxy: pd.Series, lr: float = 0.05):
    # initialize weights to zeros on first row's features
    if not feature_rows:
        return pd.Series(dtype=float)
    keys = list(feature_rows[0].keys())
    w = {k: 0.0 for k in keys}
    b = 0.0
    preds = []
    index = y_proxy.index[:len(feature_rows)]
    for i, t in enumerate(index):
        x = feature_rows[i]
        # linear prediction
        yhat = b + sum(w[k]*x.get(k,0.0) for k in keys)
        # map to [0,1] as a risk-like value
        yhat_risk = 1/(1+math.exp(-yhat))
        # convert to adaptive credit score
        score = float(1.0 - np.tanh(max(0.0, yhat_risk)))
        preds.append(score)
        # learn step on proxy risk
        ytrue = float(y_proxy.iloc[i] if not pd.isna(y_proxy.iloc[i]) else 0.0)
        # squared error gradient wrt yhat_risk; chain rule through sigmoid
        grad = 2*(yhat_risk - ytrue) * yhat_risk * (1 - yhat_risk)
        for k in keys:
            w[k] -= lr * grad * x.get(k, 0.0)
        b -= lr * grad
    return pd.Series(preds, index=index)

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Explainable Credit Scorecard", layout="wide")

st.markdown("""
<style>
.stApp { background: white; color: black; }
div[data-testid="stMetric"] { background: #f6f8fb; border-radius: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Explainable Credit Scorecard — Multi-Source & Adaptive (Non-LLM)")

col_left, col_right = st.columns([3,1])
with col_right:
    lookback_days = st.slider("Lookback (days)", 60, 730, 180, 30)
    smooth_window = st.slider("Smoothing window (days)", 1, 21, 5, 1)
    symbol = st.text_input("Ticker symbol", "AAPL").upper()
    use_sec = st.checkbox("Use SEC EDGAR features (US tickers)", value=True)
    use_fred = st.checkbox("Show FRED macro series (optional)", value=False)
    fred_series_id = st.text_input("FRED series (e.g., DCOILWTICO)", "DCOILWTICO")
    alert_drop = st.slider("Alert threshold (score drop)", 0.05, 0.30, 0.10, 0.01)
    horizon = st.slider("Adaptive model horizon (days)", 5, 60, 20, 1)

with col_left:
    st.markdown("Enter a ticker (e.g., AAPL, MSFT, TSLA, RELIANCE.NS). Toggle SEC/FRED on the right.")

# Prices
period = f"{max(lookback_days, 120)}d"
df_prices = fetch_price_df(symbol, period=period)

if df_prices.empty:
    st.error("No price data found for that symbol. Check ticker or network. Try including exchange suffix like .NS for NSE.")
    st.stop()

# Fundamentals and news
funds_raw = get_fundamentals_alpha(symbol)
news = get_news_sentiment_alpha(symbol)
funds = fundamentals_to_features(funds_raw)

# Optional SEC features
sec_feats = {}
if use_sec:
    cf = fetch_sec_companyfacts(symbol)
    if cf:
        sec_feats = sec_companyfacts_features(cf)

# Optional Macro (display only)
fred_df = fetch_fred_series(fred_series_id) if use_fred else pd.DataFrame()

# Build historical scores (interpretable linear model)
start_idx = 60 if len(df_prices) > 60 else 0
hist = []

if not df_prices.empty:
    price_index = list(df_prices.index)
    for i in range(start_idx, len(price_index)):
        date = price_index[i]
        m = build_features_from_history(df_prices, i)
        feats = {
            **funds,
            "price_momentum_3m": clamp(m.get("mom_3m", 0) / 0.3),
            "volatility_30d": clamp(-m.get("vol_30d", 0) / 0.6),
            "news_sentiment_7d": clamp(news.get("sentiment", 0)),
        }
        # include SEC features if available
        feats.update(sec_feats)
        s, _ = score_with_attribution(feats)
        hist.append({"Date": date, "score": s})

if not hist:
    st.error("No historical scores could be generated (likely due to missing price data). Please check ticker or network.")
    st.stop()

hist_df = pd.DataFrame(hist).set_index("Date").sort_index()

# Subset window
cutoff = pd.Timestamp(datetime.now() - timedelta(days=lookback_days))
hist_df = hist_df[hist_df.index >= cutoff]

# ======================
# Adaptive model (incremental learning; no extra libs)
# ======================
# Build feature rows over time for online SGD
feature_rows = []
price_index = list(df_prices.index)
for i in range(start_idx, len(price_index)):
    m = build_features_from_history(df_prices, i)
    feats = {
        **funds,
        "price_momentum_3m": clamp(m.get("mom_3m", 0) / 0.3),
        "volatility_30d": clamp(-m.get("vol_30d", 0) / 0.6),
        "news_sentiment_7d": clamp(news.get("sentiment", 0)),
    }
    feats.update(sec_feats)
    feature_rows.append(feats)

# Compute proxy risk target from prices (future return/vol/dd)
y_proxy = compute_proxy_risk_target(df_prices["Close_adj"], horizon_days=horizon)
adaptive_series = online_sgd_scores(feature_rows, y_proxy)

# Align adaptive series to hist_df index
if not adaptive_series.empty:
    adaptive_series = adaptive_series.reindex(hist_df.index, method="nearest")

# ======================
# Charts
# ======================
if hist_df.empty:
    st.warning("Not enough historical trading data for the selected lookback. Try smaller lookback.")
else:
    # Smooth linears
    if smooth_window > 1:
        hist_df["score_smooth"] = hist_df["score"].rolling(window=smooth_window, min_periods=1).mean()
        y_lin = "score_smooth"
    else:
        y_lin = "score"

    # Compose score_line with proper Date column (fixes date/Date error)
    score_line = pd.DataFrame({
        "Date": hist_df.index,
        "linear_score": hist_df[y_lin].values,
    })
    if not adaptive_series.empty:
        score_line["adaptive_score"] = adaptive_series.reindex(hist_df.index).rolling(smooth_window, min_periods=1).mean().values
    score_line["linear_score_smooth"] = score_line["linear_score"]  # for compatibility with your old UI

    # Trend plot
    y_cols = ["linear_score"]
    if "adaptive_score" in score_line.columns:
        y_cols = ["adaptive_score", "linear_score"]
    pl = px.line(score_line, x="Date", y=y_cols, title=f"Credit Score Trend — {symbol}")
    pl.update_yaxes(range=[0, 1])
    st.plotly_chart(pl, use_container_width=True)

    # Price chart (candlestick)
    recent_prices = df_prices.tail(180).copy()
    if not recent_prices.empty:
        rp = recent_prices.reset_index()
        candle = go.Figure(data=[go.Candlestick(
            x=rp["Date"], open=rp["Open"], high=rp["High"], low=rp["Low"], close=rp["Close"]
        )])
        candle.update_layout(template="plotly_white", height=360, title=f"{symbol} Price (last {len(rp)} days)")
        st.plotly_chart(candle, use_container_width=True)

# KPIs (current)
features_now = {
    **funds,
    "price_momentum_3m": clamp(hist_df["score"].iloc[-1] if not hist_df.empty else 0),   # proxy for momentum
    "volatility_30d": clamp(-(hist_df["score"].iloc[-1] if not hist_df.empty else 0)),   # proxy for volatility
    "news_sentiment_7d": clamp(news.get("sentiment", 0)),
}
features_now.update(sec_feats)
score_now, attrs_now = score_with_attribution(features_now)

st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Current Credit Score", f"{score_now:.3f}")
k2.metric("Rating Band", score_to_band(score_now))
k3.metric("Last Update (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
k4.metric("Data Sources", ", ".join(sorted(set(funds_raw.get("_sources", []) + news.get("_sources", []) + ["yfinance"] + (["SEC"] if sec_feats else [])))))

# Explainability table + bar viz
st.subheader("🔎 Current Feature Contributions (Linear model)")
df_attrs = pd.DataFrame(attrs_now)
st.dataframe(df_attrs, use_container_width=True)
try:
    bar = px.bar(df_attrs.sort_values("contribution", key=lambda s: s.abs(), ascending=True),
                 x="contribution", y="feature", orientation="h",
                 title="Feature contribution (signed)")
    st.plotly_chart(bar, use_container_width=True)
except Exception:
    pass

# Alerts for sudden changes (adaptive)
if "adaptive_score" in score_line.columns and len(score_line) >= 2:
    last = float(score_line["adaptive_score"].iloc[-1])
    prev = float(score_line["adaptive_score"].iloc[-2])
    if (prev - last) > alert_drop:
        st.warning(f"⚠ Sudden score drop: {prev:.2f} → {last:.2f} (Δ {last - prev:+.2f})")

# News
st.subheader("📰 Recent Headlines & Sentiment (Unstructured)")
st.metric("News Sentiment (proxy)", f"{news.get('sentiment', 0):+.3f}")
for h in news.get("headlines", []):
    st.write("•", h)

# Macro (optional)
if use_fred and not fred_df.empty:
    st.subheader(f"🌍 FRED Macro — {fred_series_id}")
    st.line_chart(fred_df["value"].rename(fred_series_id))

# Raw data toggle
if st.button("📂 Show Raw Data (prices + fundamentals + news + SEC)"):
    st.write("Prices (last 10 rows):")
    st.write(df_prices.tail(10))
    st.write("Fundamentals (raw):")
    st.json(funds_raw)
    if sec_feats:
        st.write("SEC features:")
        st.json(sec_feats)
    st.write("News:")
    st.json(news)
