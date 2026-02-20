import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import base64
from pathlib import Path

# =========================================================
# Gaza Aid Intelligence — Pro Dashboard (SAFE THEME ✅)
# - Background image (gaza_bg.jpg) + simple overlay (NO z-index tricks)
# - Glass-like main content so everything remains readable
# - Fix sidebar widgets + tags + file uploader visibility
# - KPI + Tabs + Forecast + Gaza Map Heatmap
# =========================================================

DEFAULT_FILE_PATH = "commodities-received-13.xlsx"
BG_IMAGE_PATH = "gaza_bg.jpg"

st.set_page_config(
    page_title="Gaza Aid Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# SAFE BACKGROUND + CSS
# =========================
def img_to_b64(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")

bg_b64 = img_to_b64(BG_IMAGE_PATH)

bg_css = ""
if bg_b64:
    bg_css = f"""
    /* Background on the whole app container */
    [data-testid="stAppViewContainer"] {{
        background-image:
          linear-gradient(rgba(6,20,43,0.55), rgba(6,20,43,0.55)),
          url("data:image/jpg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """

st.markdown(
    f"""
<style>
:root{{
  --ink:#0B1F3A;
  --muted:#5B6B82;
  --navy:#071B33;
  --navy2:#06162B;
  --border: rgba(217,230,255,.85);
  --accent:#2AA3FF;
  --danger:#EF4444;
}}

/* Apply background */
{bg_css}

/* Main content glass panel (SAFE selectors) */
[data-testid="stAppViewContainer"] .main {{
  padding-top: 10px;
}}
[data-testid="stAppViewContainer"] .main .block-container {{
  background: rgba(255,255,255,0.88);
  border: 1px solid rgba(217,230,255,.65);
  border-radius: 18px;
  box-shadow: 0 14px 34px rgba(0,0,0,.18);
  backdrop-filter: blur(6px);
  padding: 26px 26px 40px 26px;
}}

/* Header */
.pro-title{{
  text-align:center; font-weight: 950; letter-spacing:.2px;
  color: #FFFFFF; margin: 6px 0 2px 0;
  text-shadow: 0 10px 26px rgba(0,0,0,.35);
}}
.pro-sub{{
  text-align:center; color: #CFEAFF; margin: 0 0 2px 0; font-weight: 850;
}}
.pro-sub2{{
  text-align:center; color: #86CBFF; margin: 0 0 10px 0; font-weight: 900;
}}
.pro-hr{{
  border: 0; height:1px;
  background: linear-gradient(90deg, transparent, rgba(160,210,255,0.7), transparent);
  margin: 12px 0 14px;
}}

/* Section headings inside panel */
.h-sec{{ color: var(--ink); font-weight: 950; margin: 10px 0 2px; }}
.p-muted{{ color: var(--muted); margin: 0 0 10px; }}

/* Badges */
.badge{{
  display:inline-block; padding: 6px 12px; border-radius: 999px;
  font-weight: 900; font-size: 12px;
  border: 1px solid rgba(210,225,255,0.65);
  background: rgba(7, 23, 51, 0.78);
  color: #EAF2FF;
  box-shadow: 0 10px 26px rgba(0,0,0,.18);
}}

/* KPI cards */
.kpi-grid{{ display:grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
@media(max-width:1200px){{ .kpi-grid{{ grid-template-columns: repeat(2, 1fr);}} }}
.kpi{{
  background: rgba(255,255,255,.95);
  border: 1px solid rgba(217,230,255,.92);
  border-radius: 18px;
  padding: 14px 14px;
  box-shadow: 0 14px 35px rgba(0,0,0,0.12);
}}
.kpi .t{{ color: var(--muted); font-weight: 900; font-size: 12px; text-transform: uppercase; letter-spacing: .6px; }}
.kpi .v{{ color: var(--ink); font-weight: 950; font-size: 26px; margin-top: 6px; }}
.kpi .n{{ color: var(--muted); font-size: 12px; margin-top: 6px; }}

/* Sidebar */
section[data-testid="stSidebar"]{{
  background: linear-gradient(180deg, var(--navy) 0%, var(--navy2) 100%) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}}
section[data-testid="stSidebar"] *{{ color: #EAF2FF !important; }}

/* Sidebar widget containers */
section[data-testid="stSidebar"] .stFileUploader,
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="textarea"]{{
  background: rgba(255,255,255,.10) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  padding: 8px 10px !important;
}}

/* Inputs readable */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [role="combobox"] input,
section[data-testid="stSidebar"] [role="spinbutton"] input{{
  color: #0B1F3A !important;
  -webkit-text-fill-color:#0B1F3A !important;
  background: rgba(255,255,255,.95) !important;
  border-radius: 12px !important;
}}

/* File uploader dropzone + button */
section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"]{{
  background: rgba(255,255,255,.95) !important;
  border: 1px dashed rgba(7,27,51,.35) !important;
  border-radius: 14px !important;
}}
section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"] *{{
  color:#0B1F3A !important;
}}
section[data-testid="stSidebar"] .stFileUploader button{{
  background: var(--accent) !important;
  color: white !important;
  border-radius: 12px !important;
  font-weight: 900 !important;
  border: 0 !important;
  width: 100% !important;
  padding: 10px 14px !important;
}}

/* Multiselect tags readable */
section[data-testid="stSidebar"] span[data-baseweb="tag"]{{
  background: rgba(239,68,68,.20) !important;
  border: 1px solid rgba(239,68,68,.55) !important;
}}
section[data-testid="stSidebar"] span[data-baseweb="tag"] span{{ color: #FFECEC !important; font-weight: 900 !important; }}
section[data-testid="stSidebar"] span[data-baseweb="tag"] svg{{ fill:#FFECEC !important; }}

/* Tabs / Buttons */
button[data-baseweb="tab"]{{ font-weight: 900 !important; }}
.stButton button, .stDownloadButton button{{ border-radius: 12px !important; font-weight: 900 !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"

def fmt_float(x) -> str:
    try:
        return f"{float(x):,.1f}"
    except Exception:
        return "0.0"

def normalize_date_range(date_range):
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
    else:
        start = pd.to_datetime(date_range)
        end = pd.to_datetime(date_range)
    if start > end:
        start, end = end, start
    return start, end

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes=None, path=None):
    if file_bytes is not None:
        df = pd.read_excel(file_bytes)
        src = "Uploaded file"
    else:
        df = pd.read_excel(path)
        src = f"Local: {path}"

    required = ["Received Date", "Cargo Category", "No. of Trucks"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Received Date"] = pd.to_datetime(df["Received Date"], errors="coerce")
    df["No. of Trucks"] = pd.to_numeric(df["No. of Trucks"], errors="coerce").fillna(0)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) if "Quantity" in df.columns else 0
    df["Cargo Category"] = df["Cargo Category"].fillna("Unknown").astype(str)
    df["Crossing"] = df["Crossing"].fillna("Unknown").astype(str) if "Crossing" in df.columns else "Unknown"

    df = df.dropna(subset=["Received Date"])
    return df, src

@st.cache_data(show_spinner=False)
def build_agg(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    d = df.copy().set_index("Received Date")
    out = (
        d.groupby([pd.Grouper(freq=freq), "Cargo Category", "Crossing"], dropna=False)
         .agg(trucks=("No. of Trucks", "sum"), quantity=("Quantity", "sum"))
         .reset_index()
         .rename(columns={"Received Date": "ds"})
         .sort_values("ds")
    )
    return out

def make_series(agg: pd.DataFrame, cats: list[str], metric_col: str,
                start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.Series:
    d = agg[(agg["ds"] >= start) & (agg["ds"] <= end)].copy()
    d = d[d["Cargo Category"].isin(cats)]
    s = d.groupby("ds")[metric_col].sum().sort_index()
    s = s.asfreq(freq).fillna(0)
    return s

def hw_forecast(series: pd.Series, periods: int, freq: str) -> pd.Series:
    series = series.asfreq(freq).fillna(0)

    if len(series) < 8 or series.sum() <= 0:
        idx = pd.date_range(series.index.max() + pd.tseries.frequencies.to_offset(freq),
                            periods=periods, freq=freq)
        return pd.Series([0.0] * periods, index=idx)

    if freq == "D":
        sp, ok = 7, len(series) >= 14
    elif freq == "W":
        sp, ok = 4, len(series) >= 12
    else:
        sp, ok = 12, len(series) >= 24

    try:
        model = ExponentialSmoothing(
            series, trend="add",
            seasonal="add" if ok else None,
            seasonal_periods=sp if ok else None
        ).fit(optimized=True)
        return model.forecast(periods).clip(lower=0)
    except Exception:
        w = max(3, min(14, len(series)//3))
        level = float(series.rolling(w).mean().iloc[-1])
        idx = pd.date_range(series.index.max() + pd.tseries.frequencies.to_offset(freq),
                            periods=periods, freq=freq)
        return pd.Series([max(level, 0.0)] * periods, index=idx)

# =========================
# HEADER
# =========================
st.markdown('<h1 class="pro-title">📊 Gaza Aid Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="pro-sub">Real-Time Monitoring & Predictive Analytics</h3>', unsafe_allow_html=True)
st.markdown('<h4 class="pro-sub2">Humanitarian Flow • Supply Gaps • Forecasting</h4>', unsafe_allow_html=True)
st.markdown('<hr class="pro-hr">', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 📁 Data Source")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("## ⚙️ Controls")
metric = st.sidebar.selectbox("Metric", ["Trucks", "Quantity"], index=0)
metric_col = "trucks" if metric == "Trucks" else "quantity"

freq_label = st.sidebar.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=0)
freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label]

default_required = 200.0 if (metric == "Trucks" and freq_label == "Daily") else 0.0
required_per_period = st.sidebar.number_input(
    f"Required {metric} / {freq_label}",
    min_value=0.0,
    value=float(default_required),
    step=10.0,
)

forecast_periods = st.sidebar.slider(f"Forecast Horizon ({freq_label.lower()} periods)", 7, 60, 21)
show_zeros = st.sidebar.toggle("Include Zero Periods", value=True)

# =========================
# LOAD DATA
# =========================
with st.spinner("Loading dataset..."):
    try:
        if uploaded is not None:
            df, src_badge = load_and_clean(file_bytes=uploaded.getvalue())
        else:
            df, src_badge = load_and_clean(path=DEFAULT_FILE_PATH)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

agg = build_agg(df, freq=freq)

# =========================
# FILTERS
# =========================
all_categories = sorted(agg["Cargo Category"].unique().tolist())
all_crossings = sorted(agg["Crossing"].unique().tolist())

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Filters")

select_all_cats = st.sidebar.checkbox("Select all categories", value=True)
selected_categories = st.sidebar.multiselect(
    "Cargo Categories",
    options=all_categories,
    default=all_categories if select_all_cats else (all_categories[:1] if all_categories else [])
)
if not selected_categories:
    selected_categories = all_categories[:]

select_all_cross = st.sidebar.checkbox("Select all crossings", value=True)
selected_crossings = st.sidebar.multiselect(
    "Crossings",
    options=all_crossings,
    default=all_crossings if select_all_cross else all_crossings
)

min_date = agg["ds"].min().date()
max_date = agg["ds"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date))
start, end = normalize_date_range(date_range)

agg_f = agg[(agg["ds"] >= start) & (agg["ds"] <= end)].copy()
if selected_crossings:
    agg_f = agg_f[agg_f["Crossing"].isin(selected_crossings)].copy()

series = make_series(agg_f, selected_categories, metric_col, start, end, freq=freq)
series_plot = series if show_zeros else series[series > 0]
enough_data = (len(series) >= 8) and (series.sum() > 0)

# =========================
# TOP BADGES
# =========================
b1, b2 = st.columns([1, 1])
with b1:
    st.markdown(f'<span class="badge">Source: {src_badge}</span>', unsafe_allow_html=True)
with b2:
    st.markdown(
        f'<div style="text-align:right;">'
        f'<span class="badge">Range: {start.date()} → {end.date()} • {freq_label} • {metric}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# =========================
# KPI
# =========================
total_val = float(series.sum())
avg_val = float(series.mean()) if len(series) else 0.0
periods_count = int(len(series))
gap_total = float((required_per_period - series).clip(lower=0).sum())
coverage_rate = float((series >= required_per_period).mean() * 100) if periods_count else 0.0

if len(series) >= 5:
    last_val = float(series.iloc[-1])
    prev4 = float(series.iloc[-5:-1].mean())
    momentum = last_val - prev4
else:
    momentum = 0.0

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi"><div class="t">Total {metric}</div><div class="v">{fmt_int(total_val)}</div><div class="n">Selected filters & range</div></div>
  <div class="kpi"><div class="t">Avg / {freq_label}</div><div class="v">{fmt_float(avg_val)}</div><div class="n">Mean per period</div></div>
  <div class="kpi"><div class="t">Periods</div><div class="v">{fmt_int(periods_count)}</div><div class="n">{freq_label} in range</div></div>
  <div class="kpi"><div class="t">Total Supply Gap</div><div class="v">{fmt_float(gap_total)}</div><div class="n">Sum(max(required - actual, 0))</div></div>
  <div class="kpi"><div class="t">Coverage ≥ Required</div><div class="v">{fmt_float(coverage_rate)}%</div><div class="n">Share meeting target</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Overview",
    "📈 Trends",
    "🧩 Composition",
    "🔮 Forecast",
    "🚨 Alerts",
    "✅ Data Quality",
    "🧠 Insights",
    "🗺️ Gaza Map Heatmap"
])

# -------- TAB 1
with tab1:
    st.markdown('<h3 class="h-sec">Executive Overview</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Top categories and crossings based on your current filters.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cat_sum = (agg_f.groupby("Cargo Category", as_index=False)[metric_col]
                   .sum().sort_values(metric_col, ascending=False))
        fig_cat = px.bar(cat_sum.head(12), x=metric_col, y="Cargo Category", orientation="h",
                         title=f"Top Categories by Total {metric}")
        fig_cat.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        cross_sum = (agg_f.groupby("Crossing", as_index=False)[metric_col]
                     .sum().sort_values(metric_col, ascending=False))
        fig_cross = px.bar(cross_sum, x=metric_col, y="Crossing", orientation="h",
                           title=f"Crossings by Total {metric}")
        fig_cross.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_cross, use_container_width=True)

# -------- TAB 2
with tab2:
    st.markdown('<h3 class="h-sec">Trends</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Time series trend + supply gap view.</p>', unsafe_allow_html=True)

    if series_plot.empty:
        st.warning("No data to plot for this selection.")
    else:
        df_plot = pd.DataFrame({"ds": series_plot.index, "value": series_plot.values})
        fig_line = px.line(df_plot, x="ds", y="value", title=f"{metric} Over Time ({freq_label})")
        fig_line.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

    gap_df = pd.DataFrame({"date": series.index, "actual": series.values, "required": required_per_period})
    gap_df["gap"] = (gap_df["required"] - gap_df["actual"]).clip(lower=0)
    st.markdown('<h3 class="h-sec">Supply Gap (Actual)</h3>', unsafe_allow_html=True)
    st.dataframe(gap_df[gap_df["gap"] > 0].head(80), use_container_width=True)

# -------- TAB 3
with tab3:
    st.markdown('<h3 class="h-sec">Composition</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Category contribution over time (stacked area).</p>', unsafe_allow_html=True)

    comp = agg_f.groupby(["ds", "Cargo Category"], as_index=False)[metric_col].sum()
    if comp.empty:
        st.info("No composition data for current filters.")
    else:
        fig_area = px.area(comp, x="ds", y=metric_col, color="Cargo Category",
                           title=f"Category Composition — {metric} ({freq_label})")
        fig_area.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_area, use_container_width=True)

# -------- TAB 4
with tab4:
    st.markdown('<h3 class="h-sec">Forecast</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Holt-Winters forecast with safe fallback for sparse data.</p>', unsafe_allow_html=True)

    if not enough_data:
        st.warning("Not enough data for forecasting. Expand range or adjust filters.")
    else:
        fcst = hw_forecast(series, forecast_periods, freq=freq)
        hist_df = pd.DataFrame({"ds": series.index, "actual": series.values})
        fcst_df = pd.DataFrame({"ds": fcst.index, "forecast": fcst.values})

        fig = px.line(hist_df, x="ds", y="actual", title=f"Actual + Forecast ({metric}, {freq_label})")
        fig.add_scatter(x=fcst_df["ds"], y=fcst_df["forecast"], mode="lines", name="Forecast")
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        out = fcst_df.copy()
        out["required"] = required_per_period
        out["gap"] = (out["required"] - out["forecast"]).clip(lower=0)

        st.markdown('<h3 class="h-sec">Forecast Table</h3>', unsafe_allow_html=True)
        st.dataframe(out.head(80), use_container_width=True)

        st.download_button(
            "⬇️ Download Forecast CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="forecast.csv",
            mime="text/csv",
        )

# -------- TAB 5
with tab5:
    st.markdown('<h3 class="h-sec">Alerts</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Alerts when required exceeds actual or forecast.</p>', unsafe_allow_html=True)

    actual_alerts = pd.DataFrame({"date": series.index, "actual": series.values, "required": required_per_period})
    actual_alerts["gap"] = (actual_alerts["required"] - actual_alerts["actual"]).clip(lower=0)
    actual_alerts = actual_alerts[actual_alerts["gap"] > 0].copy()

    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<h3 class="h-sec">Actual Alerts (History)</h3>', unsafe_allow_html=True)
        st.dataframe(actual_alerts.head(120), use_container_width=True)

    with a2:
        st.markdown('<h3 class="h-sec">Forecast Alerts (Next Periods)</h3>', unsafe_allow_html=True)
        if enough_data:
            fcst2 = hw_forecast(series, forecast_periods, freq=freq)
            f = pd.DataFrame({"date": fcst2.index, "forecast": fcst2.values})
            f["required"] = required_per_period
            f["gap"] = (f["required"] - f["forecast"]).clip(lower=0)
            f = f[f["gap"] > 0].copy()
            st.dataframe(f.head(120), use_container_width=True)
        else:
            st.info("Forecast alerts require enough data.")

# -------- TAB 6
with tab6:
    st.markdown('<h3 class="h-sec">Data Quality</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Missing values, date coverage, and preview of cleaned data.</p>', unsafe_allow_html=True)

    n = len(df)
    miss_date = int(df["Received Date"].isna().sum())
    miss_cat = int(df["Cargo Category"].isna().sum())
    miss_tr = int(df["No. of Trucks"].isna().sum())

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{n:,}")
    q2.metric("Missing Received Date", f"{miss_date:,}")
    q3.metric("Missing Category", f"{miss_cat:,}")
    q4.metric("Missing Trucks", f"{miss_tr:,}")

    st.markdown('<h3 class="h-sec">Preview (Cleaned)</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(400), use_container_width=True)

# -------- TAB 7
with tab7:
    st.markdown('<h3 class="h-sec">Auto Insights</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Quick narrative insights generated from your current selection.</p>', unsafe_allow_html=True)

    if len(series) == 0:
        st.info("No data for current filters.")
    else:
        avg_flow = float(series.mean())
        max_day = series.idxmax()
        max_val = float(series.max())
        min_day = series.idxmin()
        min_val = float(series.min())
        gap_ratio = float((series < required_per_period).mean() * 100) if required_per_period > 0 else 0.0
        trend = "Increasing" if (series.iloc[-1] > series.iloc[0]) else "Decreasing"

        st.markdown(
            f"""
<div class="kpi">
  <div class="t">Summary</div>
  <div class="n" style="font-size:14px; color:#0B1F3A;">
    • Average <b>{metric}</b> per {freq_label}: <b>{fmt_float(avg_flow)}</b><br>
    • Highest flow: <b>{fmt_int(max_val)}</b> on <b>{max_day.date()}</b><br>
    • Lowest flow: <b>{fmt_int(min_val)}</b> on <b>{min_day.date()}</b><br>
    • Gap periods (below required): <b>{fmt_float(gap_ratio)}%</b><br>
    • Trend: <b>{trend}</b><br>
    • Momentum: <b>{fmt_float(momentum)}</b>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# -------- TAB 8
with tab8:
    st.markdown('<h3 class="h-sec">Gaza Map Heatmap (Crossings)</h3>', unsafe_allow_html=True)
    st.markdown('<p class="p-muted">Heat intensity shows estimated aid inflow concentration by crossing.</p>', unsafe_allow_html=True)

    crossing_coords = {
        "Erez": (31.559, 34.565),
        "Western Erez": (31.555, 34.560),
        "Kerem Shalom": (31.219, 34.284),
        "Rafah Crossing": (31.262, 34.247),
        "Gate 96": (31.250, 34.320),
        "Kissufim": (31.367, 34.403),
        "JLOTS": (31.520, 34.430),
    }

    dmap = df.copy()
    dmap = dmap[(dmap["Received Date"] >= pd.to_datetime(start)) & (dmap["Received Date"] <= pd.to_datetime(end))]

    if selected_categories:
        dmap = dmap[dmap["Cargo Category"].isin(selected_categories)]
    if selected_crossings:
        dmap = dmap[dmap["Crossing"].isin(selected_crossings)]

    value_col = "No. of Trucks" if metric == "Trucks" else "Quantity"

    cross_sum = (
        dmap.groupby("Crossing", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
    )

    cross_sum["lat"] = cross_sum["Crossing"].map(lambda x: crossing_coords.get(x, (None, None))[0])
    cross_sum["lon"] = cross_sum["Crossing"].map(lambda x: crossing_coords.get(x, (None, None))[1])
    cross_sum = cross_sum.dropna(subset=["lat", "lon"])

    if cross_sum.empty:
        st.info("No crossings with known coordinates for current filters. Update crossing_coords.")
    else:
        fig = px.density_mapbox(
            cross_sum,
            lat="lat",
            lon="lon",
            z="value",
            radius=35,
            zoom=9.2,
            center=dict(lat=31.42, lon=34.35),
            mapbox_style="open-street-map",
            hover_name="Crossing",
            hover_data={"value": True, "lat": False, "lon": False},
            title=f"Heatmap by Crossings — {metric}"
        )

        fig_points = px.scatter_mapbox(
            cross_sum,
            lat="lat",
            lon="lon",
            size="value",
            size_max=28,
            zoom=9.2,
            center=dict(lat=31.42, lon=34.35),
            mapbox_style="open-street-map",
            hover_name="Crossing",
            hover_data={"value": True},
        )

        for tr in fig_points.data:
            fig.add_trace(tr)

        fig.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<h3 class="h-sec">Crossings Table</h3>', unsafe_allow_html=True)
        st.dataframe(cross_sum.sort_values("value", ascending=False), use_container_width=True)
