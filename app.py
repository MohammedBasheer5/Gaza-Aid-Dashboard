import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import base64
from pathlib import Path

# =========================
# CONFIG
# =========================
DEFAULT_FILE_PATH = "commodities-received-13.xlsx"
BG_IMAGE_PATH = "gaza_bg.jpg"

st.set_page_config(
    page_title="Gaza Aid Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME / CSS
# =========================
def _img_to_base64(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")

bg_b64 = _img_to_base64(BG_IMAGE_PATH)

bg_css = ""
if bg_b64:
    bg_css = f"""
[data-testid="stAppViewContainer"] {{
  background:
    linear-gradient(rgba(0,0,0,0.62), rgba(0,0,0,0.62)),
    url("data:image/jpg;base64,{bg_b64}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}
.main {{
  background:
    linear-gradient(rgba(6, 20, 43, 0.70), rgba(6, 20, 43, 0.70)),
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
  --bg:#0b1630;
  --card: rgba(255,255,255,0.92);
  --ink:#0B1F3A;
  --muted:#51627A;
  --navy:#06152b;
  --blue:#2AA3FF;
  --border: rgba(210, 225, 255, 0.70);
  --shadow: 0 14px 35px rgba(0,0,0,0.25);
  --shadow2: 0 10px 26px rgba(0,0,0,0.18);
}}

{bg_css}

.stApp {{
  background: transparent !important;
}}

[data-testid="stHeader"] {{
  background: rgba(0,0,0,0) !important;
}}

.block-container {{
  padding-top: 1.2rem;
}}

h1, h2, h3, h4, h5, h6 {{
  color: #FFFFFF !important;
  text-shadow: 0 2px 0 rgba(0,0,0,.55), 0 10px 24px rgba(0,0,0,.55);
}}

/* ====== Sidebar ====== */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #06152b 0%, #061b38 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}}
section[data-testid="stSidebar"] * {{
  color: #EAF2FF !important;
}}
section[data-testid="stSidebar"] a {{
  color: #BFE3FF !important;
}}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [role="combobox"],
section[data-testid="stSidebar"] [role="spinbutton"] {{
  color: #0B1F3A !important;
  -webkit-text-fill-color: #0B1F3A !important;
}}

section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {{
  color: #6B7A90 !important;
  -webkit-text-fill-color: #6B7A90 !important;
}}

section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="select"] {{
  background: #FFFFFF !important;
  border-radius: 12px;
  padding: 6px 8px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}}

/* File uploader — white card, dark readable text */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
  background: #FFFFFF !important;
  border-radius: 12px !important;
  padding: 10px !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.15) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label span {{
  color: #0B1F3A !important;
  -webkit-text-fill-color: #0B1F3A !important;
  font-weight: 700 !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
  background: #EEF4FF !important;
  border: 1.5px dashed #3B8ED6 !important;
  border-radius: 10px !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
  color: #0B1F3A !important;
  -webkit-text-fill-color: #0B1F3A !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {{
  color: #4A6080 !important;
  -webkit-text-fill-color: #4A6080 !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
  background: #FFFFFF !important;
  color: #0B1F3A !important;
  -webkit-text-fill-color: #0B1F3A !important;
  border: 1px solid #3B8ED6 !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}}

section[data-testid="stSidebar"] [data-baseweb="tag"] {{
  background: #ff3b3b !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}}
section[data-testid="stSidebar"] [data-baseweb="tag"] span {{
  color: #ffffff !important;
  font-weight: 900 !important;
}}
section[data-testid="stSidebar"] [data-baseweb="tag"] svg {{
  color: #ffffff !important;
  fill: #ffffff !important;
  opacity: 0.95 !important;
}}

/* ====== Header ====== */
.pro-title {{
  text-align: center;
  font-weight: 950;
  font-size: 3.6rem !important;
  letter-spacing: -1px;
  margin: 0 0 10px 0;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  text-shadow:
    0 0 60px rgba(255,200,80,0.55),
    0 0 30px rgba(255,255,255,0.35),
    0 4px 28px rgba(0,0,0,0.95),
    0 2px 6px rgba(0,0,0,1.0);
}}
.pro-sub {{
  text-align: center;
  font-size: 1.55rem !important;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #B8D8F0 !important;
  -webkit-text-fill-color: #B8D8F0 !important;
  letter-spacing: 0.2px;
  text-shadow:
    0 2px 20px rgba(0,0,0,0.90),
    0 1px 4px rgba(0,0,0,1.0);
}}
.pro-sub2 {{
  text-align: center;
  font-size: 1.00rem !important;
  font-weight: 500;
  margin: 0 0 14px 0;
  color: rgba(255,210,120,0.95) !important;
  -webkit-text-fill-color: rgba(255,210,120,0.95) !important;
  letter-spacing: 3.5px;
  text-transform: uppercase;
  text-shadow:
    0 2px 18px rgba(0,0,0,0.90),
    0 1px 4px rgba(0,0,0,1.0);
}}
.pro-hr {{
  border: 0;
  height: 1.5px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(255,210,120,0.60) 20%,
    rgba(255,255,255,0.50) 50%,
    rgba(255,210,120,0.60) 80%,
    transparent 100%);
  margin: 10px 0 18px;
  border-radius: 2px;
}}
