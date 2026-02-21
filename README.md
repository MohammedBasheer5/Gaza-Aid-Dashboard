# 📊 Gaza Aid Intelligence

> **Real-Time Monitoring & Predictive Analytics for Humanitarian Aid**
> 
> `Humanitarian Flow` • `Supply Gaps` • `Forecasting`

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

---

## 🌍 Overview

**Gaza Aid Intelligence** is a professional-grade humanitarian data analytics dashboard built with Python and Streamlit. It transforms raw commodity delivery data into actionable intelligence — enabling aid organizations to monitor delivery trends, identify supply gaps, and forecast future shortfalls before they become crises.

The platform provides a unified interface for humanitarian coordinators, analysts, and policymakers to answer critical questions:

- **How much aid actually arrived?** — and through which crossings?
- **Where are the gaps?** — between what was delivered and what was needed?
- **What comes next?** — statistical forecasts for upcoming periods
- **Which categories are at risk?** — food, medical, non-food items?

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 **Executive Overview** | Top cargo categories and crossing activity — updates with all filters |
| 📈 **Trend Analysis** | Time-series visualization of aid flow with supply gap table |
| 🧩 **Composition View** | Stacked area chart showing category contribution over time |
| 🔮 **Forecasting** | Holt-Winters model predicting future aid flow with gap analysis |
| 🚨 **Smart Alerts** | Historical and forecast shortage alerts — both exportable as CSV |
| ✅ **Data Quality** | Audit tool showing missing values, date coverage, and data preview |
| 🧠 **Auto Insights** | Narrative summary with trend, momentum, and risk analysis |
| 🗺️ **Map Heatmap** | Interactive OpenStreetMap density heatmap by crossing point |

---

## 🖥️ Screenshots

> Dashboard running with the Gaza commodities dataset

```
📊 GAZA AID INTELLIGENCE
Real-Time Monitoring & Predictive Analytics
HUMANITARIAN FLOW  •  SUPPLY GAPS  •  FORECASTING
```

*Dark-themed UI with glass morphism design, blue glow header, and gold accent dividers*

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gaza-aid-intelligence.git
cd gaza-aid-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add required files

Place these files in the project root:
- `commodities-received-13.xlsx` — Your OCHA-format dataset
- `Gaza_BG.jpg` — Background image for the UI

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
statsmodels>=0.14.0
openpyxl>=3.1.0
```

Or install directly:

```bash
pip install streamlit pandas numpy plotly statsmodels openpyxl
```

---

## 📂 Project Structure

```
gaza-aid-intelligence/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Gaza_BG.jpg                     # UI background image
├── commodities-received-13.xlsx    # Default dataset (OCHA format)
└── README.md                       # This file
```

---

## 📋 Data Schema

Your Excel file must contain these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Received Date` | Date | ✅ Yes | Delivery date |
| `Cargo Category` | String | ✅ Yes | Type of goods |
| `No. of Trucks` | Numeric | ✅ Yes | Vehicle count |
| `Quantity` | Numeric | Optional | Cargo volume |
| `Crossing` | String | Optional | Border crossing name |
| `Status` | String | Optional | Delivery status |
| `Destination Recipient/ Partner` | String | Optional | Aid recipient |

---

## ⚙️ Sidebar Controls

| Control | Function |
|---------|----------|
| **Upload Excel** | Load your own `.xlsx` dataset |
| **Metric** | Switch between Trucks or Quantity |
| **Aggregation** | Daily / Weekly / Monthly grouping |
| **Required / Period** | Minimum threshold for gap calculations |
| **Forecast Horizon** | 7–60 future periods to predict |
| **Include Zero Periods** | Show/hide zero-delivery days |
| **Cargo Categories** | Filter by cargo type |
| **Crossings** | Filter by border crossing |
| **Date Range** | Restrict analysis to a time window |

---

## 🔮 Forecasting Model

Uses **Holt-Winters Triple Exponential Smoothing** via `statsmodels`:

```
Seasonal Periods:
  Daily   → 7  (weekly cycle)   — requires 14+ data points
  Weekly  → 4  (monthly cycle)  — requires 12+ data points
  Monthly → 12 (annual cycle)   — requires 24+ data points
```

**Fallback:** When data is insufficient, a rolling-window average is projected forward — ensuring the dashboard always returns a usable result.

---

## 🗺️ Mapped Crossings

| Crossing | Latitude | Longitude | Location |
|----------|----------|-----------|----------|
| Erez | 31.559 | 34.565 | Northern Gaza |
| Western Erez | 31.555 | 34.560 | Northern Gaza |
| Kerem Shalom | 31.219 | 34.284 | Southern Gaza (Main) |
| Rafah Crossing | 31.262 | 34.247 | Southern Gaza / Egypt |
| Gate 96 | 31.250 | 34.320 | Southern Gaza |
| Kissufim | 31.367 | 34.403 | Central Gaza |
| JLOTS | 31.520 | 34.430 | Coastal / Sea Route |

---

## 🎨 Design

- **Theme:** Full dark UI with Gaza background imagery
- **Header:** Three-tier hierarchy — white glow title, silver-blue subtitle, gold tagline
- **Tabs:** Glass morphism pill navigation with blue gradient active state
- **Cards:** KPI cards with backdrop blur and drop shadows
- **Sidebar:** Deep navy gradient with white input fields

---

## 🚢 Deployment

### Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `app.py` as the main file
5. Deploy ✅

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t gaza-aid-intelligence .
docker run -p 8501:8501 gaza-aid-intelligence
```

---

## 🔭 Roadmap

- [ ] Live API integration with OCHA HDX
- [ ] Confidence interval bands on forecast charts
- [ ] Anomaly detection for unusual spikes/drops
- [ ] PDF/Excel full report export
- [ ] Arabic language support (RTL)
- [ ] Multi-file dataset merging
- [ ] Per-category individual forecasts
- [ ] Animated time-lapse heatmap

---

## 📄 Data Source

This dashboard is designed to work with datasets from:

**[UN OCHA](https://www.unocha.org/)** — Office for the Coordination of Humanitarian Affairs  
**[Humanitarian Data Exchange (HDX)](https://data.humdata.org/)**

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## ⚠️ Disclaimer

This tool is built for humanitarian monitoring and analysis purposes. All data displayed reflects officially reported figures. The forecasting model provides statistical estimates and should not be used as the sole basis for operational decisions.

---

## 📬 Contact

For questions, suggestions, or collaboration inquiries — open an issue on GitHub.

---

<div align="center">

**Gaza Aid Intelligence**

*Turning data into dignity.*

</div>
