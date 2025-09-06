import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="MSME Productivity Dashboard", layout="wide")

st.title("📊 AI-Powered MSME Productivity Tracker")
st.write("Analyze, forecast, and visualize sector data with actionable AI insights.")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("🔍 Filters")

metric = st.sidebar.selectbox(
    "Select Metric",
    ["Manufacturing Exports", "Production Total", "Mining Services"]
)

viz_type = st.sidebar.selectbox(
    "Select Visualization",
    ["Line Plot", "Histogram", "Rolling Statistics", "Seasonal Decomposition",
     "Volatility Zones", "Moving Averages"]
)

# -------------------------------
# LOAD DATA (Replace with your dataset)
# -------------------------------
# Example dummy data
dates = pd.date_range(start="2015-01-01", periods=120, freq="M")
data = {
    "Date": dates,
    "Manufacturing Exports": np.random.randint(200, 500, len(dates)),
    "Production Total": np.random.randint(300, 700, len(dates)),
    "Mining Services": np.random.randint(100, 400, len(dates)),
}
df = pd.DataFrame(data)
df = df.set_index("Date")

# -------------------------------
# 1. DATA OVERVIEW
# -------------------------------
st.subheader("📑 Data Overview")
st.write(df.head())
st.write(df.describe())

# -------------------------------
# 2. TIME SERIES FORECASTS (PLACEHOLDER)
# -------------------------------
st.subheader("📈 Time Series Forecasts")
tab1, tab2, tab3 = st.tabs(["Manufacturing Exports", "Production Total", "Mining Services"])

with tab1:
    st.line_chart(df["Manufacturing Exports"])
    st.write("🔮 Forecasting model output will appear here.")

with tab2:
    st.line_chart(df["Production Total"])
    st.write("🔮 Forecasting model output will appear here.")

with tab3:
    st.line_chart(df["Mining Services"])
    st.write("🔮 Forecasting model output will appear here.")

# -------------------------------
# 3. COMPARATIVE GROWTH DASHBOARD
# -------------------------------
st.subheader("📊 Comparative Growth Dashboard")
st.line_chart(df)

growth_rates = df.pct_change().mean() * 100
st.write("### 📈 Average Growth Rates (%)")
st.write(growth_rates)

# -------------------------------
# 4. AI RECOMMENDATIONS (PLACEHOLDER)
# -------------------------------
st.subheader("🤖 AI Recommendations")
recommendations = pd.DataFrame({
    "Metric": ["Manufacturing Exports", "Production Total", "Mining Services"],
    "Forecast": ["Stable Growth", "Moderate Growth", "High Volatility"],
    "Recommendation": [
        "Expand export incentives",
        "Focus on automation",
        "Diversify supply chain"
    ],
    "Risk/Opportunity": [
        "Low risk",
        "Medium opportunity",
        "High risk"
    ]
})
st.table(recommendations)

# -------------------------------
# 5. VISUALIZATIONS
# -------------------------------
st.subheader("📉 Visualizations")

if viz_type == "Line Plot":
    st.line_chart(df[metric])

elif viz_type == "Histogram":
    fig, ax = plt.subplots()
    df[metric].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f"Histogram of {metric}")
    st.pyplot(fig)

elif viz_type == "Rolling Statistics":
    fig, ax = plt.subplots()
    df[metric].plot(ax=ax, label="Original", alpha=0.7)
    df[metric].rolling(12).mean().plot(ax=ax, label="Rolling Mean")
    df[metric].rolling(12).std().plot(ax=ax, label="Rolling Std")
    ax.set_title(f"Rolling Statistics - {metric}")
    ax.legend()
    st.pyplot(fig)

elif viz_type == "Seasonal Decomposition":
    result = seasonal_decompose(df[metric], model="additive", period=12)
    fig = result.plot()
    st.pyplot(fig)

elif viz_type == "Volatility Zones":
    mean = df[metric].mean()
    std = df[metric].std()
    fig, ax = plt.subplots()
    ax.plot(df.index, df[metric], label="Data")
    ax.axhline(mean, color='black', linestyle='--', label="Mean")
    ax.axhline(mean + std, color='orange', linestyle='--', label="+1 Std")
    ax.axhline(mean - std, color='orange', linestyle='--')
    ax.axhline(mean + 2*std, color='red', linestyle='--', label="+2 Std")
    ax.axhline(mean - 2*std, color='red', linestyle='--')
    ax.set_title(f"Volatility Zones - {metric}")
    ax.legend()
    st.pyplot(fig)

elif viz_type == "Moving Averages":
    fig, ax = plt.subplots()
    df[metric].plot(ax=ax, label="Original")
    df[metric].rolling(7).mean().plot(ax=ax, label="7-day MA")
    df[metric].rolling(30).mean().plot(ax=ax, label="30-day MA")
    ax.set_title(f"Moving Averages - {metric}")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 6. RISKS & OPPORTUNITIES SUMMARY
# -------------------------------
st.subheader("⚠️ Risks & Opportunities Summary")
st.write("""
- 📉 **Risks:** High volatility in Mining Services, dependency on seasonal exports.  
- 📈 **Opportunities:** Stable growth in Production, steady upward trend in Exports.  
- 🧠 **Strategy:** Focus on automation in production, expand exports, and reduce mining risk.  
""")
