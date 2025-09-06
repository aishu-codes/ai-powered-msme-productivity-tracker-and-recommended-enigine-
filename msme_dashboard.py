# msme_streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="AI-Powered MSME Productivity & Forecast Dashboard", layout="wide")

st.title("AI-Powered MSME Productivity & Forecast Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your startup funding CSV", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Overview")
        st.dataframe(df.head())
        st.write(df.describe())

        # --- Column selection ---
        # Identify date column
        date_candidates = [col for col in df.columns if 'date' in col.lower()]
        if len(date_candidates) == 0:
            st.error("No date column found! Please check your CSV.")
        else:
            date_col = date_candidates[0]
            st.success(f"Detected date column: {date_col}")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Identify numeric column
        numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_candidates) == 0:
            # Try converting columns that look like Amount
            for col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_candidates) == 0:
            st.error("No numeric columns found after conversion. Cannot continue.")
        else:
            numeric_col = numeric_candidates[0]
            st.success(f"Detected numeric column: {numeric_col}")

            # --- Moving Averages ---
            df = df.dropna(subset=[date_col, numeric_col]).sort_values(date_col)
            df['MA_7'] = df[numeric_col].rolling(window=7).mean()
            df['MA_30'] = df[numeric_col].rolling(window=30).mean()

            st.subheader("Funding Amount with Moving Averages")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df[date_col], df[numeric_col], label='Original')
            ax.plot(df[date_col], df['MA_7'], label='7-day MA')
            ax.plot(df[date_col], df['MA_30'], label='30-day MA')
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount")
            ax.legend()
            st.pyplot(fig)

            # --- Daily Returns ---
            daily_returns = df[numeric_col].pct_change()*100
            st.subheader("Histogram of Daily Returns")
            fig2, ax2 = plt.subplots(figsize=(12,6))
            ax2.hist(daily_returns.dropna(), bins=50, color='skyblue', edgecolor='black')
            ax2.set_xlabel("Daily Return (%)")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

            # --- Rolling Statistics ---
            st.subheader("Rolling Statistics")
            rolling_mean = df[numeric_col].rolling(window=7).mean()
            rolling_std = df[numeric_col].rolling(window=7).std()
            fig3, ax3 = plt.subplots(figsize=(12,6))
            ax3.plot(df[date_col], df[numeric_col], label='Original')
            ax3.plot(df[date_col], rolling_mean, label='Rolling Mean')
            ax3.plot(df[date_col], rolling_std, label='Rolling Std')
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Amount")
            ax3.legend()
            st.pyplot(fig3)

            # --- Seasonal Decomposition ---
            st.subheader("Seasonal Decomposition")
            try:
                decomposition = seasonal_decompose(df[numeric_col], model='additive', period=30)
                fig4, ax4 = plt.subplots(4,1, figsize=(12,10), sharex=True)
                ax4[0].plot(df[date_col], decomposition.observed)
                ax4[0].set_title("Observed")
                ax4[1].plot(df[date_col], decomposition.trend)
                ax4[1].set_title("Trend")
                ax4[2].plot(df[date_col], decomposition.seasonal)
                ax4[2].set_title("Seasonal")
                ax4[3].plot(df[date_col], decomposition.resid)
                ax4[3].set_title("Residual")
                st.pyplot(fig4)
            except Exception as e:
                st.warning(f"Seasonal decomposition failed: {e}")

            # --- ADF Test ---
            st.subheader("ADF Test (Stationarity Check)")
            adf_result = adfuller(df[numeric_col].dropna())
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"p-value: {adf_result[1]:.4f}")
            st.write("Stationary" if adf_result[1] < 0.05 else "Non-stationary")

            # --- Volatility Zones ---
            st.subheader("Volatility Zones (±1,2 Std Dev)")
            mean = df[numeric_col].mean()
            std = df[numeric_col].std()
            fig5, ax5 = plt.subplots(figsize=(12,6))
            ax5.plot(df[date_col], df[numeric_col], label='Original')
            ax5.fill_between(df[date_col], mean-std, mean+std, color='yellow', alpha=0.3, label='±1 Std')
            ax5.fill_between(df[date_col], mean-2*std, mean+2*std, color='orange', alpha=0.2, label='±2 Std')
            ax5.set_xlabel("Date")
            ax5.set_ylabel("Amount")
            ax5.legend()
            st.pyplot(fig5)

    except Exception as e:
        st.error(f"Error loading CSV: {e}")

else:
    st.info("Please upload a CSV file to continue.")









