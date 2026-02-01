"""
CLV ANALYTICS: Interactive Guide
================================
Standard Clean Theme for Maximum Readability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from pathlib import Path

# ============================================================================
# 1. CLEAN CONFIG
# ============================================================================
st.set_page_config(
    page_title="CLV Analytics Guide",
    page_icon="📘",
    layout="wide"
)

# No Custom CSS - Relying on Streamlit's native clean design for readability

# ============================================================================
# 2. DATA ENGINE
# ============================================================================
BASE_DIR = Path(__file__).parent
RAW_PATH = BASE_DIR.parent / 'WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv'
REPORT_PATH = BASE_DIR / 'report' / 'CLV_Complete_Guide.pdf'

@st.cache_data
def load_data():
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    return None

def get_pdf_download_btn(path):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label="📥 Download Complete Textbook PDF",
        data=pdf_bytes,
        file_name="CLV_Complete_Guide.pdf",
        mime="application/pdf",
        type="primary"
    )

# ============================================================================
# 3. CHAPTERS
# ============================================================================

def chapter_home():
    st.title("📘 CLV Prediction: The Complete Guide")
    st.markdown("""
    Welcome to the interactive companion to the **CLV Analysis Report**.
    This app allows you to explore the data, run the code, and see the results live.
    """)
    
    if REPORT_PATH.exists():
        st.success("✅ **Report Updated:** The latest PDF is ready.")
        get_pdf_download_btn(REPORT_PATH)
    else:
        st.warning("⚠️ Report PDF not found. Please generate it.")

    st.header("Executive Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Customers", "9,134")
    with c2: st.metric("Avg CLV", "$8,004")
    with c3: st.metric("Top Strategy", "Focus on Employment Status")

def chapter_1_data(df):
    st.header("Chapter 1: The Data")
    st.markdown("We start with the raw dataset. This is the foundation of our analysis.")
    
    st.subheader("Data Dictionary")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Key Statistics")
    st.write(df.describe())

def chapter_2_eda(df):
    st.header("Chapter 2: Forensic Audit")
    st.markdown("We look for the 'Bleeding Necks' - segments with high risk.")
    
    tab1, tab2 = st.tabs(["Target Analysis", "Risk Analysis"])
    
    with tab1:
        st.subheader("Customer Lifetime Value Distribution")
        fig = px.histogram(df, x='customer_lifetime_value', nbins=50, 
                           title="Distribution of CLV",
                           color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig, use_container_width=True)
        st.info("Note the long tail (Positive Skew). High value customers are rare.")

    with tab2:
        st.subheader("Claims by Employment Status")
        fig = px.box(df, x='employmentstatus', y='total_claim_amount', 
                     color='employmentstatus',
                     title="Impact of Unemployment on Claims")
        st.plotly_chart(fig, use_container_width=True)
        st.error("Unemployed customers show higher median claims and more variability.")

def chapter_7_predictor(df):
    st.header("Chapter 7: Live Predictor")
    st.markdown("Estimate CLV for a new customer (Demo Model).")
    
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual Income ($)", 0, 200000, 50000)
        premium = st.number_input("Monthly Premium ($)", 50, 500, 100)
    with c2:
        policies = st.slider("Number of Policies", 1, 9, 2)
        vehicle = st.selectbox("Vehicle Class", ['Four-Door Car', 'SUV', 'Luxury', 'Sports'])

    # Demo prediction
    pred = (premium * 12 * 3) + (income * 0.05)
    if 'Luxury' in vehicle: pred *= 1.5
    
    st.success(f"### Predicted CLV: ${pred:,.2f}")

# ============================================================================
# 4. MAIN ROUTER
# ============================================================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", 
        ['Home', 'Ch 1: The Data', 'Ch 2: Forensic Audit', 'Ch 7: Predictor'])
    
    df = load_data()
    if df is not None:
        if page == 'Home': chapter_home()
        elif page == 'Ch 1: The Data': chapter_1_data(df)
        elif page == 'Ch 2: Forensic Audit': chapter_2_eda(df)
        elif page == 'Ch 7: Predictor': chapter_7_predictor(df)
    else:
        st.error("Data file not found.")

if __name__ == "__main__":
    main()
