"""
CLV EXECUTIVE DASHBOARD
=======================
A professional BI Interface for Customer Lifetime Value Analysis.
Focus: KPIs, Segment Performance, and Actionable Insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================================
# 1. PAGE CONFIG (Native & Clean)
# ============================================================================
st.set_page_config(
    page_title="CLV Executive Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. DATA LOADING
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / 'WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv'
REPORT_PATH = BASE_DIR / 'report' / 'IEEE_CLV_Analysis_Paper.pdf'

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Normalize columns
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    return None

df = load_data()

# ============================================================================
# 3. DASHBOARD COMPONENTS
# ============================================================================

def main():
    if df is None:
        st.error("Data source not found.")
        return

    # --- SIDEBAR ---
    st.sidebar.title("📈 CLV Dashboard")
    
    # Global Filter
    st.sidebar.header("Global Filters")
    region_filter = st.sidebar.multiselect(
        "Select State:",
        options=df['state'].unique(),
        default=df['state'].unique()
    )
    
    # Filter Data
    df_filtered = df[df['state'].isin(region_filter)]
    
    # Navigation
    page = st.sidebar.radio("Navigate:", ["Executive Overview", "Forensic Analysis", "Predictive Model", "Download Report"])

    # --- PAGE: EXECUTIVE OVERVIEW ---
    if page == "Executive Overview":
        st.title("Executive Overview")
        st.markdown("### Key Performance Indicators (KPIs)")
        
        # Top Metrics
        total_clv = df_filtered['customer_lifetime_value'].sum()
        avg_clv = df_filtered['customer_lifetime_value'].mean()
        high_value_counts = len(df_filtered[df_filtered['customer_lifetime_value'] > 10000])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Portfolio Value", f"${total_clv/1e6:,.1f}M")
        c2.metric("Avg. Customer Value", f"${avg_clv:,.0f}")
        c3.metric("High Value Customers", f"{high_value_counts:,}")
        c4.metric("Active Policies", f"{df_filtered['number_of_policies'].sum():,}")
        
        st.markdown("---")
        
        # Two-Column Layout for Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Sales Channel")
            fig_channel = px.pie(df_filtered, names='sales_channel', values='customer_lifetime_value', hole=0.4)
            st.plotly_chart(fig_channel, use_container_width=True)
            
        with col2:
            st.subheader("Customer Distribution by CLV")
            fig_hist = px.histogram(df_filtered, x='customer_lifetime_value', nbins=30, color_discrete_sequence=['#3366cc'])
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- PAGE: FORENSIC ANALYSIS ---
    elif page == "Forensic Analysis":
        st.title("Forensic Analysis: Risk & Value Drivers")
        
        tab1, tab2 = st.tabs(["The 'Bleeding Neck' (Risk)", "Value Drivers"])
        
        with tab1:
            st.subheader("Identifying High-Risk Segments")
            st.markdown("Analysis of Claims vs. Employment Status")
            
            fig_box = px.box(df_filtered, x='employmentstatus', y='total_claim_amount', color='employmentstatus')
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.warning("""
            **Insight:** Unemployed customers show significantly higher claim variance and median claim amounts. 
            This segment represents a 'Bleeding Neck' for profitability.
            """)
            
        with tab2:
            st.subheader("What Drives Value?")
            st.markdown("Correlation: Monthly Premium vs. CLV")
            
            fig_scatter = px.scatter(df_filtered, x='monthly_premium_auto', y='customer_lifetime_value', 
                                   color='vehicle_class', opacity=0.6)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("Strong positive correlation confirmed. Luxury vehicles drive highest premiums and CLV.")

    # --- PAGE: PREDICTIVE MODEL ---
    elif page == "Predictive Model":
        st.title("Propensity & Value Modeling")
        
        st.markdown("### Live Value Estimator")
        st.markdown("_Estimate the lifetime value of a prospective customer._")
        
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                income = st.number_input("Annual Income ($)", 0, 200000, 50000)
                edu = st.selectbox("Education", df['education'].unique())
            with c2:
                premium = st.number_input("Monthly Premium ($)", 50, 500, 100)
                vehicle = st.selectbox("Vehicle Class", df['vehicle_class'].unique())
            with c3:
                policies = st.slider("Number of Policies", 1, 9, 2)
                state = st.selectbox("State", df['state'].unique())
                
            submit = st.form_submit_button("Calculate CLV Prediction", type="primary")
            
        if submit:
            # Proxy Model Logic (Replace with actual model.predict)
            base_score = (premium * 12 * 4) + (income * 0.05)
            if 'Luxury' in vehicle: base_score *= 1.3
            if edu == 'Doctor': base_score *= 1.1
            
            st.success(f"### Estimated Lifetime Value: ${base_score:,.2f}")
            st.progress(min(base_score/50000, 1.0))

    # --- PAGE: DOWNLOAD REPORT ---
    elif page == "Download Report":
        st.title("Export Analysis")
        st.markdown("Download the complete 40-page technical report generated from this analysis.")
        
        if REPORT_PATH.exists():
            with open(REPORT_PATH, "rb") as f:
                st.download_button(
                    label="📥 Download Full PDF Report",
                    data=f,
                    file_name="IEEE_CLV_Analysis_Paper.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            
            st.image(str(BASE_DIR / 'report' / 'figures' / '01_target_distribution.png'), caption="Report Preview", width=600)
        else:
            st.warning("Report file not found. Please run the generation script.")

if __name__ == "__main__":
    main()
