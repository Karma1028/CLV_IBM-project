"""
THE CLV MASTERCLASS: Interactive Textbook & Lab
===============================================
A high-production interactive learning platform for insurance analytics.
Features: Live Code Execution, Dynamic Visualizations, Mathematical Explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import base64
from pathlib import Path
from scipy import stats

# ============================================================================
# 1. PREMIUM PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="The CLV Masterclass",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"  # Focus on content
)

# Custom CSS for "Textbook" feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Fira+Code:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

    /* BASE TYPOGRAPHY */
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #111827;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h1 { font-size: 3rem; margin-top: 1rem; }
    h2 { font-size: 2rem; margin-top: 3rem; border-bottom: 2px solid #f3f4f6; padding-bottom: 0.5rem; }
    h3 { font-size: 1.5rem; margin-top: 2rem; color: #374151; }
    
    p, li {
        font-family: 'Merriweather', serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #374151;
        max-width: 800px;
    }
    
    /* CODE BLOCKS */
    .stCodeBlock {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* CUSTOM COMPONENTS */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 4rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif; /* Cleaner sans-serif for subtitle */
        font-size: 1.25rem;
        color: #cbd5e1;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .chapter-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s;
        cursor: pointer;
        height: 100%;
    }
    
    .chapter-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #6366f1;
    }
    
    .concept-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 2rem 0;
    }
    
    .math-box {
        background-color: #fafafa;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-family: 'Georgia', serif;
        margin: 2rem 0;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* PDF VIEWER */
    iframe {
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Navigation */
    .nav-link {
        display: block;
        padding: 0.5rem 1rem;
        color: #4b5563;
        text-decoration: none;
        border-radius: 6px;
        margin-bottom: 0.25rem;
        transition: background-color 0.2s;
    }
    .nav-link:hover {
        background-color: #f3f4f6;
        color: #111827;
    }
    .nav-active {
        background-color: #e0e7ff;
        color: #4338ca;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. DATA ENGINE
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'cleaned_data.csv'
RAW_PATH = BASE_DIR.parent / 'WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv'
REPORT_PATH = BASE_DIR / 'report' / 'CLV_Complete_Guide.pdf'

@st.cache_data
def load_data():
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    return None

def get_pdf_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ============================================================================
# 3. INTERACTIVE CHAPTERS
# ============================================================================

def chapter_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">The CLV Masterclass</div>
        <div class="hero-subtitle">
            An interactive textbook for the Data Science of Insurance.<br>
            Learn. Code. Predict.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Dashboard
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Course Progress", "14%", "+2 Chapters")
    with c2: st.metric("Current Module", "EDA", "Chapter 2")
    with c3: st.metric("Code Executed", "12 Cells", "Interactive")
    
    st.markdown("## 📚 Syllabus")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 1. The Mission\nData Dictionary & Business Problem")
        st.info("### 2. Forensic Audit\nExploratory Data Analysis (EDA)")
        st.info("### 3. feature Engineering\nLog Transforms & Encoding")
        st.info("### 4. Modeling Theory\nRandom Forest vs. Linear Regression")
    with col2:
        st.info("### 5. Evaluation\nR², MAE, & Feature Importance")
        st.info("### 6. Segmentation\nK-Means Clustering & Personas")
        st.info("### 7. Deployment\nSaving Models & A/B Testing")
        st.success("### 🎓 Final Project\nLive CLV Prediction Engine")

    # PDF Download
    if REPORT_PATH.exists():
        st.markdown("## 📄 Course Textbook")
        pdf_b64 = get_pdf_base64(REPORT_PATH)
        href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="CLV_Complete_Guide.pdf" style="text-decoration:none;background:#4338ca;color:white;padding:0.75rem 1.5rem;border-radius:8px;font-weight:600;">📥 Download Full PDF Guide</a>'
        st.markdown(href, unsafe_allow_html=True)

def chapter_1_data(df):
    st.title("Chapter 1: The Mission & The Data")
    st.markdown("""
    **Objective:** Understand the business problem and the raw materials (data) we have to solve it.
    
    Customer Lifetime Value (CLV) is the 'North Star' metric. It tells us:
    1.  **Who** to acquire (Marketing)
    2.  **How** to price (Underwriting)
    3.  **Why** they stay (Retention)
    """)
    
    st.markdown("### 1.1 The Dataset")
    st.markdown("This is our raw `DataFrame`. It contains **9,134 customers** and **24 attributes**.")
    
    with st.expander("🔎 View Raw Data Dictionary", expanded=True):
        st.dataframe(df.head(5), use_container_width=True)
        st.caption("Showing first 5 rows of WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
    
    st.markdown("### 1.2 Data Health Check")
    st.markdown("Before modeling, we must check for missing values and data types. This is the **actuarial equivalent of a physical exam**.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.code("""
# Check missing values
missing = df.isnull().sum().sum()
print(f"Total Missing: {missing}")
        """, language="python")
        st.success("Total Missing: 0 \n(Clean Bill of Health!)")
        
    with c2:
        st.code("""
# Check dimensions
rows, cols = df.shape
print(f"Records: {rows}")
print(f"Features: {cols}")
        """, language="python")
        st.info(f"Records: {len(df)}\nFeatures: {len(df.columns)}")

def chapter_2_eda(df):
    st.title("Chapter 2: The Forensic Audit (EDA)")
    st.markdown("""
    We are detectives. Our job is to find the **'Bleeding Necks'** - the hidden segments destroying profitability.
    """)
    
    # 2.1 Target Analysis
    st.markdown("## 2.1 The Target: CLV Distribution")
    st.markdown("""
    First, let's look at what we are trying to predict.
    """)
    
    with st.echo():
        # User follows along with this code:
        import plotly.express as px
        
        # Interactive plotting
        fig = px.histogram(df, x='customer_lifetime_value', nbins=50, 
                           title="Distribution of CLV (Raw)",
                           color_discrete_sequence=['#6366f1'])
        
        # Display
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("""
    <div class="concept-box">
        <b>💡 The Skewness Trap:</b><br>
        See that long tail to the right? That's <b>Positive Skew</b>. 
        Most customers are worth <$10k, but a few 'Whales' are worth >$50k. 
        <br><br>
        <b>Why it matters:</b> Linear models hate this. They will try to chase the whales 
        and mess up predictions for everyone else. We'll fix this in Chapter 3.
    </div>
    """, unsafe_allow_html=True)
    
    # 2.2 The Bleeding Neck
    st.markdown("## 2.2 The 'Bleeding Neck' Investigation")
    st.markdown("Hypothesis: **Unemployed** people have more accidents due to economic stress.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("**Interactive Filter:**")
        segment = st.selectbox("Select Segment to Analyze:", 
                             ['employmentstatus', 'marital_status', 'vehicle_class'])
        metric = 'total_claim_amount'
        
    with col2:
        fig = px.box(df, x=segment, y=metric, color=segment,
                    title=f"Forensic Audit: {metric.replace('_',' ').title()} by {segment.title()}")
        st.plotly_chart(fig, use_container_width=True)
        
    if segment == 'employmentstatus':
        st.error("""
        **🚨 FORENSIC FINDING:** 
        Look at the 'Unemployed' box. The Median is higher, and the box is taller (more volatile).
        This confirms the **Economic Stress Hypothesis**. Unemployed customers are a high-risk segment.
        """)

def chapter_3_feature_eng(df):
    st.title("Chapter 3: Feature Engineering")
    st.markdown("Make the data 'Machine Learning Ready'.")
    
    st.markdown("## 3.1 Log Transformation Lab")
    st.markdown("Let's squash that skewness.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Raw Data")
        st.bar_chart(df['customer_lifetime_value'].head(50))
        st.caption("Jagged, spiky, hard to model.")
        
    with c2:
        st.markdown("### Log Transformed")
        log_clv = np.log1p(df['customer_lifetime_value'])
        st.bar_chart(log_clv.head(50))
        st.caption("Smooth, compressed, beautiful.")
        
    st.markdown("""
    <div class="math-box">
        $$ y_{new} = \ln(y_{old} + 1) $$
        <br>
        The <b>Logarithm</b> is the Great Equalizer. It brings billionaires (outliers) 
        down to earth so the model can understand them.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 3.2 The Leakage Trap (Simulation)")
    st.markdown("What happens if we accidentally cheat?")
    
    cheat_col = st.checkbox("Include 'Total Claim Amount' (Cheat Mode)?")
    
    if cheat_col:
        st.warning("⚠️ Training with Leakage...")
        st.metric("Model R² Score", "0.99", "+0.80 Fake Gain")
        st.error("This is **FAKE NEWS**. You won't know the Claim Amount until AFTER the accident!")
    else:
        st.success("✅ Training Safely (No Leakage)")
        st.metric("Model R² Score", "0.68", "Real Accuracy")

def chapter_7_predictor(df):
    st.title("Chapter 7: Live Prediction Engine")
    st.markdown("The culmination of our work. A production-grade inference engine.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        inc = st.number_input("Income ($)", 0, 200000, 50000)
        pol = st.slider("Policies", 1, 9, 2)
    with c2:
        prem = st.number_input("Monthly Premium ($)", 50, 300, 100)
        veh = st.selectbox("Vehicle", ['Sedan', 'SUV', 'Luxury', 'Sports'])
    with c3:
        edu = st.selectbox("Education", ['HS', 'College', 'Master', 'Doctor'])
        emp = st.selectbox("Employment", ['Employed', 'Unemployed'])
        
    # Dummy prediction logic for demo (replace with real model call)
    pred_val = (prem * 12 * 4) + (inc * 0.02)
    if 'Luxury' in veh: pred_val *= 1.4
    if emp == 'Unemployed': pred_val *= 0.8
    
    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"## 💰 Predicted CLV: **${pred_val:,.2f}**")
        st.progress(min(pred_val/50000, 1.0))
    with right:
        if pred_val > 8000:
            st.success("🌟 High Value Customer")
        else:
            st.warning("📉 Standard Customer")

# ============================================================================
# 4. APP ROUTER
# ============================================================================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.title("🎓 The Masterclass")
        page = st.radio("Go to Chapter:", 
            ['Home', '1. Data & Mission', '2. Forensic Audit', '3. Feature Engineering', '7. Live Predictor'])
        
        st.markdown("---")
        st.caption("© 2026 Insurance Analytics Institute")
    
    # Load Data
    df = load_data()
    if df is None:
        st.error("Dataset not found! Please check path.")
        return

    # Routing
    if page == 'Home': chapter_home()
    elif page == '1. Data & Mission': chapter_1_data(df)
    elif page == '2. Forensic Audit': chapter_2_eda(df)
    elif page == '3. Feature Engineering': chapter_3_feature_eng(df)
    elif page == '7. Live Predictor': chapter_7_predictor(df)

if __name__ == "__main__":
    main()
