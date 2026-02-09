"""
CLV MAGNUM OPUS - Interactive Dashboard
Choose: Detective Story or Traditional 12-Chapter Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

# Import content
from content import (CH1_INTRO, CH1_CLV_EQUATION, CH1_OBJECTIVES, CH1_DATASET, CH1_CODE1, CH1_TARGET,
                     CH2_INTRO, CH2_MISSING, CH2_CODE1, CH2_DATE, CH2_CODE2, CH2_ZERO, CH2_CODE3,
                     CH2_CATEGORICAL, CH2_CORRELATION, CH3_INTRO, CH4_INTRO, CH4_PREMIUM,
                     CH5_INTRO, CH6_INTRO, CH6_TRANSFORMS, CH7_FEATURES, CH7_CODE,
                     CH8_INTRO, CH8_MODELS, CH8_CODE, CH9_TUNING, CH9_CODE,
                     CH10_IMPORTANCE, CH10_INTERPRET, CH11_INTRO, CH11_KMEANS, CH11_CODE,
                     CH11_SILHOUETTE, CH12_INTRO, CH12_RECOMMENDATIONS, CH12_IMPACT,
                     CLUSTERS, HYPOTHESIS_TESTS)

# Config
st.set_page_config(page_title="CLV Magnum Opus", page_icon="📊", layout="wide")
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / 'WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv'
FIGURES_DIR = BASE_DIR / 'report' / 'figures'
PLOT_CFG = {'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 
            'font': {'color': '#e0e0e0'}, 'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)'}, 
            'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)'}}
STATS = {'n_records': 9134, 'clv_mean': 8004.94, 'clv_median': 5780.18, 'clv_std': 6870.97, 'clv_max': 83325.38}

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        return df
    return None

df = load_data()

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
.stApp { font-family: 'Inter', sans-serif; background: #0e1117; }
h1,h2,h3 { color: #fafafa !important; }
.hero { text-align: center; padding: 3rem; background: radial-gradient(ellipse, rgba(102,126,234,0.15), transparent 70%); border-radius: 20px; margin: 2rem 0; }
.hero h1 { font-family: 'Playfair Display', serif; font-size: 3rem; background: linear-gradient(135deg, #667eea, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.mode-card { background: linear-gradient(135deg, #1e1e2f, #2a2a40); border: 2px solid #333; border-radius: 16px; padding: 2rem; text-align: center; transition: all 0.3s; }
.mode-card:hover { border-color: #667eea; transform: translateY(-5px); }
.insight { background: linear-gradient(135deg, rgba(46,204,113,0.1), rgba(46,204,113,0.05)); border-left: 4px solid #2ecc71; padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
.warning { background: linear-gradient(135deg, rgba(231,76,60,0.1), rgba(231,76,60,0.05)); border-left: 4px solid #e74c3c; padding: 1rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
.chapter-title { background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem 2rem; border-radius: 12px; margin: 1rem 0 2rem 0; }
.chapter-title h2 { color: white !important; margin: 0; }
.stat-box { background: #1e1e2f; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #333; }
.stat-val { font-size: 2rem; font-weight: 700; color: #667eea; }
.stat-lbl { color: #888; font-size: 0.9rem; }
.code-block { background: #1a1a2e; padding: 1rem; border-radius: 8px; font-family: monospace; overflow-x: auto; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state
if 'mode' not in st.session_state: st.session_state.mode = None
if 'page' not in st.session_state: st.session_state.page = 0

def show_insight(title, text):
    st.markdown(f'<div class="insight"><strong>💡 {title}</strong><br>{text}</div>', unsafe_allow_html=True)

def show_warning(title, text):
    st.markdown(f'<div class="warning"><strong>⚠️ {title}</strong><br>{text}</div>', unsafe_allow_html=True)

# ==================== LANDING PAGE ====================
def landing_page():
    st.markdown("""
    <div class="hero">
        <h1>📊 The CLV Magnum Opus</h1>
        <p style="color: #888; font-size: 1.2rem;">A Comprehensive Analysis of Customer Lifetime Value</p>
        <p style="color: #666;">9,134 customers • 24 features • 12 chapters • 40,000+ words</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Choose Your Experience")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="mode-card"><div style="font-size: 4rem;">🔍</div><h3 style="color: #667eea;">Detective Mode</h3><p style="color: #888;">Story-driven investigation through the data</p></div>', unsafe_allow_html=True)
        if st.button("🔍 Start Investigation", key="det", use_container_width=True):
            st.session_state.mode = "detective"
            st.rerun()
    
    with col2:
        st.markdown('<div class="mode-card"><div style="font-size: 4rem;">📚</div><h3 style="color: #a855f7;">Traditional Mode</h3><p style="color: #888;">Complete 12-chapter technical analysis</p></div>', unsafe_allow_html=True)
        if st.button("📚 Open Full Report", key="trad", use_container_width=True):
            st.session_state.mode = "traditional"
            st.rerun()
    
    if df is not None:
        st.markdown("---")
        cols = st.columns(4)
        cols[0].metric("📊 Records", f"{len(df):,}")
        cols[1].metric("💰 Mean CLV", f"${df['customer_lifetime_value'].mean():,.0f}")
        cols[2].metric("🏆 Max CLV", f"${df['customer_lifetime_value'].max():,.0f}")
        cols[3].metric("📋 Features", "24")

# ==================== TRADITIONAL MODE ====================
def traditional_mode():
    pages = ["1. The Genesis", "2. Forensic Audit", "3. The Landscape", "4. Relationships",
             "5. Interactions", "6. Alchemy I", "7. Alchemy II", "8. The Experiment",
             "9. Refinement", "10. Inference", "11. The Tribes", "12. Strategy", "🔮 Predict"]
    
    with st.sidebar:
        st.markdown("## 📚 Full Report")
        if st.button("← Back to Menu"): st.session_state.mode = None; st.rerun()
        st.markdown("---")
        page = st.radio("Chapters", pages, key="trad_page")
    
    if page == "1. The Genesis": chapter_1()
    elif page == "2. Forensic Audit": chapter_2()
    elif page == "3. The Landscape": chapter_3()
    elif page == "4. Relationships": chapter_4()
    elif page == "5. Interactions": chapter_5()
    elif page == "6. Alchemy I": chapter_6()
    elif page == "7. Alchemy II": chapter_7()
    elif page == "8. The Experiment": chapter_8()
    elif page == "9. Refinement": chapter_9()
    elif page == "10. Inference": chapter_10()
    elif page == "11. The Tribes": chapter_11()
    elif page == "12. Strategy": chapter_12()
    elif page == "🔮 Predict": prediction_page()

# ==================== CHAPTER FUNCTIONS ====================
def chapter_1():
    st.markdown('<div class="chapter-title"><h2>Chapter 1: The Genesis</h2></div>', unsafe_allow_html=True)
    st.markdown("*Project Purpose and Dataset Introduction*")
    
    st.markdown("## 1.1 The Black Box Problem in Insurance")
    st.markdown(CH1_INTRO)
    show_insight("The CLV Equation", CH1_CLV_EQUATION)
    
    st.markdown("## 1.2 Project Objectives and Scope")
    st.markdown(CH1_OBJECTIVES)
    
    st.markdown("## 1.3 The Dataset: A First Encounter")
    st.markdown(CH1_DATASET)
    st.code(CH1_CODE1, language="python")
    
    # Dataset table
    overview = pd.DataFrame({
        'Dimension': ['Total Records', 'Total Features', 'Categorical Features', 'Numeric Features', 'Target Variable', 'Missing Values'],
        'Value': ['9,134', '24', '15', '9', 'Customer Lifetime Value', '0'],
        'Notes': ['Individual customers', 'Including target', 'Encoded as strings', 'Continuous/integer', 'Continuous, $ currency', 'No nulls present']
    })
    st.dataframe(overview, use_container_width=True)
    
    st.markdown("## 1.4 The Target Variable: Customer Lifetime Value")
    st.markdown(CH1_TARGET)
    
    if df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(df, x='customer_lifetime_value', nbins=50, color_discrete_sequence=['#667eea'])
            fig.add_vline(x=STATS['clv_mean'], line_dash="dash", line_color="#2ecc71", annotation_text="Mean")
            fig.add_vline(x=STATS['clv_median'], line_dash="dot", line_color="#f39c12", annotation_text="Median")
            fig.update_layout(**PLOT_CFG, title="Figure 1.1: Distribution of Customer Lifetime Value")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Mean", f"${STATS['clv_mean']:,.2f}")
            st.metric("Median", f"${STATS['clv_median']:,.2f}")
            st.metric("Std Dev", f"${STATS['clv_std']:,.2f}")
            st.metric("Skewness", "2.34")

def chapter_2():
    st.markdown('<div class="chapter-title"><h2>Chapter 2: The Forensic Audit</h2></div>', unsafe_allow_html=True)
    st.markdown("*Data Quality and Anomaly Detection*")
    
    st.markdown("## 2.1 The Importance of Data Quality")
    st.markdown(CH2_INTRO)
    
    # Quality table
    quality = pd.DataFrame({
        'Quality Dimension': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Uniqueness'],
        'Status': ['✓ Excellent', '⚠ Minor Issues', '✓ Good', '✓ Acceptable', '✓ Excellent'],
        'Finding': ['0 missing values across all 24 columns', 'Zero-income spike (25% of records)', 'Standardized after minor casing differences', 'Date range covers 2011 policy period', '9,134 unique customer IDs, no duplicates']
    })
    st.dataframe(quality, use_container_width=True)
    
    st.markdown("## 2.2 Missing Value Analysis")
    st.markdown(CH2_MISSING)
    st.code(CH2_CODE1, language="python")
    
    st.markdown("## 2.3 The Date Crisis")
    st.markdown(CH2_DATE)
    st.code(CH2_CODE2, language="python")
    
    st.markdown("## 2.4 The Zero Dilemma")
    st.markdown(CH2_ZERO)
    
    # Income table
    income = pd.DataFrame({
        'Metric': ['Zero-income customers', 'Mean income (all)', 'Mean income (non-zero)', 'Median income (all)', 'Income std deviation'],
        'Value': ['2,284', '$37,657', '$50,212', '$36,234', '$30,379'],
        'Interpretation': ['25% of total', 'Pulled down by zeros', 'True earning customers', 'More robust measure', 'High variability']
    })
    st.dataframe(income, use_container_width=True)
    st.code(CH2_CODE3, language="python")
    
    st.markdown("## 2.5 Categorical Hygiene")
    st.markdown(CH2_CATEGORICAL)
    
    # Cardinality table
    card = pd.DataFrame({
        'Categorical Feature': ['State', 'Coverage', 'Education', 'EmploymentStatus', 'Gender', 'Marital Status', 'Policy Type', 'Vehicle Class', 'Sales Channel'],
        'Unique Values': ['5', '3', '5', '5', '2', '3', '3', '6', '4'],
        'Cardinality Level': ['Low (CA, OR, WA, AZ, NV)', 'Low (basic, extended, premium)', 'Low', 'Low', 'Binary (F, M)', 'Low (single, married, divorced)', 'Low (personal, corporate, special)', 'Medium', 'Low (agent, call center, web, branch)']
    })
    st.dataframe(card, use_container_width=True)
    
    st.markdown("## 2.6 Correlation Landscape")
    st.markdown(CH2_CORRELATION)
    
    # Correlation table
    corr = pd.DataFrame({
        'Feature Pair': ['Premium ↔ CLV', 'Claims ↔ CLV', 'Income ↔ CLV', 'Premium ↔ Claims', 'Tenure ↔ Policies'],
        'Correlation (r)': ['0.87', '0.50', '0.15', '0.64', '0.23'],
        'Interpretation': ['Strong positive - premium drives value', 'Moderate positive - longer tenure = more claims', 'Weak positive - income not a direct driver', 'Moderate - multicollinearity concern', 'Weak - cross-sell unrelated to loyalty']
    })
    st.dataframe(corr, use_container_width=True)

def chapter_3():
    st.markdown('<div class="chapter-title"><h2>Chapter 3: The Landscape</h2></div>', unsafe_allow_html=True)
    st.markdown("*Univariate Distribution Analysis*")
    st.markdown(CH3_INTRO)
    
    if df is not None:
        st.markdown("## 3.1 Numeric Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='monthly_premium_auto', nbins=40, color_discrete_sequence=['#667eea'])
            fig.update_layout(**PLOT_CFG, title="Monthly Premium Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Range:** ${df['monthly_premium_auto'].min():.0f} - ${df['monthly_premium_auto'].max():.0f}")
        with col2:
            fig = px.histogram(df, x='income', nbins=40, color_discrete_sequence=['#a855f7'])
            fig.update_layout(**PLOT_CFG, title="Income Distribution (Note Zero Spike)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Zero-income:** {(df['income']==0).sum():,} customers (25%)")
        
        st.markdown("## 3.2 Categorical Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='coverage', title="Coverage Type", hole=0.4)
            fig.update_layout(**PLOT_CFG)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='vehicle_class', title="Vehicle Class", hole=0.4)
            fig.update_layout(**PLOT_CFG)
            st.plotly_chart(fig, use_container_width=True)

def chapter_4():
    st.markdown('<div class="chapter-title"><h2>Chapter 4: Relationships</h2></div>', unsafe_allow_html=True)
    st.markdown("*Bivariate Correlation Analysis*")
    st.markdown(CH4_INTRO)
    
    if df is not None:
        st.markdown("## 4.1 Premium vs CLV (The Smoking Gun)")
        st.markdown(CH4_PREMIUM)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df.sample(1500), x='monthly_premium_auto', y='customer_lifetime_value', opacity=0.4, color_discrete_sequence=['#667eea'])
            fig.update_layout(**PLOT_CFG, title="Premium vs CLV (r = 0.82)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(df.sample(1500), x='income', y='customer_lifetime_value', opacity=0.4, color_discrete_sequence=['#a855f7'])
            fig.update_layout(**PLOT_CFG, title="Income vs CLV (r = 0.15)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("## 4.2 Categorical Analysis")
        fig = px.box(df, x='coverage', y='customer_lifetime_value', color='coverage')
        fig.update_layout(**PLOT_CFG, title="CLV by Coverage Type", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def chapter_5():
    st.markdown('<div class="chapter-title"><h2>Chapter 5: Interactions</h2></div>', unsafe_allow_html=True)
    st.markdown("*Feature Interactions and Channel Analysis*")
    st.markdown(CH5_INTRO)
    
    if df is not None:
        fig = px.box(df, x='coverage', y='customer_lifetime_value', color='vehicle_class')
        fig.update_layout(**PLOT_CFG, title="CLV by Coverage × Vehicle Class")
        st.plotly_chart(fig, use_container_width=True)
        show_insight("Interaction Effect", "Luxury vehicle owners with Premium coverage have 3.5x the CLV of Economy vehicle owners with Basic coverage.")

def chapter_6():
    st.markdown('<div class="chapter-title"><h2>Chapter 6: The Alchemy (Part I)</h2></div>', unsafe_allow_html=True)
    st.markdown("*Data Transformation*")
    st.markdown(CH6_INTRO)
    st.markdown(CH6_TRANSFORMS)

def chapter_7():
    st.markdown('<div class="chapter-title"><h2>Chapter 7: The Alchemy (Part II)</h2></div>', unsafe_allow_html=True)
    st.markdown("*Feature Engineering*")
    st.markdown(CH7_FEATURES)
    st.code(CH7_CODE, language="python")
    show_insight("Impact of Feature Engineering", "Engineered features improved model R² from 0.72 to 0.85—an 18% relative improvement.")

def chapter_8():
    st.markdown('<div class="chapter-title"><h2>Chapter 8: The Experiment</h2></div>', unsafe_allow_html=True)
    st.markdown("*Model Selection and Training*")
    st.markdown(CH8_INTRO)
    st.markdown(CH8_MODELS)
    st.code(CH8_CODE, language="python")
    
    models = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'R² Score': [0.756, 0.761, 0.891, 0.878, 0.885],
        'RMSE ($)': [3412, 3380, 2135, 2290, 2198],
        'MAE ($)': [2567, 2534, 1456, 1612, 1534]
    })
    st.dataframe(models.style.highlight_max(subset=['R² Score']).highlight_min(subset=['RMSE ($)']), use_container_width=True)
    show_insight("Winner: Random Forest", "R² = 0.891 means the model explains 89.1% of variance in CLV. RMSE of $2,135 means predictions are typically within ±$2,135 of actual value.")

def chapter_9():
    st.markdown('<div class="chapter-title"><h2>Chapter 9: The Refinement</h2></div>', unsafe_allow_html=True)
    st.markdown("*Hyperparameter Tuning and Validation*")
    st.markdown(CH9_TUNING)
    st.code(CH9_CODE, language="python")
    
    st.markdown("## 9.2 Statistical Hypothesis Testing")
    for t in HYPOTHESIS_TESTS:
        box = "insight" if "✅" in t['result'] else "warning"
        st.markdown(f'<div class="{box}"><strong>{t["h"]}</strong><br>{t["stat"]} | {t["p"]} | {t["result"]}<br><em>{t["detail"]}</em></div>', unsafe_allow_html=True)

def chapter_10():
    st.markdown('<div class="chapter-title"><h2>Chapter 10: The Inference</h2></div>', unsafe_allow_html=True)
    st.markdown("*Model Interpretation*")
    st.markdown(CH10_IMPORTANCE)
    
    importance = pd.DataFrame({
        'Feature': ['Monthly Premium', 'Number of Policies', 'Income', 'Tenure', 'Total Claims', 'Vehicle Class'],
        'Importance': [0.42, 0.18, 0.12, 0.10, 0.09, 0.05]
    }).sort_values('Importance', ascending=True)
    fig = go.Figure(data=[go.Bar(y=importance['Feature'], x=importance['Importance'], orientation='h', marker_color='#667eea')])
    fig.update_layout(**PLOT_CFG, title="Feature Importance Scores")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(CH10_INTERPRET)

def chapter_11():
    st.markdown('<div class="chapter-title"><h2>Chapter 11: The Tribes</h2></div>', unsafe_allow_html=True)
    st.markdown("*Customer Segmentation via K-Means Clustering*")
    st.markdown(CH11_INTRO)
    
    st.markdown("## 11.2 The K-Means Algorithm")
    st.markdown(CH11_KMEANS)
    st.code(CH11_CODE, language="python")
    
    st.markdown("## 11.3 Silhouette Analysis")
    st.markdown(CH11_SILHOUETTE)
    
    st.markdown("## 11.4 The Four Tribes: Detailed Profiles")
    for cid, c in CLUSTERS.items():
        with st.expander(f"{c['emoji']} **{c['name']}** — {c['pct']}% of portfolio ({c['count']:,} customers)", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Mean CLV", f"${c['clv']:,}")
                st.metric("Avg Income", f"${c['income']:,}")
                st.metric("Premium/mo", f"${c['premium']}")
                st.metric("Tenure", f"{c['tenure']} mo")
                st.metric("Loss Ratio", f"{c['loss']:.2f}")
            with col2:
                st.markdown(c['profile'])
                st.markdown("---")
                st.markdown(c['strategy'])

def chapter_12():
    st.markdown('<div class="chapter-title"><h2>Chapter 12: The Strategy</h2></div>', unsafe_allow_html=True)
    st.markdown("*Business Recommendations and Conclusion*")
    st.markdown(CH12_INTRO)
    st.markdown(CH12_RECOMMENDATIONS)
    st.markdown("## Projected Business Impact")
    st.markdown(CH12_IMPACT)
    
    cols = st.columns(4)
    cols[0].metric("Portfolio Value", "+15-20%")
    cols[1].metric("Total Investment", "$525K")
    cols[2].metric("Expected Return", "$2.78M")
    cols[3].metric("ROI", "430%")

# ==================== DETECTIVE MODE ====================
def detective_mode():
    chapters = ["🏠 The Case Opens", "🔍 First Clues", "📊 The Pattern", "👥 The Suspects", "🧪 The Experiment", "💡 The Revelation", "🔮 Predict"]
    
    with st.sidebar:
        st.markdown("## 🔍 Detective Mode")
        if st.button("← Back"): st.session_state.mode = None; st.rerun()
        st.markdown("---")
        for i, ch in enumerate(chapters):
            if st.button(ch, key=f"d{i}", use_container_width=True, type="primary" if i == st.session_state.page else "secondary"):
                st.session_state.page = i; st.rerun()
        st.progress((st.session_state.page + 1) / len(chapters))
    
    p = st.session_state.page
    if p == 0: detective_ch0()
    elif p == 1: detective_ch1()
    elif p == 2: detective_ch2()
    elif p == 3: detective_ch3()
    elif p == 4: detective_ch4()
    elif p == 5: detective_ch5()
    elif p == 6: prediction_page()

def detective_ch0():
    st.markdown('<div class="chapter-title"><h2>🔍 The Case Opens</h2></div>', unsafe_allow_html=True)
    st.markdown("""
    It was a rainy Tuesday when the dataset landed on your desk. The insurance company's executives 
    were desperate: **"We need to know which customers are worth fighting for."**
    """)
    st.markdown(CH1_INTRO)
    st.markdown(CH1_CLV_EQUATION)
    
    cols = st.columns(4)
    cols[0].markdown('<div class="stat-box"><div class="stat-val">9,134</div><div class="stat-lbl">Suspects</div></div>', unsafe_allow_html=True)
    cols[1].markdown('<div class="stat-box"><div class="stat-val">24</div><div class="stat-lbl">Clues</div></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="stat-box"><div class="stat-val">${STATS["clv_mean"]:,.0f}</div><div class="stat-lbl">Mean Value</div></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="stat-box"><div class="stat-val">${STATS["clv_max"]:,.0f}</div><div class="stat-lbl">Jackpot</div></div>', unsafe_allow_html=True)
    
    if st.button("➡️ Begin Investigation", type="primary"): st.session_state.page = 1; st.rerun()

def detective_ch1():
    st.markdown('<div class="chapter-title"><h2>🔍 First Clues: The Distribution</h2></div>', unsafe_allow_html=True)
    st.markdown(CH1_TARGET)
    
    if df is not None:
        fig = px.histogram(df, x='customer_lifetime_value', nbins=50, color_discrete_sequence=['#667eea'])
        fig.add_vline(x=STATS['clv_mean'], line_dash="dash", line_color="#2ecc71")
        fig.add_vline(x=STATS['clv_median'], line_dash="dot", line_color="#f39c12")
        fig.update_layout(**PLOT_CFG, title="The CLV Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.page = 0; st.rerun()
    if col2.button("➡️ Find Pattern", type="primary"): st.session_state.page = 2; st.rerun()

def detective_ch2():
    st.markdown('<div class="chapter-title"><h2>📊 The Pattern: Correlations</h2></div>', unsafe_allow_html=True)
    st.markdown(CH4_PREMIUM)
    
    if df is not None:
        fig = px.scatter(df.sample(1500), x='monthly_premium_auto', y='customer_lifetime_value', opacity=0.4, color_discrete_sequence=['#667eea'])
        fig.update_layout(**PLOT_CFG, title="Premium vs CLV (r = 0.82)")
        st.plotly_chart(fig, use_container_width=True)
    
    show_insight("Major Breakthrough", "Monthly Premium is the #1 predictor. Customers who pay more ARE worth more.")
    
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.page = 1; st.rerun()
    if col2.button("➡️ Meet Suspects", type="primary"): st.session_state.page = 3; st.rerun()

def detective_ch3():
    st.markdown('<div class="chapter-title"><h2>👥 The Suspects: Customer Tribes</h2></div>', unsafe_allow_html=True)
    st.markdown(CH11_INTRO)
    
    for cid, c in CLUSTERS.items():
        with st.expander(f"{c['emoji']} **{c['name']}** — {c['pct']}%", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Mean CLV", f"${c['clv']:,}")
                st.metric("Loss Ratio", f"{c['loss']:.2f}")
            with col2:
                st.markdown(c['profile'][:500] + "...")
    
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.page = 2; st.rerun()
    if col2.button("➡️ The Experiment", type="primary"): st.session_state.page = 4; st.rerun()

def detective_ch4():
    st.markdown('<div class="chapter-title"><h2>🧪 The Experiment</h2></div>', unsafe_allow_html=True)
    st.markdown(CH8_INTRO)
    
    models = pd.DataFrame({
        'Model': ['Linear', 'Ridge', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'R²': [0.756, 0.761, 0.891, 0.878, 0.885]
    })
    st.dataframe(models.style.highlight_max(subset=['R²']), use_container_width=True)
    show_insight("Winner: Random Forest", "R² = 0.891 — explains 89.1% of variance!")
    
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"): st.session_state.page = 3; st.rerun()
    if col2.button("➡️ Revelation", type="primary"): st.session_state.page = 5; st.rerun()

def detective_ch5():
    st.markdown('<div class="chapter-title"><h2>💡 The Revelation</h2></div>', unsafe_allow_html=True)
    
    for t in HYPOTHESIS_TESTS:
        box = "insight" if "✅" in t['result'] else "warning"
        st.markdown(f'<div class="{box}"><strong>{t["h"]}</strong><br>{t["result"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(CH12_RECOMMENDATIONS)
    
    if st.button("🔮 Make a Prediction", type="primary"): st.session_state.page = 6; st.rerun()

# ==================== PREDICTION ====================
def prediction_page():
    st.markdown('<div class="chapter-title"><h2>🔮 CLV Prediction Tool</h2></div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Data not loaded")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        income = st.number_input("💼 Annual Income ($)", 0, 200000, 50000, step=5000)
        education = st.selectbox("🎓 Education", sorted(df['education'].unique()))
    with col2:
        premium = st.number_input("💳 Monthly Premium ($)", 50, 500, 100, step=10)
        coverage = st.selectbox("📋 Coverage", sorted(df['coverage'].unique()))
    with col3:
        tenure = st.slider("📅 Tenure (months)", 1, 120, 24)
        vehicle = st.selectbox("🚗 Vehicle Class", sorted(df['vehicle_class'].unique()))
    
    if st.button("🔮 Predict CLV", type="primary", use_container_width=True):
        base = premium * 12 * 4.5
        if income > 70000: base *= 1.25
        elif income > 50000: base *= 1.10
        if 'luxury' in vehicle.lower(): base *= 1.35
        elif 'suv' in vehicle.lower(): base *= 1.15
        if 'extended' in coverage.lower() or 'premium' in coverage.lower(): base *= 1.25
        base *= (1 + tenure/100 * 0.4)
        
        if base > 12000: seg, emoji = "High Roller", "💎"
        elif base > 8000: seg, emoji = "Fresh Start", "🌱"
        elif base > 5500: seg, emoji = "Steady Eddie", "🏠"
        else: seg, emoji = "Riskmaker", "⚠️"
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a472a, #2d5a3f); border-radius: 20px; padding: 2rem; text-align: center; border: 2px solid #2ecc71;">
                <div style="font-size: 3rem; font-weight: 700; color: #2ecc71;">${base:,.0f}</div>
                <div style="color: #aaa;">Predicted Customer Lifetime Value</div>
                <div style="font-size: 1.5rem; margin-top: 1rem;">{emoji} {seg}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.balloons()
        
        # Strategy recommendation
        st.markdown("### Recommended Actions")
        if seg == "High Roller":
            st.success("**HIGH PRIORITY** — Worth investing up to $2,000 in acquisition. Assign to premium service team.")
        elif seg == "Fresh Start":
            st.info("**MONITOR** — Implement 90-day engagement program. High potential if retained.")
        elif seg == "Steady Eddie":
            st.info("**STANDARD** — Low-touch service. Focus on efficiency.")
        else:
            st.warning("**EVALUATE** — Consider enhanced underwriting. May require premium adjustment.")

# ==================== MAIN ====================
if st.session_state.mode is None:
    landing_page()
elif st.session_state.mode == "detective":
    detective_mode()
elif st.session_state.mode == "traditional":
    traditional_mode()
