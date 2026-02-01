
# 🚗 Customer Lifetime Value (CLV) Analysis Project

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Executive Summary
This project provides a comprehensive analysis of Customer Lifetime Value (CLV) for an auto insurance company. It includes an end-to-end machine learning pipeline, deep exploratory data analysis, k-means clustering for customer segmentation, and an interactive Streamlit dashboard.

We analyzed **9,134 customers** to identify key value drivers and built a predictive model achieving **R² = 0.68** (predicting 68% of value variance).

---

## 🚀 Key Features

### 1. 📊 Deep Analysis & Segmentation
- **Drivers of Value**: Identified Monthly Premium (0.87 correlation) and Coverage Type (2.4x value multiplier) as top drivers.
- **Customer Personas**: Segmented customers into 4 strategic groups:
  - 💎 **High-Value Loyalists** (VIPs)
  - 🌱 **Growing Potentials** (Upsell targets)
  - 💰 **Premium Hunters** (Quality focused)
  - 🔄 **Price-Sensitive Switchers** (Efficiency focused)

### 2. 📊 Advanced Visual Analytics (New)
- **High-Resolution Plots**: All figures generated at 300 DPI for publication quality.
- **Complex Interactions**:
  - Violin plots for multivariate distribution analysis.
  - Hexbin density plots for premium vs. claims.
  - Pairplots for key metric relationships.

### 3. 🤖 Predictive Modeling Pipeline `scripts/`
- **Modular Design**: Separate steps for data cleaning, EDA, feature engineering, modeling, and inference.
- **Advanced Techniques**:
  - Log-transformation for skewed CLV targets.
  - Interaction features (e.g., `Risk_Score`, `Premium_per_Policy`).
  - Iterative model selection (Linear -> Tree -> Tuned Random Forest).
  - Cross-validation for robust performance estimation.

### 3. 🖥️ Interactive Dashboard `app.py`
- Real-time **CLV Predictor** for new customers.
- Interactive visualizations of customer segments and distributions.
- Overview of key portfolio metrics.

### 4. 📄 Comprehensive IEEE Report
- Automated PDF generation with embedded figures.
- Conversational, business-focused writing style.
- Detailed methodological appendix.

---

## 📂 Project Structure

```
CLV_IEEE_Project/
├── app.py                     # Streamlit Dashboard application
├── generate_pdf.py            # Automated PDF report generator
├── requirements.txt           # Project dependencies
├── models/                    # Saved ML models & preprocessors
│   ├── final_model.joblib
│   └── preprocessor.joblib
├── report/                    # Generated reports & figures
│   ├── IEEE_CLV_Analysis_Report.pdf
│   └── figures/               # 20+ generated visualizations
├── data/                      # Data storage
│   └── processed/             # Cleaned & featured datasets
└── scripts/                   # Analysis Pipeline
    ├── step_01_data_cleaning.py
    ├── step_02_eda.py
    ├── step_03_feature_engineering.py
    ├── step_04_modeling.py
    ├── step_05_inference.py
    ├── step_06_clustering_analysis.py   # NEW: Segmentation
    ├── step_07_deep_eda.py              # NEW: Marketing insights
    ├── step_08_model_iterations.py      # NEW: Model tuning
    └── run_all.py                       # Master execution script
```

---

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Analysis Pipeline
To regenerate all analysis, models, and figures:
```bash
python scripts/run_all.py
```
*Note: This will execute steps 1 through 8 sequentially.*

### 3. Launch Dashboard
To explore insights interactively:
```bash
streamlit run app.py
```

### 4. Generate PDF Report
To build the final PDF report:
```bash
python generate_pdf.py
```

---

## 📈 Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | **0.68** | Explains 68% of variance in customer value |
| **MAE** | **$1,378** | Average prediction error |
| **MAPE** | **8.8%** | Average percentage error |

**Key Insight**: The model significantly outperforms baseline approaches and linear regression (which failed to capture non-linear patterns).

---
*Generated for IEEE CLV Analysis Project*
