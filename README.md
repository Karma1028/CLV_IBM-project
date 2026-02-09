# Customer Lifetime Value (CLV) Analysis Project

A comprehensive data analysis project investigating Customer Lifetime Value through an interactive, story-driven web experience. This project combines rigorous statistical analysis with modern web design to present insights in an engaging format.

## 🌟 Live Demo

[View the Project Live](https://unique-rugelach-c75c88.netlify.app/)

## 🔍 Project Overview

The core objective is to answer: **"What is a customer truly worth?"**

Using a dataset of 9,134 customers with 24 variables, we perform a deep-dive forensic analysis to predict CLV and segment customers into actionable groups.

### Key Features:

- **🕵️‍♀️ Detective Mode**: An immersive, chapter-by-chapter storytelling experience that guides you through the analysis step-by-step.
- **📊 Traditional Mode**: A comprehensive, single-column scrollable report for quick access to all data and charts.
- **🔮 CLV Predictor**: An interactive tool to calculate predicted CLV based on customer inputs (Income, Policy Tenure, etc.).
- **🤖 AI Assistant**: Integration with AI to provide on-demand insights and explanations about the data.
- **📈 Advanced Visualizations**: Interactive Plotly charts for deep data exploration.

## 🛠️ Technology Stack

- **Frontend**: 
  - HTML5, CSS3 (Custom animations, Glassmorphism design)
  - Vanilla JavaScript (ES6+)
  - Plotly.js (Data Visualization)
  - Marked.js (Markdown rendering)
- **Backend / Analysis**:
  - Python (Pandas, Scikit-learn, XGBoost)
  - Streamlit (Initial prototyping)
  - Netlify (Deployment)

## 📂 Project Structure

```
/
├── website/              # Production-ready web application
│   ├── css/              # Main stylesheet with animations
│   ├── js/               # Application logic
│   │   ├── main.js       # Core functionality & UI handling
│   │   ├── chapters.js   # Content for all 10 analysis chapters
│   │   └── ai-integration.js # AI assistant logic
│   └── index.html        # Main entry point
├── content.py            # Python source for analysis content
└── netlify.toml          # Deployment configuration
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/CLV_IEEE_Project.git
   ```

2. **Run Locally:**
   - Simply open `website/index.html` in your browser.
   - Or use a simple HTTP server:
     ```bash
     cd website
     python -m http.server 8000
     ```

## 📊 Analysis Highlights

- **Model Accuracy**: ~89.1% using XGBoost Regressor.
- **Customer Segmentation**: Identified 4 distinct "Tribes" tailored for specific marketing strategies.
- **Projected ROI**: Strategy implementation estimated to yield ~$2.78M return.

## 📝 License

This project is open-source and available under the MIT License.

---
**Analysis by Tuhin Bhattacharya**  
PGDM Business Data Analytics  
Goa Institute of Management
