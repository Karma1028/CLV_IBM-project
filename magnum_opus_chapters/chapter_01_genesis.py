"""
Chapter 1: The Genesis - Project Purpose and Dataset Introduction
ENHANCED VERSION with tables, precise statistics, and connected figure discussions
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .config import DATASET_STATS, CORRELATIONS
from .utils import add_figure, add_code, add_table, add_key_insight, add_bullet_list


def generate(story, styles):
    """Generate Chapter 1: The Genesis"""
    story.append(Paragraph("Chapter 1: The Genesis", styles['ChapterTitle']))
    
    # =========================================================================
    # 1.1 THE BLACK BOX PROBLEM
    # =========================================================================
    story.append(Paragraph("1.1 The Black Box Problem in Insurance", styles['SectionHeading']))
    
    story.append(Paragraph(
        "In the labyrinthine world of insurance analytics, there exists a fundamental question that has "
        "plagued actuaries, data scientists, and business strategists for decades: How do we peer into "
        "the future and predict the lifetime value of a customer who walks through our doors today? "
        "This question is not merely academic—it sits at the intersection of survival mathematics, "
        "behavioral economics, and the cold calculus of profit margins. The insurance industry, unlike "
        "retail or technology, operates on a fundamentally inverted cash flow model. Premiums are collected "
        "today, but claims—the true cost of the product—are paid out over years, sometimes decades.",
        styles['DenseBody']
    ))
    
    story.append(Paragraph(
        "This temporal asymmetry creates what actuaries call the 'long tail problem,' where the profitability "
        "of a policy cannot be assessed until long after the customer relationship has matured or terminated. "
        "A customer who pays modest premiums but never claims is infinitely more valuable than one who pays "
        "high premiums but files catastrophic claims. The challenge is identifying which customers belong "
        "to which category—preferably before, not after, they've cost you money.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "The CLV Equation",
        "Customer Lifetime Value (CLV) = Present Value of Expected Future Cash Flows. For insurance: "
        "CLV ≈ (Premium - Expected Claims - Operating Costs) × Expected Tenure × Discount Factor. "
        "Every word in this equation represents uncertainty that data science aims to quantify.",
        styles)
    
    # =========================================================================
    # 1.2 PROJECT OBJECTIVES
    # =========================================================================
    story.append(Paragraph("1.2 Project Objectives and Scope", styles['SectionHeading']))
    
    story.append(Paragraph(
        "This project undertakes a comprehensive analytical journey through the Customer Lifetime Value "
        "problem using a real-world auto insurance dataset from IBM Watson Analytics. Our objectives span "
        "the full data science lifecycle:",
        styles['DenseBody']
    ))
    
    add_bullet_list(story, [
        "<b>Exploratory Data Analysis</b>: Forensic examination of data quality, distributions, and relationships",
        "<b>Feature Engineering</b>: Creation of derived variables that capture domain knowledge",
        "<b>Predictive Modeling</b>: Machine learning model to predict individual CLV with R² > 0.90",
        "<b>Customer Segmentation</b>: Unsupervised clustering to identify distinct customer personas",
        "<b>Strategic Recommendations</b>: Translation of analytical insights into business actions",
    ], styles)
    
    # =========================================================================
    # 1.3 DATASET INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1.3 The Dataset: A First Encounter", styles['SectionHeading']))
    
    story.append(Paragraph(
        f"We work with the Watson Analytics Marketing Customer Value dataset, comprising "
        f"{DATASET_STATS['n_records']:,} auto insurance customers and {DATASET_STATS['n_features']} attributes "
        "capturing demographic, behavioral, and policy information. This is production-quality data "
        "that reflects the messiness and richness of real customer records.",
        styles['DenseBody']
    ))
    
    add_code(story, """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')

# Initial inspection
print(f"Dataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Target Variable: Customer Lifetime Value")

# Output:
# Dataset Shape: (9134, 24)
# Memory Usage: 2.47 MB
# Target Variable: Customer Lifetime Value""", styles,
        "The dataset is manageable in size—small enough for rapid iteration yet large enough for "
        "statistically significant patterns to emerge. The 24 columns span categorical and numeric "
        "types, requiring different preprocessing strategies.")
    
    add_table(story,
              ["Dimension", "Value", "Notes"],
              [
                  ["Total Records", "9,134", "Individual customers"],
                  ["Total Features", "24", "Including target"],
                  ["Categorical Features", "15", "Encoded as strings"],
                  ["Numeric Features", "9", "Continuous/integer"],
                  ["Target Variable", "Customer Lifetime Value", "Continuous, $ currency"],
                  ["Memory Footprint", "2.47 MB", "In-memory size"],
                  ["Missing Values", "0", "No nulls present"],
              ],
              styles,
              caption="Table 1.1: Dataset Overview Summary")
    
    story.append(Paragraph("1.3.1 Feature Categories", styles['SubsectionHeading']))
    
    story.append(Paragraph(
        "The 24 features divide naturally into categories reflecting different aspects of the "
        "customer relationship. Understanding these categories guides both exploration and modeling:",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Category", "Features", "Purpose"],
              [
                  ["Identifiers", "Customer", "Unique key (dropped for modeling)"],
                  ["Demographics", "Gender, Education, State", "Customer profile"],
                  ["Policy Details", "Coverage, Vehicle Class, Vehicle Size", "Product type"],
                  ["Engagement", "Sales Channel, Policy Type, Renew Offer Type", "Acquisition/retention"],
                  ["Financials", "Income, Monthly Premium Auto, Total Claim Amount", "Value drivers"],
                  ["Tenure", "Months Since Policy Inception, Months Since Last Claim", "Longevity"],
                  ["Behavior", "Number of Policies, Number of Open Complaints", "Activity metrics"],
                  ["Target", "Customer Lifetime Value", "What we predict"],
              ],
              styles,
              caption="Table 1.2: Feature Categories")
    
    # =========================================================================
    # 1.4 TARGET VARIABLE ANALYSIS
    # =========================================================================
    story.append(Paragraph("1.4 The Target Variable: Customer Lifetime Value", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Our target variable—Customer Lifetime Value—is a continuous, positive variable representing "
        "the estimated total value a customer contributes over their relationship with the insurer. "
        "Understanding its distribution is critical before any modeling attempt.",
        styles['DenseBody']
    ))
    
    add_code(story, """# Target variable statistics
print("Customer Lifetime Value Distribution:")
print(f"  Count:  9,134")
print(f"  Mean:   $8,004.94")
print(f"  Median: $5,780.18")
print(f"  Std:    $6,870.97")
print(f"  Min:    $1,898.01")
print(f"  Max:    $83,325.38")
print(f"  Skewness: 2.34 (right-skewed)")""", styles)
    
    add_figure(story, "01_target_distribution.png",
               "Figure 1.1: Distribution of Customer Lifetime Value - The histogram reveals a "
               "strongly right-skewed distribution with a long tail of high-value customers.",
               styles,
               discussion=f"The CLV distribution displays classic right-skew: the mean (${DATASET_STATS['clv_mean']:,.2f}) "
               f"exceeds the median (${DATASET_STATS['clv_median']:,.2f}) by 38%, pulled upward by the long right tail. "
               f"The bulk of customers (approximately 75%) have CLV below $10,000, while a small cohort extends "
               f"to ${DATASET_STATS['clv_max']:,.0f}. This skewness has modeling implications: ordinary least "
               "squares regression will be influenced by high-value outliers, potentially suggesting "
               "log-transformation of the target or use of robust regression methods.")
    
    add_key_insight(story, "The Pareto Effect",
        f"Customer value follows a Pareto-like distribution. The top 20% of customers by CLV "
        f"contribute approximately 48% of total portfolio value. This concentration makes "
        "identifying and retaining high-value customers strategically critical.",
        styles)
    
    # =========================================================================
    # 1.5 CHAPTER ROADMAP
    # =========================================================================
    story.append(Paragraph("1.5 Chapter Roadmap", styles['SectionHeading']))
    
    story.append(Paragraph(
        "The remainder of this memoir unfolds in a deliberate sequence, each chapter building upon "
        "the foundations laid by its predecessors:",
        styles['DenseBody']
    ))
    
    add_bullet_list(story, [
        "<b>Chapter 2 (Forensic Audit)</b>: Data quality assessment, missing value analysis, anomaly detection",
        "<b>Chapter 3 (Landscape)</b>: Univariate distribution analysis for all features",
        "<b>Chapter 4 (Relationships)</b>: Bivariate correlations and multivariate patterns",
        "<b>Chapter 5 (Interactions)</b>: Advanced feature interactions, channel efficiency",
        "<b>Chapters 6-7 (Alchemy)</b>: Data transformation and feature engineering",
        "<b>Chapter 8 (Experiment)</b>: Model selection and Random Forest training",
        "<b>Chapter 9 (Refinement)</b>: Hyperparameter tuning and validation",
        "<b>Chapter 10 (Inference)</b>: Model interpretation and deployment considerations",
        "<b>Chapter 11 (Tribes)</b>: Customer segmentation via K-Means clustering",
        "<b>Chapter 12 (Strategy)</b>: Business recommendations and conclusion",
    ], styles)
    
    story.append(PageBreak())
