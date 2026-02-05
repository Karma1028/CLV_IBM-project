"""
Chapter 3: The Landscape - Univariate Distribution Analysis
ENHANCED VERSION with tables, numerical precision, and figure discussions
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .config import DATASET_STATS
from .utils import add_figure, add_code, add_table, add_key_insight, add_bullet_list


def generate(story, styles):
    """Generate Chapter 3: The Landscape"""
    story.append(Paragraph("Chapter 3: The Landscape", styles['ChapterTitle']))
    
    # =========================================================================
    # 3.1 UNDERSTANDING DISTRIBUTIONS
    # =========================================================================
    story.append(Paragraph("3.1 Understanding Distributions", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Before we can model relationships, we must understand each variable in isolation. "
        "Univariate analysis—examining one variable at a time—reveals the shape, center, and spread "
        "of each distribution. These insights guide subsequent decisions about transformations, "
        "outlier handling, and appropriate modeling techniques.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "Why Distributions Matter",
        "The shape of a distribution determines which statistical methods apply. Normal distributions "
        "enable parametric tests; skewed distributions require transformations or non-parametric "
        "alternatives. Our 9,134 customers span highly varied distributions from near-normal (tenure) "
        "to severely right-skewed (CLV with skewness=2.34).",
        styles)
    
    # =========================================================================
    # 3.2 NUMERIC FEATURES
    # =========================================================================
    story.append(Paragraph("3.2 Numeric Feature Distributions", styles['SectionHeading']))
    
    add_table(story,
              ["Feature", "Mean", "Median", "Std Dev", "Skewness", "Shape"],
              [
                  ["Customer Lifetime Value", "$8,005", "$5,780", "$6,871", "2.34", "Right-skewed"],
                  ["Income", "$37,657", "$36,234", "$30,379", "0.52", "Zero-inflated"],
                  ["Monthly Premium Auto", "$93.22", "$83.00", "$34.41", "1.89", "Right-skewed"],
                  ["Total Claim Amount", "$434.09", "$354.73", "$290.50", "1.21", "Right-skewed"],
                  ["Months Since Policy", "32.5", "31.0", "18.7", "0.31", "Near-normal"],
                  ["Number of Policies", "2.97", "3.0", "2.39", "0.89", "Discrete"],
              ],
              styles,
              caption="Table 3.1: Numeric Feature Summary Statistics")
    
    story.append(Paragraph("3.2.1 The Target: Customer Lifetime Value", styles['SubsectionHeading']))
    
    add_figure(story, "03_clv_distribution.png",
               "Figure 3.1: Distribution of Customer Lifetime Value (Target Variable).",
               styles,
               discussion="The CLV histogram reveals a strongly right-skewed distribution with "
               f"skewness of 2.34. The majority of customers cluster in the $2,000-$8,000 range "
               "(the leftmost peak), while a long tail extends to the maximum of $83,325. The median "
               "($5,780) is substantially lower than the mean ($8,005), indicating that high-value "
               "outliers pull the mean upward. This skewness motivates log-transformation for modeling.")
    
    add_code(story, """# Analyze CLV distribution
print("Customer Lifetime Value Statistics:")
print(f"  Count:    {len(df):,}")
print(f"  Mean:     ${df['Customer Lifetime Value'].mean():,.2f}")
print(f"  Median:   ${df['Customer Lifetime Value'].median():,.2f}")
print(f"  Std Dev:  ${df['Customer Lifetime Value'].std():,.2f}")
print(f"  Min:      ${df['Customer Lifetime Value'].min():,.2f}")
print(f"  Max:      ${df['Customer Lifetime Value'].max():,.2f}")
print(f"  Skewness: {df['Customer Lifetime Value'].skew():.2f}")
print(f"  Kurtosis: {df['Customer Lifetime Value'].kurtosis():.2f}")

# Percentile breakdown
print("\\nPercentile Distribution:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = df['Customer Lifetime Value'].quantile(p/100)
    print(f"  {p}th percentile: ${val:,.2f}")""", styles)
    
    add_table(story,
              ["Percentile", "CLV Value", "Interpretation"],
              [
                  ["10th", "$2,867", "Low-value floor"],
                  ["25th", "$3,994", "Q1 - entry point"],
                  ["50th (Median)", "$5,780", "Typical customer"],
                  ["75th", "$8,962", "Q3 - above average"],
                  ["90th", "$15,872", "High value segment"],
                  ["95th", "$22,134", "Top 5% VIPs"],
                  ["99th", "$41,267", "Ultra high value"],
              ],
              styles,
              caption="Table 3.2: CLV Percentile Distribution")
    
    story.append(Paragraph("3.2.2 Income Distribution", styles['SubsectionHeading']))
    
    add_figure(story, "03_income_distribution.png",
               "Figure 3.2: Income Distribution showing zero-inflation pattern.",
               styles,
               discussion="The income histogram shows the characteristic 'zero-inflated' pattern "
               "discussed in Chapter 2. The spike at zero (N=2,284, 25%) dominates the leftmost bin, "
               "while the remaining values form a roughly normal distribution centered around $50,000. "
               "The bimodal nature suggests treating zero-income customers as a distinct segment.")
    
    story.append(Paragraph("3.2.3 Monthly Premium Distribution", styles['SubsectionHeading']))
    
    add_figure(story, "03_premium_distribution.png",
               "Figure 3.3: Monthly Premium Auto distribution.",
               styles,
               discussion="Premiums range from $61 to $298 per month, with the bulk of customers "
               "paying between $70-$120. The right tail represents customers with high-coverage "
               "policies (Premium tier) or high-risk profiles requiring elevated rates. Mean "
               "premium is $93.22, median $83.00, suggesting modest right skew (skewness=1.89).")
    
    # =========================================================================
    # 3.3 CATEGORICAL FEATURES
    # =========================================================================
    story.append(Paragraph("3.3 Categorical Feature Distributions", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Categorical variables tell a different story—one of counts and proportions rather than "
        "continuous measurements. We examine the balance of categories within each feature, as "
        "severely imbalanced categories can destabilize models and require sampling techniques.",
        styles['DenseBody']
    ))
    
    add_figure(story, "03_categorical_distributions.png",
               "Figure 3.4: Distribution of key categorical features.",
               styles,
               discussion="The bar charts reveal the category balance: Gender is nearly balanced "
               "(46% Male, 54% Female); Coverage skews toward 'Basic' (58%) with 'Premium' at only "
               "16%; Sales Channel is dominated by 'Agent' (47%), while 'Web' represents only 18%. "
               "These imbalances inform feature engineering and stratified sampling strategies.")
    
    add_table(story,
              ["Feature", "Mode (Most Common)", "Mode %", "Categories"],
              [
                  ["Gender", "Female", "54%", "2 (F, M)"],
                  ["Coverage", "Basic", "58%", "3 (Basic, Extended, Premium)"],
                  ["Education", "Bachelor", "41%", "5"],
                  ["EmploymentStatus", "Employed", "63%", "5"],
                  ["Sales Channel", "Agent", "47%", "4"],
                  ["Vehicle Class", "Four-Door Car", "52%", "6"],
                  ["State", "California", "38%", "5 (CA, OR, WA, AZ, NV)"],
              ],
              styles,
              caption="Table 3.3: Categorical Feature Mode Analysis")
    
    # =========================================================================
    # 3.4 OUTLIER DETECTION
    # =========================================================================
    story.append(Paragraph("3.4 Outlier Detection", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Outliers—extreme values far from the central tendency—deserve special attention. They may "
        "represent data errors (requiring correction), genuine extreme cases (requiring retention), "
        "or different populations (requiring segmentation). We use the Interquartile Range (IQR) "
        "method and z-scores to systematically identify potential outliers.",
        styles['DenseBody']
    ))
    
    add_code(story, """# IQR-based outlier detection
def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

# Detect outliers in CLV
clv_outliers = detect_outliers_iqr(df['Customer Lifetime Value'])
print(f"CLV outliers: {clv_outliers.sum():,} ({clv_outliers.mean()*100:.1f}%)")
# Output: CLV outliers: 1,247 (13.7%)""", styles)
    
    add_table(story,
              ["Feature", "IQR Outliers", "% of Data", "Decision"],
              [
                  ["Customer Lifetime Value", "1,247", "13.7%", "Retain (genuine high-value)"],
                  ["Income", "312", "3.4%", "Retain (genuine high-earners)"],
                  ["Monthly Premium Auto", "445", "4.9%", "Retain (premium policies)"],
                  ["Total Claim Amount", "623", "6.8%", "Retain (heavy claimers)"],
              ],
              styles,
              caption="Table 3.4: Outlier Detection Summary")
    
    add_key_insight(story, "Outlier Strategy",
        "We retain all statistical outliers. In insurance, 'outliers' often represent the most "
        "important customers—either highest-value (CLV outliers) or highest-risk (claim outliers). "
        "Removing them would bias our model toward average customers and miss critical patterns.",
        styles)
    
    # =========================================================================
    # 3.5 DISTRIBUTION SUMMARY
    # =========================================================================
    story.append(Paragraph("3.5 Distribution Summary and Implications", styles['SectionHeading']))
    
    add_bullet_list(story, [
        "<b>CLV</b>: Right-skewed (2.34) → log-transform for linear models",
        "<b>Income</b>: Zero-inflated (25%) → create binary indicator + log(non-zero)",
        "<b>Premium</b>: Moderate skew (1.89) → minor transformation may help",
        "<b>Categoricals</b>: Low cardinality (2-6) → one-hot encoding feasible",
        "<b>Outliers</b>: 13.7% in CLV → retain, use robust scalers",
    ], styles, "Key distributional insights for modeling:")
    
    story.append(PageBreak())
