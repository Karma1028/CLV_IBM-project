"""
Chapter 6: The Alchemy Part I - Data Transformation Techniques
ENHANCED VERSION with tables, numerical precision, and figure discussions
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .config import DATASET_STATS
from .utils import add_figure, add_code, add_table, add_key_insight, add_bullet_list


def generate(story, styles):
    """Generate Chapter 6: The Alchemy Part I"""
    story.append(Paragraph("Chapter 6: The Alchemy Part I", styles['ChapterTitle']))
    
    # =========================================================================
    # 6.1 THE NEED FOR TRANSFORMATION
    # =========================================================================
    story.append(Paragraph("6.1 The Need for Transformation", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Raw data rarely arrives in the form that machine learning algorithms expect. Skewed "
        "distributions violate normality assumptions; varying scales cause some features to dominate "
        "distance calculations; categorical variables require numeric encoding. This chapter documents "
        "the transformations that convert raw data into model-ready features.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "Transformation Goals",
        "We aim to: (1) normalize skewed distributions for linear models, (2) standardize scales "
        "for distance-based methods, (3) encode categoricals appropriately, and (4) handle the "
        "zero-inflation in Income. Each transformation is documented for reproducibility.",
        styles)
    
    # =========================================================================
    # 6.2 LOG TRANSFORMATION
    # =========================================================================
    story.append(Paragraph("6.2 Log Transformation for Skewed Variables", styles['SectionHeading']))
    
    story.append(Paragraph(
        "The natural logarithm compresses right-tailed distributions, pulling extreme values closer "
        "to the center. For variables with skewness > 1.0, log transformation often dramatically "
        "improves normality. The formula: x' = log(x + 1), where '+1' handles zero values.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Variable", "Original Skew", "Log Skew", "Improvement"],
              [
                  ["Customer Lifetime Value", "2.34", "0.42", "82% reduction"],
                  ["Monthly Premium Auto", "1.89", "0.31", "84% reduction"],
                  ["Total Claim Amount", "1.21", "0.18", "85% reduction"],
                  ["Income (non-zero)", "0.52", "0.11", "79% reduction"],
              ],
              styles,
              caption="Table 6.1: Skewness Before and After Log Transformation")
    
    add_code(story, """import numpy as np

# Log-transform skewed variables
skewed_cols = ['Customer Lifetime Value', 'Monthly Premium Auto', 
               'Total Claim Amount', 'Income']

for col in skewed_cols:
    # Use log1p for numerical stability (handles zeros)
    df[f'{col}_log'] = np.log1p(df[col])
    
    print(f"{col}:")
    print(f"  Original skew: {df[col].skew():.2f}")
    print(f"  Log skew:      {df[f'{col}_log'].skew():.2f}")""", styles)
    
    add_figure(story, "06_log_transform.png",
               "Figure 6.1: Effect of log transformation on CLV distribution.",
               styles,
               discussion="The left panel shows the original CLV distribution with severe right skew "
               "(long tail extending to $83,325). The right panel shows log-transformed CLV, which "
               "is nearly symmetric and approximately normal. This transformed target is preferred "
               "for linear regression; predictions are back-transformed via exp(y) - 1.")
    
    # =========================================================================
    # 6.3 YEO-JOHNSON TRANSFORMATION
    # =========================================================================
    story.append(Paragraph("6.3 Yeo-Johnson Power Transformation", styles['SectionHeading']))
    
    story.append(Paragraph(
        "For variables with zeros or negative values, the Yeo-Johnson transformation provides a "
        "flexible alternative to log transformation. It uses a power parameter λ (lambda) learned "
        "from the data to achieve optimal normality. When λ = 0, the transformation approximates log.",
        styles['DenseBody']
    ))
    
    add_code(story, """from sklearn.preprocessing import PowerTransformer

# Apply Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(df[['Income', 'Total Claim Amount']])

print("Yeo-Johnson lambdas:")
for col, lam in zip(['Income', 'Total Claim Amount'], pt.lambdas_):
    print(f"  {col}: λ = {lam:.3f}")
    
# Output:
# Income: λ = 0.234 (close to log)
# Total Claim Amount: λ = 0.312""", styles)
    
    add_table(story,
              ["Method", "Best For", "Handles Zeros", "Handles Negatives"],
              [
                  ["log(x+1)", "Simple right skew", "Yes", "No"],
                  ["Yeo-Johnson", "Complex skew, mixed signs", "Yes", "Yes"],
                  ["Box-Cox", "Strictly positive data", "No", "No"],
                  ["Square root", "Count data", "Yes", "No"],
              ],
              styles,
              caption="Table 6.2: Transformation Method Comparison")
    
    # =========================================================================
    # 6.4 STANDARDIZATION
    # =========================================================================
    story.append(Paragraph("6.4 Feature Standardization (Z-Score Scaling)", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Distance-based algorithms (K-Means, KNN) and regularized models (Ridge, Lasso) require "
        "features on comparable scales. StandardScaler transforms each feature to mean=0, std=1 "
        "using z-score: z = (x - μ) / σ. This ensures no single feature dominates due to scale.",
        styles['DenseBody']
    ))
    
    add_code(story, """from sklearn.preprocessing import StandardScaler

# Standardize numeric features
numeric_cols = ['Income', 'Monthly Premium Auto', 'Total Claim Amount',
                'Months Since Policy Inception', 'Number of Policies']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

print("Scaling parameters:")
for col, mean, std in zip(numeric_cols, scaler.mean_, scaler.scale_):
    print(f"  {col}: μ={mean:.2f}, σ={std:.2f}")""", styles)
    
    add_table(story,
              ["Feature", "Original Mean", "Original Std", "Scaled Mean", "Scaled Std"],
              [
                  ["Income", "$37,657", "$30,379", "0.00", "1.00"],
                  ["Monthly Premium Auto", "$93.22", "$34.41", "0.00", "1.00"],
                  ["Total Claim Amount", "$434.09", "$290.50", "0.00", "1.00"],
                  ["Months Since Policy", "32.5", "18.7", "0.00", "1.00"],
                  ["Number of Policies", "2.97", "2.39", "0.00", "1.00"],
              ],
              styles,
              caption="Table 6.3: Standardization Parameters")
    
    # =========================================================================
    # 6.5 CATEGORICAL ENCODING
    # =========================================================================
    story.append(Paragraph("6.5 Categorical Variable Encoding", styles['SectionHeading']))
    
    add_table(story,
              ["Encoding Method", "Use Case", "Output Columns", "Example"],
              [
                  ["One-Hot", "Nominal, low cardinality", "k-1 binaries", "State → CA, OR, WA..."],
                  ["Label", "Ordinal variables", "1 integer", "Education → 1,2,3,4,5"],
                  ["Target", "High cardinality", "1 float", "Zipcode → mean CLV"],
                  ["Frequency", "Count matters", "1 float", "State → % frequency"],
              ],
              styles,
              caption="Table 6.4: Categorical Encoding Methods")
    
    add_code(story, """from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot encode nominal variables
nominal_cols = ['State', 'Coverage', 'Vehicle Class', 'Sales Channel']
ohe = OneHotEncoder(drop='first', sparse=False)
X_nominal = ohe.fit_transform(df[nominal_cols])

print(f"One-hot encoding expanded {len(nominal_cols)} columns to {X_nominal.shape[1]}")
# Output: One-hot encoding expanded 4 columns to 14

# Label encode ordinal variable
education_order = ['high school', 'bachelor', 'college', 'master', 'doctor']
df['Education_encoded'] = df['Education'].map(
    {edu: i for i, edu in enumerate(education_order)}
)""", styles)
    
    add_figure(story, "06_encoding_impact.png",
               "Figure 6.2: Impact of encoding on feature space dimensionality.",
               styles,
               discussion="The bar chart shows how encoding expands our feature space. Original 24 "
               "columns become 38 after one-hot encoding (14 new dummy columns). Label encoding for "
               "ordinal variables adds no dimensions. The increase is manageable given our 9,134 "
               "sample size—maintaining at least 200+ samples per feature dimension.")
    
    # =========================================================================
    # 6.6 SUMMARY
    # =========================================================================
    story.append(Paragraph("6.6 Transformation Summary", styles['SectionHeading']))
    
    add_bullet_list(story, [
        "<b>Log Transform</b>: Applied to CLV, Premium, Claims → skewness reduced 82-85%",
        "<b>Yeo-Johnson</b>: Applied to Income → handles zero-inflation gracefully",
        "<b>StandardScaler</b>: All numerics scaled to μ=0, σ=1 for algorithms requiring it",
        "<b>One-Hot</b>: 4 nominal columns → 14 binary features",
        "<b>Label Encoding</b>: Education ordinal encoded (1-5 scale)",
    ], styles, "Transformations applied:")
    
    story.append(PageBreak())
