"""
Content module for CLV Magnum Opus App
All detailed text content from PDF chapters
"""

# Chapter 1 Content
CH1_INTRO = """
In the labyrinthine world of insurance analytics, there exists a fundamental question that has 
plagued actuaries, data scientists, and business strategists for decades: **How do we peer into 
the future and predict the lifetime value of a customer who walks through our doors today?**

This question is not merely academic—it sits at the intersection of survival mathematics, 
behavioral economics, and the cold calculus of profit margins. The insurance industry, unlike 
retail or technology, operates on a fundamentally inverted cash flow model. Premiums are collected 
today, but claims—the true cost of the product—are paid out over years, sometimes decades.

This temporal asymmetry creates what actuaries call the **'long tail problem,'** where the profitability 
of a policy cannot be assessed until long after the customer relationship has matured or terminated. 
A customer who pays modest premiums but never claims is infinitely more valuable than one who pays 
high premiums but files catastrophic claims. The challenge is identifying which customers belong 
to which category—preferably before, not after, they've cost you money.
"""

CH1_CLV_EQUATION = """
**Customer Lifetime Value (CLV) = Present Value of Expected Future Cash Flows**

For insurance: `CLV ≈ (Premium - Expected Claims - Operating Costs) × Expected Tenure × Discount Factor`

Every word in this equation represents uncertainty that data science aims to quantify. The premium 
is contractual but subject to renewal decisions. Expected claims follow probability distributions 
fitted from historical loss data. Operating costs include acquisition, servicing, and overhead 
allocation. Expected tenure depends on churn modeling. The discount factor reflects time value of 
money and risk adjustment.
"""

CH1_OBJECTIVES = """
This project undertakes a comprehensive analytical journey through the Customer Lifetime Value 
problem using a real-world auto insurance dataset from IBM Watson Analytics. Our objectives span 
the full data science lifecycle:

- **Exploratory Data Analysis**: Forensic examination of data quality, distributions, and relationships
- **Feature Engineering**: Creation of derived variables that capture domain knowledge  
- **Predictive Modeling**: Machine learning model to predict individual CLV with R² > 0.90
- **Customer Segmentation**: Unsupervised clustering to identify distinct customer personas
- **Strategic Recommendations**: Translation of analytical insights into business actions
"""

CH1_DATASET = """
We work with the Watson Analytics Marketing Customer Value dataset, comprising **9,134** auto 
insurance customers and **24** attributes capturing demographic, behavioral, and policy information. 
This is production-quality data that reflects the messiness and richness of real customer records.

The code below represents our first contact with the data—a moment of discovery that every data 
scientist knows intimately. The pd.read_csv function from the pandas library ingests the comma-separated 
file and constructs a DataFrame object, which is Python's representation of a tabular dataset.
"""

CH1_CODE1 = """import pandas as pd
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
# Target Variable: Customer Lifetime Value"""

CH1_TARGET = """
Our target variable—Customer Lifetime Value—is a continuous, positive variable representing 
the estimated total value a customer contributes over their relationship with the insurer. 
Understanding its distribution is critical before any modeling attempt.

The CLV distribution displays classic right-skew: the mean ($8,005) exceeds the median ($5,780) 
by 38%, pulled upward by the long right tail. The bulk of customers (approximately 75%) have CLV 
below $10,000, while a small cohort extends to $83,325. This skewness has modeling implications: 
ordinary least squares regression will be influenced by high-value outliers, potentially suggesting 
log-transformation of the target or use of robust regression methods.

**The Pareto Effect**: Customer value follows a Pareto-like distribution. The top 20% of customers 
by CLV contribute approximately 48% of total portfolio value. This concentration makes identifying 
and retaining high-value customers strategically critical.
"""

# Chapter 2 Content
CH2_INTRO = """
In the practice of data science, the adage **'garbage in, garbage out'** is not merely a cliché—it 
is a fundamental law. No algorithm, however sophisticated, can compensate for flawed input data. 
Before we can trust any analysis or model, we must subject our dataset to rigorous quality assessment.

Data quality encompasses several dimensions:
- **Completeness**: Are there missing values?
- **Accuracy**: Are values within expected ranges?
- **Consistency**: Do related fields agree?
- **Uniqueness**: Are there duplicate records?
- **Timeliness**: Is the data current?
"""

CH2_MISSING = """
The `isnull()` method returns a boolean mask indicating True for missing values. Remarkably, our 
dataset contains **zero missing values**—a rare gift in real-world data. This cleanliness suggests 
pre-processing by the data vendor, likely through imputation or row deletion. While convenient, 
we remain alert to the possibility that realistic patterns of missingness have been artificially 
smoothed away.
"""

CH2_CODE1 = """# Check for missing values across all columns
missing_counts = df.isnull().sum()
print("Missing Values by Column:")
print(missing_counts[missing_counts > 0])

# Result: Empty Series - No missing values detected
# This is unusual for real-world data and suggests
# pre-cleaning by the data vendor

print(f"\\nTotal records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")"""

CH2_DATE = """
Our first substantive challenge emerges when we examine the 'Effective To Date' column. This field 
arrives as a string in the format 'M/D/YY' (for example, '2/18/11'), which Python and pandas cannot 
natively interpret as a date. To perform any temporal analysis—calculating tenure, identifying 
seasonality, or tracking policy cycles—we must convert these strings to proper datetime objects.

The `pd.to_datetime` function parses strings according to the format '%m/%d/%y' (month/day/two-digit-year). 
This conversion enables extraction of temporal components: month (for seasonality analysis) and day 
of week (for operational patterns).
"""

CH2_CODE2 = """# Convert date column to datetime
df['Effective To Date'] = pd.to_datetime(
    df['Effective To Date'], 
    format='%m/%d/%y'
)

# Verify the conversion
print(df['Effective To Date'].dtype)
# Output: datetime64[ns]

# Extract useful temporal features
df['Eff_Month'] = df['Effective To Date'].dt.month
df['Eff_DayOfWeek'] = df['Effective To Date'].dt.dayofweek

# Analyze temporal distribution
print("\\nPolicy dates range:")
print(f"  Earliest: {df['Effective To Date'].min()}")
print(f"  Latest:   {df['Effective To Date'].max()}")"""

CH2_ZERO = """
A more subtle issue lurks in the numeric columns. Visual inspection of the Income distribution 
reveals a striking anomaly: a massive spike at exactly zero. While zero income is technically 
possible—students, retirees living on savings, or individuals with unreported income—the magnitude 
of this spike demands scrutiny.

**The 'Bleeding Neck' Pattern**: The histogram reveals the 'bleeding neck' pattern—a technical term 
for zero-inflated distributions. The spike at zero (N=2,284) represents 25% of all customers. The 
remaining distribution is right-skewed, with a long tail extending to $99,960.

**Decision**: We retain zero-income customers rather than imputing values. The zero-income cohort 
may represent a meaningful segment with distinct behaviors. Imputation risks introducing artificial 
patterns. However, we create an 'Income_IsZero' binary feature for models.
"""

CH2_CODE3 = """# Investigate zero-income customers
zero_income = df[df['Income'] == 0]
print(f"Zero-income customers: {len(zero_income):,}")
print(f"Percentage: {len(zero_income)/len(df)*100:.1f}%")

# Profile zero-income customers by employment status
print("\\nEmployment breakdown of zero-income:")
print(zero_income['EmploymentStatus'].value_counts())

# Output:
# Unemployed     1,247  (54.6%)
# Retired          612  (26.8%)
# Employed         425  (18.6%)  <- Unusual, investigate"""

CH2_CATEGORICAL = """
Categorical variables introduce their own quality challenges. String values are susceptible to 
inconsistency: leading/trailing whitespace, inconsistent capitalization, typographical errors, 
and semantic duplicates (e.g., 'N/A' vs. 'NA' vs. 'Not Available'). We systematically cleanse 
these fields to ensure consistency.

All 15 categorical columns have low-to-medium cardinality, making one-hot encoding feasible 
without dimensionality explosion. The highest cardinality is Vehicle Class with 6 unique values.
"""

CH2_CORRELATION = """
With our data cleansed, we turn to understanding relationships between variables. The correlation 
matrix provides a bird's-eye view of pairwise linear associations. For numeric variables, we 
compute Pearson correlation coefficients; for categorical relationships, we employ ANOVA F-tests.

**Key Correlations Discovered**:
- Premium ↔ CLV: r = 0.87 (Strong positive - premium drives value)
- Claims ↔ CLV: r = 0.50 (Moderate positive - longer tenure = more claims)
- Income ↔ CLV: r = 0.15 (Weak positive - income not a direct driver)
- Premium ↔ Claims: r = 0.64 (Moderate - multicollinearity concern)

**Multicollinearity Alert**: Monthly Premium Auto and Total Claim Amount show r=0.64 correlation. 
While this doesn't bias predictions, it destabilizes coefficient estimates in linear models.
"""

# Chapter 3-4 Content
CH3_INTRO = """
With data quality established, we now explore the statistical landscape of our features. 
Univariate analysis examines each variable in isolation—its distribution shape, central tendency, 
dispersion, and outliers—before examining relationships.

Understanding distributions guides preprocessing decisions: skewed variables may need transformation, 
zero-inflated variables need special handling, and multimodal distributions suggest latent subgroups.
"""

CH4_INTRO = """
Univariate analysis tells us about individual variables; bivariate analysis reveals relationships 
between pairs. The key question: which features predict Customer Lifetime Value?

We employ Pearson correlation for numeric-numeric pairs, ANOVA F-tests for categorical-numeric pairs, 
and chi-square tests for categorical-categorical pairs. This systematic approach ensures we don't 
miss any predictive signal hiding in the data.
"""

CH4_PREMIUM = """
The scatter plot of Monthly Premium Auto vs. CLV reveals the strongest relationship in the dataset 
(r = 0.82). This is mechanically intuitive: customers who pay higher premiums generate more revenue, 
which flows directly into CLV calculations. However, premiums also reflect risk assessment—higher-risk 
customers are charged more to compensate for expected claims.

The relationship is substantially linear with some heteroscedasticity (variance increasing with 
premium level). High-premium customers show more CLV dispersion, reflecting varying claim histories.
"""

# Chapters 5-7 Content
CH5_INTRO = """
Real predictive power often emerges from **feature interactions**—combinations of variables that 
reveal patterns invisible when examined individually. A customer's value depends not just on their 
premium or tenure, but on the combination: a high-premium customer with short tenure is very 
different from a high-premium customer with long tenure.
"""

CH6_INTRO = """
Raw features rarely perform optimally in machine learning models. **Feature engineering** is the art 
and science of transforming raw data into representations that better capture the underlying patterns 
we seek to model.

Transformations serve multiple purposes:
- **Normalization**: Scale features to comparable ranges
- **Skew reduction**: Make distributions more symmetric
- **Interaction capture**: Create compound features encoding relationships
- **Domain encoding**: Inject business knowledge into features
"""

CH6_TRANSFORMS = """
**Transformations Applied**:

| Transformation | Purpose | Formula |
|---------------|---------|---------|
| Log Transform | Reduce right-skew in CLV | `np.log1p(clv)` |
| Yeo-Johnson | Normalize all numeric distributions | `PowerTransformer()` |
| Standard Scaling | Zero mean, unit variance | `StandardScaler()` |
| One-Hot Encoding | Categorical → Binary columns | `pd.get_dummies()` |
"""

CH7_FEATURES = """
Beyond transformations, we create **new features** that capture domain knowledge:

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| Loss Ratio | `claims / (premium × 12)` | Profitability indicator |
| Premium-to-Income | `(premium × 12 / income) × 100` | Affordability stress |
| Tenure Buckets | `New/Growing/Mature` | Customer lifecycle stage |
| Policy Density | `policies / tenure` | Cross-sell velocity |
| Claim Recency | `months_since_last_claim` | Recent activity indicator |

**Impact**: Engineered features improved model R² from 0.72 to 0.85—an 18% relative improvement, 
demonstrating the value of domain-driven feature creation.
"""

CH7_CODE = """# Create engineered features
df['loss_ratio'] = df['Total Claim Amount'] / (df['Monthly Premium Auto'] * 12 + 1)
df['premium_to_income'] = (df['Monthly Premium Auto'] * 12) / (df['Income'] + 1) * 100
df['tenure_bucket'] = pd.cut(
    df['Months Since Policy Inception'], 
    bins=[0, 6, 24, 120], 
    labels=['New', 'Growing', 'Mature']
)
df['policy_density'] = df['Number of Policies'] / (df['Months Since Policy Inception'] + 1)

# Verify engineered features
print("Engineered Feature Statistics:")
print(df[['loss_ratio', 'premium_to_income', 'policy_density']].describe())"""

# Chapter 8-10 Content
CH8_INTRO = """
With features engineered and data prepared, we now enter the modeling phase. Our goal: build a 
predictive model that accurately estimates Customer Lifetime Value for new customers, enabling 
informed acquisition and retention decisions.

We evaluate five regression algorithms using rigorous cross-validation, seeking the best balance 
of predictive accuracy, interpretability, and computational efficiency.
"""

CH8_MODELS = """
**Algorithm Selection Rationale**:

1. **Linear Regression**: Baseline for comparison; assumes linear relationships
2. **Ridge Regression**: L2 regularization to handle multicollinearity
3. **Random Forest**: Ensemble of decision trees; captures non-linear patterns
4. **Gradient Boosting**: Sequential ensemble; often state-of-the-art
5. **XGBoost**: Optimized gradient boosting; handles missing values natively

Each model was evaluated using 5-fold stratified cross-validation, with stratification based on 
CLV quartiles to ensure each fold has representative value distribution.
"""

CH8_CODE = """from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

# Define models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Evaluate each model with 5-fold CV
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name}: R² = {scores.mean():.3f} ± {scores.std():.3f}")
    
# Output:
# Linear: R² = 0.756 ± 0.018
# Ridge: R² = 0.761 ± 0.016
# RandomForest: R² = 0.891 ± 0.023
# GradientBoosting: R² = 0.878 ± 0.021
# XGBoost: R² = 0.885 ± 0.019"""

CH9_TUNING = """
With Random Forest identified as the best performer, we fine-tune its hyperparameters using 
GridSearchCV. The key parameters controlling Random Forest behavior are:

- **n_estimators**: Number of trees in the forest (more = better, with diminishing returns)
- **max_depth**: Maximum tree depth (controls complexity vs. overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in leaf nodes
- **max_features**: Number of features considered for each split
"""

CH9_CODE = """# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.3f}")

# Best parameters found:
# n_estimators: 200
# max_depth: 15
# min_samples_split: 5
# min_samples_leaf: 2
# Best CV R²: 0.891"""

CH10_IMPORTANCE = """
Random Forest provides built-in feature importance scores based on the decrease in impurity 
(Gini or entropy) each feature contributes across all trees. Higher importance means the feature 
is more useful for making accurate predictions.

**Top Feature Importances**:
1. Monthly Premium Auto: 42% — Dominates prediction
2. Number of Policies: 18% — Cross-selling matters
3. Income: 12% — Financial capacity signal
4. Months Since Policy Inception: 10% — Loyalty indicator
5. Total Claim Amount: 9% — Risk profile
6. Vehicle Class: 5% — Product type effect
7. Coverage: 4% — Policy tier effect
"""

CH10_INTERPRET = """
**Business Interpretation**:

The model confirms what intuition suggests but quantifies it precisely:

1. **Premium is paramount**: A $10/month premium increase predicts ~$500 higher CLV
2. **Cross-sell pays**: Each additional policy adds approximately $1,200 to predicted CLV
3. **Tenure builds value**: Each year of tenure adds ~$400 to predicted CLV
4. **Claims are costly**: High loss ratios reduce predicted CLV by up to 30%
5. **Demographics matter less**: Gender, education, marital status have <3% importance combined
"""

# Chapter 11 Content
CH11_INTRO = """
Prediction answers 'how much'—how much is this customer worth? Segmentation answers 'how 
different'—what distinct types of customers populate our portfolio? While prediction enables 
prioritization, segmentation enables personalization.

Different customer types require different communication styles, different product offerings, 
different service levels. A one-size-fits-all approach wastes resources on mismatched customers. 
Segmentation reveals the hidden tribes within our customer base, enabling tailored strategies for each.

**Why Segment?** With 9,134 customers in our portfolio, personalized attention for each is impossible. 
Segmentation groups similar customers together, allowing us to craft 4-5 distinct strategies instead 
of 9,134 individual plans—a practical compression of complexity that preserves most of the 
personalization value.
"""

CH11_KMEANS = """
**K-Means: A Mathematical Deep Dive**

K-Means is the workhorse of customer segmentation. Given K (the number of clusters), K-Means 
iteratively assigns each customer to the nearest cluster centroid, then recomputes centroids as 
the mean of all assigned customers.

**The Objective Function**:
K-Means minimizes Within-Cluster Sum of Squares (WCSS): `WCSS = Σₖ Σₓ∈Cₖ ||x - μₖ||²`

Where x is a customer's feature vector and μₖ is the centroid of cluster k. Lower WCSS indicates 
tighter, more cohesive clusters.

**Algorithm Steps**:
1. **Initialization**: Select K customers as initial centroids (using k-means++)
2. **Assignment**: Assign each customer to nearest centroid (Euclidean distance)
3. **Update**: Recalculate each centroid as mean of assigned customers
4. **Convergence**: Repeat until centroids stabilize
"""

CH11_CODE = """from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Select features for clustering
cluster_features = [
    'Income',                          # Financial capacity
    'Monthly Premium Auto',            # Product engagement
    'Total Claim Amount',              # Risk profile
    'Months Since Policy Inception',   # Tenure/loyalty
    'Number of Policies'               # Cross-sell depth
]

# Standardize features (critical for K-Means)
X_cluster = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal K using elbow method and silhouette
inertias, silhouettes = [], []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:,.0f}, Silhouette={silhouettes[-1]:.3f}")

# K=4 selected: Silhouette=0.378 (peak), clear elbow"""

CH11_SILHOUETTE = """
**Silhouette Analysis**

The silhouette coefficient measures how similar a customer is to their own cluster compared to 
other clusters. For each customer i:
- a(i) = mean distance to other customers in same cluster
- b(i) = mean distance to customers in nearest different cluster

`s(i) = [b(i) - a(i)] / max{a(i), b(i)}`

| Silhouette Range | Interpretation | Action |
|-----------------|----------------|--------|
| 0.71 - 1.00 | Strong cluster structure | Confident segmentation |
| 0.51 - 0.70 | Reasonable structure | Good for business use |
| 0.26 - 0.50 | Weak structure, overlap | Use with caution |
| < 0.25 | No substantial structure | Reconsider approach |

**Our Result**: Silhouette = 0.378 for K=4 indicates 'weak but reasonable' structure. This is 
typical for customer segmentation—real customers don't fall into perfectly discrete boxes.
"""

# Chapter 12 Content
CH12_INTRO = """
Our analytical journey culminates in actionable recommendations. Data science without business 
impact is an academic exercise. This chapter translates our statistical findings into strategic 
guidance for marketing, underwriting, and customer service operations.
"""

CH12_RECOMMENDATIONS = """
**Revenue Growth Strategies**:

1. **Protect High Rollers** (Critical Priority)
   - Assign dedicated account managers to top 500 customers
   - Implement proactive service calls at policy renewal
   - Estimated impact: Reducing churn by 5% saves $1.2M annually

2. **Cross-Sell to Steady Eddies** (High ROI)
   - Each additional policy adds $1,200 to CLV
   - 31% of customer base (2,832 customers) with moderate cross-sell propensity
   - Target: 23% success rate = 651 new policies = $781K incremental CLV

3. **Optimize Acquisition** (Efficiency Play)
   - Use CLV predictions to cap acquisition spending
   - Predicted CLV $5,000 → max acquisition cost $500 (10% guideline)
   - Focus marketing on high-CLV-lookalike prospects

**Risk Mitigation Strategies**:

1. **Early Riskmaker Detection**
   - Flag customers matching Riskmaker profile at acquisition
   - Implement enhanced underwriting review
   - Consider premium adjustments or coverage limitations

2. **Fresh Start Monitoring**
   - 90-day check-in program for new customers
   - Early intervention for satisfaction issues
   - Graduation track to Steady Eddie with loyalty benefits

3. **Claims Verification**
   - Enhanced scrutiny for loss ratio > 0.8
   - Fraud detection algorithm integration
   - Expected claims reduction: 8%
"""

CH12_IMPACT = """
**Projected Business Impact**:

| Initiative | Investment | Expected Return | ROI |
|-----------|------------|-----------------|-----|
| High Roller Retention | $200K | $1.2M saved | 500% |
| Cross-Sell Campaign | $150K | $781K CLV | 420% |
| Acquisition Optimization | $100K | $500K saved | 400% |
| Claims Verification | $75K | $300K savings | 300% |
| **Total** | **$525K** | **$2.78M** | **430%** |

The combined initiatives could increase portfolio value by 15-20% over 24 months while 
reducing acquisition costs and claims leakage.
"""

# Cluster detailed profiles
CLUSTERS = {
    0: {
        'name': 'Steady Eddies', 'emoji': '🏠', 'color': '#4A90D9', 
        'pct': 31, 'count': 2832, 'clv': 7234, 'income': 42000, 
        'premium': 78, 'tenure': 52, 'loss': 0.43, 'policies': 2.1,
        'profile': """The Steady Eddies represent 31% of our customer base (2,832 customers). They exhibit 
moderate income ($42,000 average), moderate premiums ($78/month), and remarkably stable loss ratios 
(0.43 mean). With an average tenure of 52 months, these are mature, established relationships.

**Behavioral Profile**: These customers are the 'bread and butter' of the portfolio. They don't demand 
attention, don't file excessive claims, and don't require special handling. Their payment patterns 
are consistent, their expectations are reasonable, and their loyalty is passive rather than emotional. 
They stay because switching costs outweigh perceived benefits, not because of brand affinity.""",
        'strategy': """**Strategy: Low-Touch Efficiency**
- Automate service touchpoints (digital self-service, chatbots)
- Annual loyalty bonus ($25-50 credit) to reinforce retention
- Consider gradual premium increases (2-3% annually)
- Cross-sell opportunity: moderate (23% predicted success rate)
- Do NOT over-invest in retention—they're not at risk"""
    },
    1: {
        'name': 'High Rollers', 'emoji': '💎', 'color': '#2ECC71',
        'pct': 18, 'count': 1644, 'clv': 14892, 'income': 72000,
        'premium': 142, 'tenure': 71, 'loss': 0.39, 'policies': 3.4,
        'profile': """The High Rollers are the VIP segment—just 18% of customers (1,644 individuals) but 
commanding the highest average income ($72,000), highest premiums ($142/month), and longest tenure 
(71 months). Their loss ratio of 0.39 is below portfolio average, indicating profitable risk profiles.

**Behavioral Profile**: High Rollers are sophisticated consumers who expect premium service. They likely 
own multiple vehicles (including luxury models requiring higher coverage), have complex insurance needs, 
and value convenience over price. They're less price-sensitive but highly service-sensitive. A single 
bad claims experience can trigger defection—and these are the customers you cannot afford to lose.

**Value Concentration**: While representing only 18% of customers, High Rollers generate approximately 
33% of total portfolio CLV (estimated at $24.5M in aggregate).""",
        'strategy': """**Strategy: White-Glove Service**
- Assign dedicated account managers for top 500 customers
- Priority claims processing (24-hour resolution target)
- Quarterly relationship reviews (proactive, not reactive)
- Exclusive offers: umbrella policies, collectors vehicle coverage
- Cross-sell aggressively (41% predicted success rate)
- Monitor satisfaction scores weekly—intervene immediately on any decline"""
    },
    2: {
        'name': 'Riskmakers', 'emoji': '⚠️', 'color': '#E74C3C',
        'pct': 29, 'count': 2649, 'clv': 5621, 'income': 38000,
        'premium': 83, 'tenure': 28, 'loss': 0.68, 'policies': 1.3,
        'profile': """The Riskmakers represent 29% of the portfolio (2,649 customers). They have moderate 
income ($38,000), moderate premiums ($83/month), but notably elevated loss ratios (0.68—45% above 
portfolio average). Tenure averages only 28 months.

**Behavioral Profile**: These customers claim more than they contribute proportionally. The elevated 
loss ratio suggests either adverse selection (they joined because they anticipated needing to claim) 
or behavioral risk factors (reckless driving, poor vehicle maintenance). Their short tenure may 
indicate they're rate shoppers who move when premiums increase, or that they've been non-renewed 
by previous insurers.

**Profitability Warning**: With a loss ratio of 0.68, these customers consume $0.68 in claims for 
every $1.00 of premium collected. After operating expenses (~25% of premium), the margin is razor-thin 
or negative. Approximately 12% of this cluster have loss ratios exceeding 1.0—they are definitively 
unprofitable.""",
        'strategy': """**Strategy: Risk Mitigation**
- Enhanced underwriting review at renewal
- Premium adjustments reflecting true risk (may trigger churn—acceptable)
- Claims verification protocols for suspicious patterns
- Do NOT cross-sell or offer discounts
- Consider non-renewal for worst performers (loss ratio > 1.2)
- Monitor for fraud indicators"""
    },
    3: {
        'name': 'Fresh Starts', 'emoji': '🌱', 'color': '#F39C12',
        'pct': 22, 'count': 2009, 'clv': 6487, 'income': 55000,
        'premium': 91, 'tenure': 11, 'loss': 0.52, 'policies': 1.8,
        'profile': """Fresh Starts represent 22% of the portfolio (2,009 customers). They have above-average 
income ($55,000) and premiums ($91/month), but very short tenure (11 months average). Loss ratios 
are moderate (0.52).

**Behavioral Profile**: These are new customers whose trajectory is uncertain. They have the financial 
profile of potentially valuable customers but haven't yet demonstrated loyalty or stable behavior. 
The first 12-24 months are critical: they will either mature into Steady Eddies (or even High Rollers) 
or churn to competitors.

**Critical Window**: Industry data shows 40% of insurance churn occurs in the first 18 months. Fresh 
Starts are in this danger zone. Proactive engagement now determines whether they become long-term 
assets or expensive acquisition failures.""",
        'strategy': """**Strategy: Early Engagement**
- Welcome call within 7 days of policy activation
- 30-day check-in email with self-service resources
- 90-day satisfaction survey with follow-up on any concerns
- First renewal offer: 5% loyalty discount if claims-free
- Graduation program: clear path to 'preferred customer' status
- Flag for immediate attention if complaint filed or payment missed"""
    }
}

# Hypothesis tests
HYPOTHESIS_TESTS = [
    {"h": "H1: Higher premiums → Higher CLV", "stat": "Pearson r = 0.82", "p": "p < 0.001", 
     "result": "✅ Confirmed", "detail": "Strong linear relationship; $10 premium increase predicts $500 CLV increase"},
    {"h": "H2: Coverage type affects CLV", "stat": "F = 45.2 (ANOVA)", "p": "p < 0.001",
     "result": "✅ Confirmed", "detail": "Premium coverage: $9,247 mean CLV; Basic: $5,890 mean CLV"},
    {"h": "H3: Multi-policy customers have higher CLV", "stat": "t = 8.7 (t-test)", "p": "p < 0.001",
     "result": "✅ Confirmed", "detail": "Each additional policy adds ~$1,200 to predicted CLV"},
    {"h": "H4: Longer tenure → Higher CLV", "stat": "Pearson r = 0.31", "p": "p < 0.001",
     "result": "✅ Confirmed", "detail": "Each year of tenure adds ~$400 to CLV; loyalty matters"},
    {"h": "H5: High claims reduce CLV", "stat": "Partial r = -0.12", "p": "p = 0.003",
     "result": "⚠️ Weak effect", "detail": "Claims hurt value, but effect smaller than expected due to premium offset"},
]
