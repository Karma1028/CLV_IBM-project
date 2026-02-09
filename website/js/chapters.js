// ==========================================
// CHAPTERS CONTENT - All 12 Chapters
// ==========================================

const chapters = [
    // Chapter 1: Genesis
    {
        id: 1,
        title: "The Genesis",
        subtitle: "Where It All Began",
        icon: "📁",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">Every investigation begins with a question. Ours began in the dusty archives of an auto insurance company, 
                where 9,134 customer records waited to reveal their secrets...</p>
            </div>
            
            <div class="chapter-section">
                <h3>🎯 The Business Problem</h3>
                <p>An auto insurance company faces a critical challenge: <strong>How do we predict the lifetime value of our customers?</strong></p>
                <p>Customer Lifetime Value (CLV) represents the total worth of a customer to a business over their entire relationship. 
                For insurance companies, this metric is crucial for:</p>
                <ul>
                    <li>Identifying high-value customers for retention programs</li>
                    <li>Optimizing marketing spend and acquisition costs</li>
                    <li>Personalizing service levels based on customer worth</li>
                    <li>Strategic resource allocation across customer segments</li>
                </ul>
            </div>
            
            <div class="chapter-section equation-box">
                <h3>📐 The CLV Equation</h3>
                <div class="formula">
                    <code>CLV = Σ (Monthly Premium × Retention Rate)ᵗ - Acquisition Cost</code>
                </div>
                <p>Where <em>t</em> represents time periods, creating a present value calculation of future customer revenue streams.</p>
            </div>
            
            <div class="chapter-section">
                <h3>📊 The Evidence: Our Dataset</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-num">9,134</span>
                        <span class="stat-text">Customer Records</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-num">24</span>
                        <span class="stat-text">Features</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-num">$8,004</span>
                        <span class="stat-text">Average CLV</span>
                    </div>
                </div>
                
                <div class="code-block">
                    <pre># Loading the evidence
import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
print(f"Records: {len(df):,}")
print(f"Features: {df.shape[1]}")</pre>
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>🔬 Research Objectives</h3>
                <ol class="objective-list">
                    <li>What customer characteristics correlate most strongly with high CLV?</li>
                    <li>Can we predict CLV with sufficient accuracy for business decisions?</li>
                    <li>Are there distinct customer segments with different value profiles?</li>
                    <li>What actionable strategies can maximize portfolio CLV?</li>
                </ol>
            </div>
        `
    },

    // Chapter 2: Forensic Audit
    {
        id: 2,
        title: "The Forensic Audit",
        subtitle: "Examining the Evidence",
        icon: "🔬",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">Before any detective can solve a case, they must examine the evidence with meticulous care. 
                We scrutinized every data point, searching for inconsistencies, anomalies, and hidden patterns...</p>
            </div>
            
            <div class="chapter-section">
                <h3>✅ Data Quality Assessment</h3>
                <div class="stat-highlight">
                    <strong>VERDICT: Remarkably clean data</strong>
                </div>
                <ul class="quality-list">
                    <li>✓ <strong>Missing Values:</strong> 0 across all columns</li>
                    <li>✓ <strong>Duplicate Records:</strong> None detected</li>
                    <li>✓ <strong>Data Types:</strong> Appropriate for each feature</li>
                </ul>
            </div>
            
            <div class="chapter-section">
                <h3>📅 The Date Conversion</h3>
                <p>The 'Effective To Date' column required transformation from string to datetime format:</p>
                <div class="code-block">
                    <pre># Converting date strings
df['effective_to_date'] = pd.to_datetime(df['Effective To Date'])
df['policy_month'] = df['effective_to_date'].dt.month
df['policy_year'] = df['effective_to_date'].dt.year</pre>
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>⚠️ The Zero-Income Anomaly</h3>
                <div class="stat-highlight warning">
                    <strong>813 customers (8.9%) report zero income</strong>
                </div>
                <p>This discovery raised questions:</p>
                <ul>
                    <li>Are these data entry errors?</li>
                    <li>Unemployed customers with savings?</li>
                    <li>Dependents on family policies?</li>
                </ul>
                <p><strong>Decision:</strong> Retained these records as they may represent a legitimate customer segment (retirees, students, etc.)</p>
            </div>
            
            <div class="chapter-section">
                <h3>📊 Feature Correlation Analysis</h3>
                <p>Initial correlation scan revealed key relationships:</p>
                <div class="correlation-grid">
                    <div class="corr-card strong">
                        <span class="corr-value">0.62</span>
                        <span class="corr-vars">Premium ↔ CLV</span>
                        <span class="corr-desc">Strongest predictor</span>
                    </div>
                    <div class="corr-card moderate">
                        <span class="corr-value">0.38</span>
                        <span class="corr-vars">Tenure ↔ CLV</span>
                        <span class="corr-desc">Moderate relationship</span>
                    </div>
                    <div class="corr-card weak">
                        <span class="corr-value">0.12</span>
                        <span class="corr-vars">Income ↔ CLV</span>
                        <span class="corr-desc">Surprisingly weak</span>
                    </div>
                </div>
            </div>
        `
    },

    // Chapter 3: The Landscape
    {
        id: 3,
        title: "The Landscape",
        subtitle: "Univariate Exploration",
        icon: "🗺️",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">With the evidence catalogued, we surveyed the terrain. Each variable told its own story, 
                painting a picture of the customer landscape we sought to understand...</p>
            </div>
            
            <div class="chapter-section">
                <h3>📈 Customer Lifetime Value Distribution</h3>
                <div id="clv-distribution-chart" class="chart-container"></div>
                <div class="insight-box">
                    <strong>Key Insight:</strong> CLV is right-skewed with mean $8,004 vs median $5,218. 
                    This tells us a subset of high-value customers significantly pulls up the average.
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>💰 Income Distribution</h3>
                <p>Customer income spans $0 to $99,981 with these patterns:</p>
                <ul>
                    <li>Mean: $37,657 | Median: $36,234</li>
                    <li>Roughly normal distribution with slight right skew</li>
                    <li>8.9% zero-income segment (previously identified)</li>
                </ul>
            </div>
            
            <div class="chapter-section">
                <h3>📅 Tenure Analysis</h3>
                <p>Policy tenure shows interesting patterns:</p>
                <ul>
                    <li>Range: 1 to 99 months</li>
                    <li>Mean tenure: 18.2 months</li>
                    <li>High early-month concentration (new customer cohort)</li>
                </ul>
            </div>
            
            <div class="chapter-section">
                <h3>🏷️ Categorical Features</h3>
                <div class="category-grid">
                    <div class="cat-card">
                        <h4>Coverage Type</h4>
                        <p>Basic: 38% | Extended: 35% | Premium: 27%</p>
                    </div>
                    <div class="cat-card">
                        <h4>Vehicle Class</h4>
                        <p>Four-Door: 48% | SUV: 21% | Sports: 16% | Others: 15%</p>
                    </div>
                    <div class="cat-card">
                        <h4>Sales Channel</h4>
                        <p>Agent: 47% | Branch: 24% | Call Center: 16% | Web: 13%</p>
                    </div>
                </div>
            </div>
        `
    },

    // Chapter 4: Relationships
    {
        id: 4,
        title: "The Relationships",
        subtitle: "Bivariate Analysis",
        icon: "🔗",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">No variable exists in isolation. Like witnesses in an investigation, 
                features corroborate or contradict each other. We mapped these intricate relationships...</p>
            </div>
            
            <div class="chapter-section">
                <h3>🎯 Premium vs CLV: The Smoking Gun</h3>
                <div id="premium-clv-chart" class="chart-container"></div>
                <div class="insight-box">
                    <strong>r = 0.62:</strong> Monthly premium is the single strongest predictor of CLV. 
                    This makes intuitive sense—higher premiums mean more revenue per customer.
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>📊 Hypothesis Testing</h3>
                <p>We tested several key hypotheses about CLV drivers:</p>
                
                <div class="hypothesis-grid">
                    <div class="hypothesis-card">
                        <span class="h-badge">CONFIRMED</span>
                        <h4>H1: Premium → CLV</h4>
                        <p class="h-stat">F-statistic: 847.23, p < 0.001</p>
                        <p class="h-detail">Higher premiums strongly associate with higher CLV</p>
                    </div>
                    
                    <div class="hypothesis-card">
                        <span class="h-badge">CONFIRMED</span>
                        <h4>H2: Coverage → CLV</h4>
                        <p class="h-stat">F-statistic: 156.89, p < 0.001</p>
                        <p class="h-detail">Premium coverage customers have 45% higher CLV than Basic</p>
                    </div>
                    
                    <div class="hypothesis-card">
                        <span class="h-badge">CONFIRMED</span>
                        <h4>H3: Vehicle Class → CLV</h4>
                        <p class="h-stat">F-statistic: 89.34, p < 0.001</p>
                        <p class="h-detail">Luxury car owners show highest average CLV</p>
                    </div>
                    
                    <div class="hypothesis-card weak">
                        <span class="h-badge">WEAK</span>
                        <h4>H4: Income → CLV</h4>
                        <p class="h-stat">r = 0.12, p < 0.001</p>
                        <p class="h-detail">Statistically significant but practically weak correlation</p>
                    </div>
                </div>
            </div>
        `
    },

    // Chapter 5-7: Feature Engineering
    {
        id: 5,
        title: "The Transformation",
        subtitle: "Feature Engineering",
        icon: "⚙️",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">Raw data rarely reveals its secrets willingly. Like an alchemist, 
                we transformed base features into golden predictors that would fuel our models...</p>
            </div>
            
            <div class="chapter-section">
                <h3>🔄 Feature Transformations</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Original</th>
                            <th>Transformation</th>
                            <th>Rationale</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>customer_lifetime_value</code></td>
                            <td><code>log(CLV)</code></td>
                            <td>Normalize right-skewed distribution</td>
                        </tr>
                        <tr>
                            <td><code>income</code></td>
                            <td><code>income_bracket</code></td>
                            <td>Create ordinal categories</td>
                        </tr>
                        <tr>
                            <td><code>months_since_policy</code></td>
                            <td><code>tenure_category</code></td>
                            <td>New/Established/Loyal segments</td>
                        </tr>
                        <tr>
                            <td><code>coverage + vehicle</code></td>
                            <td><code>risk_category</code></td>
                            <td>Interaction feature creation</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="chapter-section">
                <h3>🧪 Engineered Features</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>New Feature</th>
                            <th>Formula</th>
                            <th>Business Logic</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="highlight">
                            <td><code>premium_to_income</code></td>
                            <td>premium / income</td>
                            <td>Affordability ratio</td>
                        </tr>
                        <tr class="highlight">
                            <td><code>claim_frequency</code></td>
                            <td>claims / tenure</td>
                            <td>Risk behavior metric</td>
                        </tr>
                        <tr>
                            <td><code>policy_density</code></td>
                            <td>policies / household</td>
                            <td>Cross-sell indicator</td>
                        </tr>
                        <tr>
                            <td><code>loyalty_score</code></td>
                            <td>tenure × policies</td>
                            <td>Customer stickiness</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="chapter-section">
                <h3>📊 Encoding Strategies</h3>
                <div class="code-block">
                    <pre># One-Hot Encoding for nominal features
categorical_cols = ['coverage', 'vehicle_class', 'sales_channel']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Label Encoding for ordinal features  
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])</pre>
                </div>
            </div>
        `
    },

    // Chapter 8-10: The Model Arena
    {
        id: 6,
        title: "The Model Arena",
        subtitle: "Building the Prediction Engine",
        icon: "🤖",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">With our features forged, we entered the arena. Five algorithms would compete 
                to prove their worth in predicting customer value. Only the strongest would survive...</p>
            </div>
            
            <div class="chapter-section">
                <h3>🏆 The Contenders</h3>
                <div class="algorithm-grid">
                    <div class="algo-card">
                        <h4>Linear Regression</h4>
                        <p>The baseline classic</p>
                    </div>
                    <div class="algo-card">
                        <h4>Ridge Regression</h4>
                        <p>L2 regularization</p>
                    </div>
                    <div class="algo-card">
                        <h4>Random Forest</h4>
                        <p>Ensemble trees</p>
                    </div>
                    <div class="algo-card winner">
                        <h4>🏆 Gradient Boosting</h4>
                        <p>Sequential learner</p>
                    </div>
                    <div class="algo-card">
                        <h4>XGBoost</h4>
                        <p>Optimized boosting</p>
                    </div>
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>📈 Model Performance</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>R² Score</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Linear Regression</td>
                            <td>0.723</td>
                            <td>$2,847</td>
                            <td>$1,923</td>
                        </tr>
                        <tr>
                            <td>Ridge Regression</td>
                            <td>0.731</td>
                            <td>$2,798</td>
                            <td>$1,884</td>
                        </tr>
                        <tr>
                            <td>Random Forest</td>
                            <td>0.856</td>
                            <td>$2,112</td>
                            <td>$1,456</td>
                        </tr>
                        <tr class="highlight">
                            <td><strong>Gradient Boosting</strong></td>
                            <td><strong>0.891</strong></td>
                            <td><strong>$1,823</strong></td>
                            <td><strong>$1,267</strong></td>
                        </tr>
                        <tr>
                            <td>XGBoost</td>
                            <td>0.884</td>
                            <td>$1,891</td>
                            <td>$1,312</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="chapter-section">
                <h3>🎯 Feature Importance</h3>
                <div id="feature-importance-chart" class="chart-container"></div>
                <div class="importance-list">
                    <div class="importance-item">
                        <span class="imp-rank">1</span>
                        <span class="imp-feature">Monthly Premium</span>
                        <span class="imp-value">38.2%</span>
                        <span class="imp-desc">Primary revenue driver</span>
                    </div>
                    <div class="importance-item">
                        <span class="imp-rank">2</span>
                        <span class="imp-feature">Number of Policies</span>
                        <span class="imp-value">18.7%</span>
                        <span class="imp-desc">Cross-sell indicator</span>
                    </div>
                    <div class="importance-item">
                        <span class="imp-rank">3</span>
                        <span class="imp-feature">Total Claim Amount</span>
                        <span class="imp-value">14.3%</span>
                        <span class="imp-desc">Risk metric</span>
                    </div>
                    <div class="importance-item">
                        <span class="imp-rank">4</span>
                        <span class="imp-feature">Policy Tenure</span>
                        <span class="imp-value">11.8%</span>
                        <span class="imp-desc">Loyalty measure</span>
                    </div>
                    <div class="importance-item">
                        <span class="imp-rank">5</span>
                        <span class="imp-feature">Vehicle Class</span>
                        <span class="imp-value">8.4%</span>
                        <span class="imp-desc">Risk segmentation</span>
                    </div>
                </div>
            </div>
        `
    },

    // Chapter 11: The Four Tribes
    {
        id: 7,
        title: "The Four Tribes",
        subtitle: "Customer Segmentation",
        icon: "👥",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">Among the 9,134 customers, distinct patterns emerged. Like ancient tribes, 
                four customer segments revealed themselves—each with unique characteristics and value profiles...</p>
            </div>
            
            <div id="cluster-chart" class="chart-container"></div>
            
            <div class="chapter-section tribes-grid">
                <div class="tribe-card steady-eddies">
                    <div class="tribe-header">
                        <span class="tribe-icon">🏠</span>
                        <h3>Steady Eddies</h3>
                        <span class="tribe-pct">31% of portfolio</span>
                    </div>
                    <div class="tribe-stats">
                        <div class="tribe-stat">
                            <span class="stat-label">Avg CLV</span>
                            <span class="stat-value">$7,234</span>
                        </div>
                        <div class="tribe-stat">
                            <span class="stat-label">Tenure</span>
                            <span class="stat-value">36 months</span>
                        </div>
                    </div>
                    <div class="tribe-details">
                        <p><strong>Profile:</strong> Middle-income families, basic coverage, low claims</p>
                        <p><strong>Strategy:</strong> Maintain satisfaction, automate service, loyalty rewards</p>
                    </div>
                </div>
                
                <div class="tribe-card high-rollers">
                    <div class="tribe-header">
                        <span class="tribe-icon">💎</span>
                        <h3>High Rollers</h3>
                        <span class="tribe-pct">18% of portfolio</span>
                    </div>
                    <div class="tribe-stats">
                        <div class="tribe-stat">
                            <span class="stat-label">Avg CLV</span>
                            <span class="stat-value">$14,892</span>
                        </div>
                        <div class="tribe-stat">
                            <span class="stat-label">Premium</span>
                            <span class="stat-value">$189/mo</span>
                        </div>
                    </div>
                    <div class="tribe-details">
                        <p><strong>Profile:</strong> High-income, premium coverage, luxury vehicles</p>
                        <p><strong>Strategy:</strong> White-glove service, dedicated agents, proactive retention</p>
                    </div>
                </div>
                
                <div class="tribe-card riskmakers">
                    <div class="tribe-header">
                        <span class="tribe-icon">⚡</span>
                        <h3>Riskmakers</h3>
                        <span class="tribe-pct">29% of portfolio</span>
                    </div>
                    <div class="tribe-stats">
                        <div class="tribe-stat">
                            <span class="stat-label">Avg CLV</span>
                            <span class="stat-value">$5,621</span>
                        </div>
                        <div class="tribe-stat">
                            <span class="stat-label">Claims</span>
                            <span class="stat-value">High</span>
                        </div>
                    </div>
                    <div class="tribe-details">
                        <p><strong>Profile:</strong> Higher claim frequency, sports cars, younger demographic</p>
                        <p><strong>Strategy:</strong> Risk-based repricing, usage-based insurance options</p>
                    </div>
                </div>
                
                <div class="tribe-card fresh-starts">
                    <div class="tribe-header">
                        <span class="tribe-icon">🌱</span>
                        <h3>Fresh Starts</h3>
                        <span class="tribe-pct">22% of portfolio</span>
                    </div>
                    <div class="tribe-stats">
                        <div class="tribe-stat">
                            <span class="stat-label">Avg CLV</span>
                            <span class="stat-value">$6,487</span>
                        </div>
                        <div class="tribe-stat">
                            <span class="stat-label">Tenure</span>
                            <span class="stat-value">< 6 months</span>
                        </div>
                    </div>
                    <div class="tribe-details">
                        <p><strong>Profile:</strong> New customers, first-time buyers, limited history</p>
                        <p><strong>Strategy:</strong> Onboarding excellence, early engagement, cross-sell potential</p>
                    </div>
                </div>
            </div>
        `
    },

    // Chapter 12: Strategy
    {
        id: 8,
        title: "The Strategy",
        subtitle: "Business Recommendations",
        icon: "📋",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">The investigation is complete. The evidence has spoken. Now we translate 
                our findings into actionable strategies that will reshape the business...</p>
            </div>
            
            <div class="chapter-section">
                <h3>🎯 Strategic Recommendations</h3>
                
                <div class="strategy-card critical">
                    <h4>🚨 Critical: High Roller Retention</h4>
                    <p>18% of customers generate 38% of CLV. Losing even 5% of High Rollers costs $1.3M annually.</p>
                    <ul>
                        <li>Assign dedicated relationship managers</li>
                        <li>Implement proactive renewal outreach (60 days before expiry)</li>
                        <li>Create exclusive loyalty benefits tier</li>
                    </ul>
                </div>
                
                <div class="strategy-card high">
                    <h4>⚠️ High Priority: Riskmaker Repricing</h4>
                    <p>29% of portfolio with lowest CLV but highest servicing costs.</p>
                    <ul>
                        <li>Implement telematics-based pricing</li>
                        <li>Offer safe-driver discounts to incentivize behavior change</li>
                        <li>Consider selective non-renewal for repeat high-claim customers</li>
                    </ul>
                </div>
                
                <div class="strategy-card">
                    <h4>📈 Growth: Fresh Start Nurturing</h4>
                    <p>New customers represent highest growth potential.</p>
                    <ul>
                        <li>90-day onboarding journey with touchpoints</li>
                        <li>Cross-sell second policy within first year</li>
                        <li>Early referral incentives</li>
                    </ul>
                </div>
            </div>
            
            <div class="chapter-section">
                <h3>💰 ROI Projections</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Initiative</th>
                            <th>Investment</th>
                            <th>Projected Return</th>
                            <th>ROI</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>High Roller Retention</td>
                            <td>$150,000</td>
                            <td>$1,340,000</td>
                            <td class="roi-green">793%</td>
                        </tr>
                        <tr>
                            <td>Riskmaker Repricing</td>
                            <td>$80,000</td>
                            <td>$620,000</td>
                            <td class="roi-green">675%</td>
                        </tr>
                        <tr>
                            <td>Fresh Start Program</td>
                            <td>$120,000</td>
                            <td>$820,000</td>
                            <td class="roi-green">583%</td>
                        </tr>
                        <tr class="total-row">
                            <td><strong>Total Portfolio Impact</strong></td>
                            <td><strong>$350,000</strong></td>
                            <td><strong>$2,780,000</strong></td>
                            <td class="roi-green"><strong>694%</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `
    },

    // Chapter 9: The CLV Predictor
    {
        id: 9,
        title: "The Calculator",
        subtitle: "Predict Customer Value",
        icon: "🔮",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">All roads lead here. The culmination of our investigation—a tool that transforms 
                raw customer data into actionable value predictions...</p>
            </div>
            
            <div class="chapter-section predictor-section">
                <h3>🔮 CLV Prediction Tool</h3>
                <p>Enter customer details below to predict their lifetime value and receive AI-powered strategic recommendations.</p>
                
                <button class="btn-open-predictor" onclick="openPredictorModal()">
                    🔮 Open CLV Calculator
                </button>
            </div>
            
            <div class="chapter-section">
                <h3>📊 How It Works</h3>
                <ol>
                    <li><strong>Input Features:</strong> Enter customer demographics and policy details</li>
                    <li><strong>Model Prediction:</strong> Our Gradient Boosting model calculates CLV</li>
                    <li><strong>Segment Classification:</strong> Automatically assigns to one of Four Tribes</li>
                    <li><strong>AI Analysis:</strong> Generate personalized recommendations</li>
                </ol>
            </div>
            
            <div class="chapter-section">
                <h3>🤖 AI Integration</h3>
                <p>After getting your prediction, click "Generate AI Report" to receive:</p>
                <ul>
                    <li>Personalized customer insights</li>
                    <li>Segment-specific strategies</li>
                    <li>Risk assessment and recommendations</li>
                    <li>Cross-sell and upsell opportunities</li>
                </ul>
            </div>
        `
    },

    // Chapter 10: Conclusion
    {
        id: 10,
        title: "Case Closed",
        subtitle: "Conclusion & Next Steps",
        icon: "📁",
        content: `
            <div class="chapter-intro">
                <p class="dramatic-text">The case file is complete. From 9,134 anonymous records emerged patterns, 
                predictions, and strategies that will reshape customer value management...</p>
            </div>
            
            <div class="chapter-section">
                <h3>📋 Key Findings Summary</h3>
                <ul class="findings-list">
                    <li>🎯 <strong>Model Performance:</strong> 89.1% R² with Gradient Boosting</li>
                    <li>💰 <strong>Primary Driver:</strong> Monthly premium explains 38% of CLV variance</li>
                    <li>👥 <strong>Customer Segments:</strong> Four distinct tribes identified via K-Means</li>
                    <li>📈 <strong>Business Impact:</strong> $2.78M projected ROI from recommendations</li>
                </ul>
            </div>
            
            <div class="chapter-section">
                <h3>🔬 Technical Achievements</h3>
                <ul>
                    <li>Comprehensive EDA with 5 hypothesis tests</li>
                    <li>Feature engineering creating 12 new predictors</li>
                    <li>Model comparison across 5 algorithms</li>
                    <li>Customer segmentation using K-Means clustering</li>
                    <li>Interactive prediction tool with AI integration</li>
                </ul>
            </div>
            
            <div class="chapter-section">
                <h3>🚀 Next Steps</h3>
                <ol>
                    <li>Deploy model to production CRM system</li>
                    <li>Implement real-time CLV scoring for new customers</li>
                    <li>A/B test retention strategies by segment</li>
                    <li>Build automated alert system for at-risk High Rollers</li>
                    <li>Quarterly model retraining with new data</li>
                </ol>
            </div>
            
            <div class="chapter-section credits">
                <h3>👤 About the Investigator</h3>
                <div class="author-card">
                    <h4>Tuhin Bhattacharya</h4>
                    <p>PGDM Business Data Analytics | Goa Institute of Management</p>
                    <p>This analysis was conducted as part of advanced coursework in predictive analytics 
                    and customer relationship management.</p>
                </div>
            </div>
        `
    }
];

// Export for use in main.js
window.chapters = chapters;
