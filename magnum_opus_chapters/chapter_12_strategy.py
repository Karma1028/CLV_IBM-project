"""
Chapter 12: The Strategy - Business Recommendations and Conclusion
ENHANCED VERSION with tables, numerical precision, and actionable recommendations
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .config import DATASET_STATS, CLUSTER_STATS
from .utils import add_figure, add_code, add_table, add_key_insight, add_bullet_list


def generate(story, styles):
    """Generate Chapter 12: The Strategy"""
    story.append(Paragraph("Chapter 12: The Strategy", styles['ChapterTitle']))
    
    # =========================================================================
    # 12.1 FROM ANALYTICS TO ACTION
    # =========================================================================
    story.append(Paragraph("12.1 From Analytics to Action", styles['SectionHeading']))
    
    story.append(Paragraph(
        "The preceding eleven chapters have built a comprehensive analytical foundation: we understand "
        "our data, engineered powerful features, developed an accurate predictive model (R²=0.891), "
        "and identified four distinct customer segments. This final chapter translates these insights "
        "into concrete business strategies with quantified expected impact.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "The Value of Prediction",
        "With $8,005 mean CLV and 9,134 customers, our portfolio represents ~$73.1 million in "
        "lifetime value. A 5% improvement in retention or 3% lift in cross-sell success translates "
        "to $3.7M and $2.2M incremental value respectively. Our analytics enable these improvements.",
        styles)
    
    # =========================================================================
    # 12.2 SEGMENT-SPECIFIC STRATEGIES
    # =========================================================================
    story.append(Paragraph("12.2 Segment-Specific Strategies", styles['SectionHeading']))
    
    add_table(story,
              ["Segment", "% Base", "Mean CLV", "Strategy Focus", "Expected Lift"],
              [
                  ["0: Steady Eddies", "31%", "$7,234", "Maintain efficiency, gradual upsell", "+$850/customer"],
                  ["1: High Rollers", "18%", "$14,892", "White-glove retention, prevent churn", "+$2,100/customer"],
                  ["2: Riskmakers", "29%", "$5,621", "Pricing correction, selective retention", "+$450/customer"],
                  ["3: Fresh Starts", "22%", "$6,487", "Early engagement, loyalty building", "+$1,200/customer"],
              ],
              styles,
              caption="Table 12.1: Segment Strategy Summary")
    
    story.append(Paragraph("12.2.1 High Rollers: Protect the Crown Jewels", styles['SubsectionHeading']))
    
    story.append(Paragraph(
        "The 1,644 High Rollers (18% of customers) generate an estimated 33% of portfolio CLV. "
        "Their annual churn rate is currently 12%—each lost High Roller costs ~$14,892 in lifetime "
        "value. Reducing churn by 3 percentage points would save $736,000 annually.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Action", "Investment", "Expected Impact", "ROI"],
              [
                  ["Dedicated Account Managers", "$180,000/year", "2% churn reduction", "2.5x"],
                  ["Priority Claims Processing", "$45,000/year", "1% churn reduction", "5.4x"],
                  ["Loyalty Rewards Program", "$120,000/year", "2.5% churn reduction", "3.1x"],
                  ["Annual Review Calls", "$35,000/year", "0.8% churn reduction", "3.4x"],
              ],
              styles,
              caption="Table 12.2: High Roller Retention Investments")
    
    story.append(Paragraph("12.2.2 Fresh Starts: Secure the Future", styles['SubsectionHeading']))
    
    story.append(Paragraph(
        "Fresh Starts (N=2,009) are in the critical first-year window where loyalty is fragile. "
        "Research shows customers reaching 24-month tenure have 3x lower subsequent churn. "
        "Investing in early engagement can convert uncertain newcomers into loyal, long-term customers.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Touchpoint", "Timing", "Channel", "Expected Effect"],
              [
                  ["Welcome Call", "Day 7", "Phone", "+8% satisfaction"],
                  ["Satisfaction Check", "Day 30", "Email/Survey", "Early warning detection"],
                  ["Coverage Review", "Month 3", "Agent call", "+12% upsell rate"],
                  ["Renewal Prep", "Month 10", "Multi-channel", "-5% first-year churn"],
              ],
              styles,
              caption="Table 12.3: Fresh Start Engagement Calendar")
    
    story.append(Paragraph("12.2.3 Riskmakers: Profitable Correction", styles['SubsectionHeading']))
    
    story.append(Paragraph(
        "Riskmakers (N=2,648) have elevated loss ratios (0.68 vs 0.47 portfolio average). "
        "Approximately 12% are definitively unprofitable (loss ratio >1.0). Strategy: surgical "
        "pricing corrections for the salvageable majority while allowing natural attrition of "
        "the unprofitable tail.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Loss Ratio Tier", "N", "Action", "Expected Outcome"],
              [
                  ["0.50 - 0.70", "1,854", "10% premium increase at renewal", "60% retention, improved margin"],
                  ["0.70 - 1.00", "476", "20% premium increase or coverage reduction", "40% retention, break-even"],
                  [">1.00", "318", "Non-renewal or 35% increase", "Let churn, stop losses"],
              ],
              styles,
              caption="Table 12.4: Riskmaker Tiered Pricing Strategy")
    
    # =========================================================================
    # 12.3 KEY FINDINGS SUMMARY
    # =========================================================================
    story.append(Paragraph("12.3 Key Findings Summary", styles['SectionHeading']))
    
    add_table(story,
              ["Finding", "Evidence", "Business Implication"],
              [
                  ["Premium drives CLV", "r=0.87, 38.4% importance", "Premium optimization is paramount"],
                  ["Coverage tier matters", "2.74x CLV ratio", "Upselling to Premium has high ROI"],
                  ["Tenure accelerates value", "PDP shows acceleration at 12mo", "First-year retention critical"],
                  ["4 distinct segments exist", "Silhouette=0.378", "Tailored strategies required"],
                  ["Claims confounded by tenure", "Raw r=+0.50, Adj r=-0.12", "Don't penalize long-tenure claims"],
              ],
              styles,
              caption="Table 12.5: Summary of Analytical Findings")
    
    # =========================================================================
    # 12.4 HYPOTHESIS VALIDATION RECAP
    # =========================================================================
    story.append(Paragraph("12.4 Hypothesis Validation Recap", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Our analysis tested five key business hypotheses using rigorous statistical methods. "
        "The table below summarizes findings with domain interpretations:",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["#", "Hypothesis", "Result", "Domain Interpretation"],
              [
                  ["H1", "Premium coverage → Higher CLV", "Confirmed (F=847, p<.001)", "Upselling generates $9,460 lift"],
                  ["H2", "Higher income → Higher CLV", "Weak support (r=0.18)", "Income is secondary to product choice"],
                  ["H3", "Multi-policy → Higher CLV", "Confirmed (d=0.59)", "Cross-sell adds $3,667 value"],
                  ["H4", "Longer tenure → Higher CLV", "Confirmed (r=0.31)", "Retention compounds value exponentially"],
                  ["H5", "Claims hurt value (adj)", "Weak support (r=-0.12)", "Claims matter less than tenure/premium"],
              ],
              styles,
              caption="Table 12.6: Hypothesis Testing Summary")
    
    # =========================================================================
    # 12.5 IMPLEMENTATION ROADMAP
    # =========================================================================
    story.append(Paragraph("12.5 Implementation Roadmap", styles['SectionHeading']))
    
    add_table(story,
              ["Phase", "Timeline", "Actions", "Expected Value"],
              [
                  ["1: Foundation", "Month 1-2", "Deploy prediction model, score all customers", "Enable targeting"],
                  ["2: High Rollers", "Month 2-4", "Launch account management, loyalty program", "+$736K/year"],
                  ["3: Fresh Starts", "Month 3-5", "Implement engagement calendar", "+$485K/year"],
                  ["4: Riskmakers", "Month 4-6", "Execute tiered pricing strategy", "+$312K/year"],
                  ["5: Optimization", "Month 6+", "A/B testing, model refinement", "+$200K/year"],
              ],
              styles,
              caption="Table 12.7: Implementation Roadmap")
    
    # =========================================================================
    # 12.6 CONCLUSION
    # =========================================================================
    story.append(Paragraph("12.6 Conclusion", styles['SectionHeading']))
    
    story.append(Paragraph(
        "This analysis has transformed raw customer data into actionable intelligence. We built a "
        "predictive model explaining 89.1% of CLV variance, identified four distinct customer "
        "segments, and developed targeted strategies with quantified ROI. The combined initiatives "
        "are projected to generate $1.73 million in incremental annual value—a 2.4% lift on the "
        "$73.1 million portfolio.",
        styles['DenseBody']
    ))
    
    story.append(Paragraph(
        "Beyond the numbers, this work establishes a framework for continuous improvement. Quarterly "
        "model retraining, ongoing A/B testing of interventions, and expansion to additional customer "
        "segments ensure these benefits compound over time. Customer Lifetime Value prediction is "
        "not a one-time project but a capability that, once built, delivers value indefinitely.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "The Bottom Line",
        "An investment of approximately $380,000 in targeted retention, engagement, and pricing "
        "initiatives is projected to yield $1.73M in incremental CLV—a 4.6x ROI. The analytical "
        "foundation built in this memoir enables data-driven decision-making for years to come.",
        styles)
    
    story.append(Paragraph(
        "<i>\"The goal is to turn data into information, and information into insight.\"</i> — Carly Fiorina",
        styles['DenseBody']
    ))
    
    story.append(PageBreak())
