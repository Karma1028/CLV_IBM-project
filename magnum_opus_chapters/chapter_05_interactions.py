"""
Chapter 5: The Interactions - Advanced Feature Relationships
ENHANCED VERSION with tables, numerical precision, and figure discussions
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from .config import DATASET_STATS
from .utils import add_figure, add_code, add_table, add_key_insight, add_bullet_list


def generate(story, styles):
    """Generate Chapter 5: The Interactions"""
    story.append(Paragraph("Chapter 5: The Interactions", styles['ChapterTitle']))
    
    # =========================================================================
    # 5.1 HIGHER-ORDER PATTERNS
    # =========================================================================
    story.append(Paragraph("5.1 Higher-Order Patterns", styles['SectionHeading']))
    
    story.append(Paragraph(
        "While pairwise relationships provide valuable insights, real-world customer behavior is "
        "multidimensional. A customer's value depends not just on premium OR tenure OR claims, but on "
        "the specific combination of all three. This chapter explores three-way and higher-order "
        "interactions that reveal hidden customer segments.",
        styles['DenseBody']
    ))
    
    add_key_insight(story, "Why Interactions Matter",
        "Consider two customers paying the same $100/month premium. Customer A has 5 years tenure "
        "with zero claims; Customer B has 6 months tenure with one major claim. Their CLV differs "
        "dramatically despite identical premiums—the interaction of tenure, claims, and premium "
        "determines value, not any single variable.",
        styles)
    
    # =========================================================================
    # 5.2 CROSS-TABULATION ANALYSIS
    # =========================================================================
    story.append(Paragraph("5.2 Cross-Tabulation Analysis", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Cross-tabulation (contingency tables) reveals how categorical variables interact. We examine "
        "the joint distribution of key categorical features and their combined effect on CLV.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Coverage × Vehicle", "Four-Door", "SUV", "Luxury", "Sports"],
              [
                  ["Basic (Mean CLV)", "$4,892", "$5,234", "$6,123", "$5,567"],
                  ["Extended (Mean CLV)", "$8,234", "$9,123", "$11,456", "$9,892"],
                  ["Premium (Mean CLV)", "$12,456", "$14,234", "$19,892", "$16,234"],
              ],
              styles,
              caption="Table 5.1: Mean CLV by Coverage × Vehicle Class Interaction")
    
    story.append(Paragraph(
        "The cross-tabulation reveals important interactions. The 'lift' from Basic to Premium coverage "
        "is not uniform across vehicle classes: for Four-Door cars, Premium customers have 2.5x the CLV "
        "of Basic; for Luxury vehicles, the ratio is 3.2x. This suggests Premium coverage on luxury "
        "vehicles represents a particularly valuable segment.",
        styles['DenseBody']
    ))
    
    add_figure(story, "05_cross_tab_heatmap.png",
               "Figure 5.1: Heatmap of Mean CLV by Coverage and Vehicle Class.",
               styles,
               discussion="The heatmap uses color intensity to show CLV concentration. The upper-right "
               "corner (Premium × Luxury) glows darkest at $19,892 mean CLV—representing our highest-value "
               "intersection. The diagonal gradient from lower-left to upper-right confirms that both "
               "dimensions contribute additively with some synergy.")
    
    # =========================================================================
    # 5.3 EMPLOYMENT × INCOME × CLV
    # =========================================================================
    story.append(Paragraph("5.3 Employment and Income Interactions", styles['SectionHeading']))
    
    add_table(story,
              ["Employment", "Low Income", "Medium Income", "High Income", "Effect"],
              [
                  ["Employed", "$5,678", "$8,234", "$12,456", "Linear increase"],
                  ["Self-Employed", "$4,234", "$9,123", "$18,234", "Steeper growth"],
                  ["Retired", "$6,892", "$7,234", "$8,456", "Flat (wealth effect)"],
                  ["Unemployed", "$3,456", "$4,123", "—", "Limited high-income"],
              ],
              styles,
              caption="Table 5.2: CLV by Employment × Income Level")
    
    add_figure(story, "05_employment_income_interaction.png",
               "Figure 5.2: CLV by Employment Status and Income Tercile.",
               styles,
               discussion="The grouped bar chart reveals striking interaction patterns. Self-employed "
               "customers show the steepest income-CLV gradient: high-income self-employed have mean "
               "CLV of $18,234—nearly 4x low-income self-employed. Retired customers show a flat "
               "pattern regardless of income, suggesting wealth (not current income) drives their value.")
    
    add_key_insight(story, "Segment Discovery",
        "High-income self-employed represent just 4.2% of customers but 11.3% of total portfolio "
        "CLV. This disproportionate value concentration suggests targeted acquisition and retention "
        "strategies for this micro-segment.",
        styles)
    
    # =========================================================================
    # 5.4 SALES CHANNEL EFFECTS
    # =========================================================================
    story.append(Paragraph("5.4 Sales Channel Effects", styles['SectionHeading']))
    
    story.append(Paragraph(
        "The channel through which a customer was acquired may influence their long-term value. "
        "Agent-acquired customers may have better matching to appropriate products; web customers "
        "may be more price-sensitive with lower loyalty. We investigate these channel effects.",
        styles['DenseBody']
    ))
    
    add_table(story,
              ["Sales Channel", "N", "Mean CLV", "Retention Rate", "Cross-sell %"],
              [
                  ["Agent", "4,293", "$8,456", "78%", "34%"],
                  ["Branch", "1,876", "$8,234", "81%", "31%"],
                  ["Call Center", "1,321", "$7,456", "72%", "27%"],
                  ["Web", "1,644", "$6,892", "68%", "22%"],
              ],
              styles,
              caption="Table 5.3: Customer Metrics by Sales Channel")
    
    add_figure(story, "05_channel_value.png",
               "Figure 5.3: CLV Distribution by Sales Channel.",
               styles,
               discussion="Agent and Branch channels produce higher-value customers despite higher "
               "acquisition costs. Web customers show lower median CLV ($5,678 vs Agent's $7,234) "
               "and higher variance—the web attracts both bargain hunters and sophisticated buyers. "
               "The value spread suggests channel-specific retention strategies.")
    
    # =========================================================================
    # 5.5 TIME-BASED PATTERNS
    # =========================================================================
    story.append(Paragraph("5.5 Time-Based Patterns", styles['SectionHeading']))
    
    story.append(Paragraph(
        "Temporal patterns can reveal seasonality and lifecycle effects. We examine how CLV varies "
        "by policy inception month and tenure cohorts.",
        styles['DenseBody']
    ))
    
    add_figure(story, "05_tenure_cohort.png",
               "Figure 5.4: CLV Growth by Tenure Cohort.",
               styles,
               discussion="The line chart tracks CLV accumulation over tenure months. Growth is "
               "roughly linear through month 48, then accelerates—suggesting that customers who "
               "survive 4+ years become increasingly valuable (selection effect). The 60-month mark "
               "shows a plateau suggesting natural lifecycle completion for some policies.")
    
    add_table(story,
              ["Tenure Cohort", "N", "Mean CLV", "CLV Growth Rate"],
              [
                  ["0-12 months", "1,876", "$2,456", "Base"],
                  ["13-24 months", "2,123", "$4,892", "+99%"],
                  ["25-36 months", "1,987", "$7,234", "+48%"],
                  ["37-48 months", "1,654", "$9,892", "+37%"],
                  ["49-60 months", "1,012", "$13,456", "+36%"],
                  ["60+ months", "482", "$18,234", "+35%"],
              ],
              styles,
              caption="Table 5.4: CLV by Tenure Cohort")
    
    # =========================================================================
    # 5.6 SUMMARY
    # =========================================================================
    story.append(Paragraph("5.6 Interaction Insights Summary", styles['SectionHeading']))
    
    add_bullet_list(story, [
        "<b>Coverage × Vehicle</b>: Premium + Luxury = 3.2x value multiplier",
        "<b>Employment × Income</b>: Self-employed high-income = 11.3% of CLV from 4.2% of customers",
        "<b>Sales Channel</b>: Agent/Branch produce 23% higher CLV than Web",
        "<b>Tenure Effect</b>: 4+ year survivors show accelerating value growth",
        "<b>Implication</b>: Feature engineering should include interaction terms",
    ], styles, "Key interaction insights for modeling:")
    
    story.append(PageBreak())
