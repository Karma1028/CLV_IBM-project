"""
Main build script for the Magnum Opus CLV Technical Memoir
Uses chapter functions from generate_magnum_opus.py with local styles/utils

Run from CLV_IEEE_Project directory:
  python magnum_opus_chapters/main.py
"""
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

# Import local styles
from magnum_opus_chapters.styles import create_styles

# Import chapter functions from original script
import importlib.util
spec = importlib.util.spec_from_file_location("original_script", os.path.join(PROJECT_ROOT, "generate_magnum_opus.py"))
original_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_script)

# Re-export chapter functions for external use
chapter_01_genesis = original_script.chapter_01_genesis
chapter_02_forensic_audit = original_script.chapter_02_forensic_audit
chapter_03_landscape = original_script.chapter_03_landscape
chapter_04_relationships = original_script.chapter_04_relationships
chapter_05_interactions = original_script.chapter_05_interactions
chapter_06_alchemy_part1 = original_script.chapter_06_alchemy_part1
chapter_07_alchemy_part2 = original_script.chapter_07_alchemy_part2
chapter_08_experiment = original_script.chapter_08_experiment
chapter_09_refinement = original_script.chapter_09_refinement
chapter_10_inference = original_script.chapter_10_inference
chapter_11_tribes = original_script.chapter_11_tribes
chapter_12_strategy = original_script.chapter_12_strategy

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'report')
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'CLV_Magnum_Opus.pdf')


def build_title_page(story, styles):
    """Create the title page."""
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph(
        "The Mathematical Memoir of Customer Lifetime Value",
        styles['BookTitle']
    ))
    story.append(Paragraph(
        "A Technical Deep Dive into Predictive Analytics for Auto Insurance",
        styles['BookSubtitle']
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        "From Raw Data to Actionable Intelligence:<br/>"
        "Feature Engineering, Machine Learning, and Customer Segmentation",
        styles['DenseBody']
    ))
    story.append(PageBreak())


def build_table_of_contents(story, styles):
    """Create the table of contents."""
    story.append(Paragraph("Table of Contents", styles['ChapterTitle']))
    
    toc_items = [
        "Chapter 1: The Genesis — Project Purpose and Dataset Introduction",
        "Chapter 2: The Forensic Audit — Data Quality and Anomaly Detection",
        "Chapter 3: The Landscape — Univariate Distribution Analysis",
        "Chapter 4: The Relationships — Bivariate and Multivariate Exploration",
        "Chapter 5: The Interactions — Advanced Feature Relationships",
        "Chapter 6: The Alchemy Part I — Data Transformation Techniques",
        "Chapter 7: The Alchemy Part II — Feature Engineering at Scale",
        "Chapter 8: The Experiment — Model Selection and Training",
        "Chapter 9: The Refinement — Hyperparameter Tuning and Validation",
        "Chapter 10: The Inference — Model Interpretation and Deployment",
        "Chapter 11: The Tribes — Customer Segmentation via Clustering",
        "Chapter 12: The Strategy — Business Recommendations and Conclusion",
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, styles['DenseBody']))
    
    story.append(PageBreak())


def build_document():
    """Build the complete PDF document."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = create_styles()
    story = []
    
    # Title and TOC
    build_title_page(story, styles)
    build_table_of_contents(story, styles)
    
    # Generate chapters
    print("=" * 60)
    print("MAGNUM OPUS CLV TECHNICAL MEMOIR - BUILD STARTED")
    print("=" * 60)
    
    print("\n[1/12] Generating Chapter 1: The Genesis...")
    chapter_01_genesis(story, styles)
    
    print("[2/12] Generating Chapter 2: The Forensic Audit...")
    chapter_02_forensic_audit(story, styles)
    
    print("[3/12] Generating Chapter 3: The Landscape...")
    chapter_03_landscape(story, styles)
    
    print("[4/12] Generating Chapter 4: The Relationships...")
    chapter_04_relationships(story, styles)
    
    print("[5/12] Generating Chapter 5: The Interactions...")
    chapter_05_interactions(story, styles)
    
    print("[6/12] Generating Chapter 6: The Alchemy Part I...")
    chapter_06_alchemy_part1(story, styles)
    
    print("[7/12] Generating Chapter 7: The Alchemy Part II...")
    chapter_07_alchemy_part2(story, styles)
    
    print("[8/12] Generating Chapter 8: The Experiment...")
    chapter_08_experiment(story, styles)
    
    print("[9/12] Generating Chapter 9: The Refinement...")
    chapter_09_refinement(story, styles)
    
    print("[10/12] Generating Chapter 10: The Inference...")
    chapter_10_inference(story, styles)
    
    print("[11/12] Generating Chapter 11: The Tribes...")
    chapter_11_tribes(story, styles)
    
    print("[12/12] Generating Chapter 12: The Strategy...")
    chapter_12_strategy(story, styles)
    
    # Build PDF
    print("\n" + "=" * 60)
    print("Building PDF document...")
    doc.build(story)
    print(f"PDF saved to: {OUTPUT_PDF}")
    print("=" * 60)
    
    return OUTPUT_PDF


if __name__ == "__main__":
    build_document()
