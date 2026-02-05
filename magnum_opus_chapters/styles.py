"""
Styles for the Magnum Opus CLV Technical Memoir
Matches the styles from generate_magnum_opus.py
"""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def create_styles():
    """Create all paragraph styles for the document."""
    styles = getSampleStyleSheet()
    
    # Title Style
    styles.add(ParagraphStyle(
        name='BookTitle',
        fontName='Times-Bold',
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=colors.HexColor('#1a1a2e')
    ))
    
    # Subtitle
    styles.add(ParagraphStyle(
        name='BookSubtitle',
        fontName='Times-Italic',
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=50,
        textColor=colors.HexColor('#4a4e69')
    ))
    
    # Chapter Title
    styles.add(ParagraphStyle(
        name='ChapterTitle',
        fontName='Times-Bold',
        fontSize=22,
        leading=28,
        alignment=TA_LEFT,
        spaceBefore=30,
        spaceAfter=20,
        textColor=colors.HexColor('#1a1a2e')
    ))
    
    # Section Heading
    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Times-Bold',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#22223b')
    ))
    
    # Body Text - Dense, justified
    styles.add(ParagraphStyle(
        name='DenseBody',
        fontName='Times-Roman',
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=6,
        firstLineIndent=20
    ))
    
    # Code Style
    styles.add(ParagraphStyle(
        name='CodeBlock',
        fontName='Courier',
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceBefore=8,
        spaceAfter=8,
        leftIndent=20,
        backColor=colors.HexColor('#f8f9fa'),
        borderColor=colors.HexColor('#dee2e6'),
        borderWidth=1,
        borderPadding=8
    ))
    
    # Caption
    styles.add(ParagraphStyle(
        name='FigCaption',
        fontName='Times-Italic',
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=16,
        textColor=colors.HexColor('#4a4e69')
    ))
    
    # Formula explanation
    styles.add(ParagraphStyle(
        name='FormulaExplain',
        fontName='Times-Italic',
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=10,
        textColor=colors.HexColor('#2d3436')
    ))
    
    return styles
