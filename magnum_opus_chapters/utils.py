"""
Utility functions for the Magnum Opus CLV Technical Memoir
Matches the utility functions from generate_magnum_opus.py
"""
import os
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Image

# Configuration - paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures")


def add_figure(story, fig_name, caption, styles, width=5.5*inch):
    """Add a figure with caption to the story, ensuring it fits on page."""
    fig_path = os.path.join(FIGURES_DIR, fig_name)
    if os.path.exists(fig_path):
        try:
            # Calculate proportional height while ensuring page fit
            from PIL import Image as PILImage
            with PILImage.open(fig_path) as pil_img:
                orig_width, orig_height = pil_img.size
                aspect_ratio = orig_height / orig_width
                
            # Calculate dimensions that fit on page - limit max height
            max_height = 4.5 * inch  # Reduced to prevent too tall images
            calc_height = width * aspect_ratio
            
            if calc_height > max_height:
                # Scale down to fit
                width = max_height / aspect_ratio
                calc_height = max_height
            
            img = Image(fig_path, width=width, height=calc_height)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 4))  # Reduced spacer
            story.append(img)
            story.append(Paragraph(caption, styles['FigCaption']))
            story.append(Spacer(1, 4))  # Small spacer after caption
        except Exception as e:
            story.append(Paragraph(f"[Figure: {fig_name} - {str(e)}]", styles['DenseBody']))
    else:
        story.append(Paragraph(f"[Figure not found: {fig_name}]", styles['DenseBody']))


def add_code(story, code_text, styles):
    """Add a code block to the story."""
    # Escape special characters
    code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    code_text = code_text.replace('\n', '<br/>')
    story.append(Paragraph(code_text, styles['CodeBlock']))
