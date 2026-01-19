"""Helper functions for creating test PDF files"""

import io


def create_test_pdf_report(title="Model Training Report", filename="report.pdf"):
    """
    Create a synthetic PDF file with sample ML report content.

    Args:
        title: Title of the PDF report
        filename: Output filename (not used, returns bytes)

    Returns:
        PDF content as bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        # Fallback: create a minimal PDF
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n%%EOF"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    # Introduction
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(
        Paragraph(
            "This report summarizes the model training results for the ResNet-50 "
            "architecture on the CIFAR-10 dataset. The model achieved strong performance "
            "with minimal overfitting.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Metrics Table
    story.append(Paragraph("Performance Metrics", styles["Heading2"]))
    data = [
        ["Metric", "Training", "Validation"],
        ["Accuracy", "98.5%", "94.2%"],
        ["Loss", "0.045", "0.182"],
        ["F1-Score", "0.985", "0.941"],
        ["Precision", "0.987", "0.945"],
        ["Recall", "0.983", "0.937"],
    ]
    table = Table(data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # Hyperparameters
    story.append(Paragraph("Hyperparameters", styles["Heading2"]))
    story.append(
        Paragraph(
            "<b>Learning Rate:</b> 0.001<br/>"
            "<b>Batch Size:</b> 64<br/>"
            "<b>Optimizer:</b> Adam<br/>"
            "<b>Epochs:</b> 50<br/>"
            "<b>Weight Decay:</b> 0.0001",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Conclusions
    story.append(Paragraph("Conclusions", styles["Heading2"]))
    story.append(
        Paragraph(
            "The model demonstrates excellent performance on both training and validation sets. "
            "The validation accuracy of 94.2% indicates good generalization with acceptable "
            "overfitting levels. Recommend deployment to production for A/B testing.",
            styles["BodyText"],
        )
    )

    # Build PDF
    doc.build(story)
    return buffer.getvalue()
