"""
Helper functions for creating rich notebook entries using BlockNote's native JSON format
This allows us to use custom blocks like math blocks with LaTeX
"""

import io
import json
import uuid
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


def get_default_props():
    """Get BlockNote default props for blocks"""
    return {"backgroundColor": "default", "textColor": "default", "textAlignment": "left"}


def create_block(block_type: str, content=None, props=None, children=None):
    """Create a BlockNote block with proper structure"""
    # Merge provided props with defaults
    default_props = get_default_props()
    merged_props = {**default_props, **(props or {})}

    block = {
        "id": str(uuid.uuid4()),
        "type": block_type,
        "props": merged_props,
        "content": content if content is not None else [],
        "children": children or [],
    }
    return block


def create_text_content(text: str, styles=None):
    """Create inline text content"""
    return {"type": "text", "text": text, "styles": styles or {}}


def create_heading(text: str, level: int = 1):
    """Create a heading block"""
    return create_block("heading", [create_text_content(text)], {"level": level})


def create_paragraph(text: str, bold: bool = False):
    """Create a paragraph block"""
    styles = {"bold": True} if bold else {}
    return create_block("paragraph", [create_text_content(text, styles)])


def create_bullet_list_item(text: str, children=None):
    """Create a bullet list item"""
    return create_block("bulletListItem", [create_text_content(text)], {}, children)


def create_table(headers: List[str], rows: List[List[str]]):
    """Create a table block"""
    table_content = {"type": "tableContent", "rows": []}

    # Header row
    header_cells = []
    for header in headers:
        header_cells.append([create_text_content(header, {"bold": True})])
    table_content["rows"].append({"cells": header_cells})

    # Data rows
    for row in rows:
        cells = [[create_text_content(str(cell))] for cell in row]
        table_content["rows"].append({"cells": cells})

    return create_block("table", table_content)


def create_code_block(code: str, language: str = "python"):
    """Create a code block"""
    return create_block("codeBlock", [create_text_content(code)], {"language": language})


def create_math_block(latex: str):
    """Create a paragraph with LaTeX that will be rendered on the frontend"""
    # Use a paragraph block with the LaTeX wrapped in $$
    return create_paragraph(latex)


def create_notebook_entry(
    project_id: str,
    title: str,
    blocks: List[Dict[str, Any]],
    api_url: str,
) -> int:
    """Create a notebook entry with BlockNote JSON blocks"""
    response = requests.post(
        f"{api_url}/api/projects/{project_id}/notes",
        json={
            "title": title,
            "content": json.dumps(blocks),
        },
    )
    response.raise_for_status()
    return response.json()["id"]


def upload_attachment(
    note_id: int, filename: str, file_content: bytes, api_url: str
) -> Dict[str, Any]:
    """Upload file attachment to a note"""
    response = requests.post(
        f"{api_url}/api/notes/{note_id}/attachments",
        files={"file": (filename, file_content)},
    )
    response.raise_for_status()
    return response.json()


def generate_confusion_matrix_image(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> bytes:
    """Generate confusion matrix heatmap image"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def create_experiment_summary_notebook(
    project_id: str,
    experiment_name: str,
    run_ids: List[str],
    metrics_summary: List[Dict[str, Any]],
    best_config: Dict[str, Any],
    api_url: str,
) -> int:
    """Create experiment summary notebook with LaTeX equations"""

    blocks = []

    # Title
    blocks.append(create_heading(f"{experiment_name} - Experiment Summary", 1))

    # Overview
    blocks.append(create_heading("Overview", 2))
    blocks.append(
        create_paragraph(
            f"This notebook contains results from {len(run_ids)} experimental runs "
            f"exploring different hyperparameter configurations."
        )
    )

    # Key Findings
    blocks.append(create_heading("Key Findings", 2))
    blocks.append(create_bullet_list_item(f"Total runs completed: {len(run_ids)}"))
    blocks.append(
        create_bullet_list_item(f"Best configuration identified: {list(best_config.keys())}")
    )
    blocks.append(create_bullet_list_item("All experiments logged with full provenance tracking"))

    # Results Table
    blocks.append(create_heading("Results Summary", 2))
    if metrics_summary:
        headers = ["Run ID"] + list(metrics_summary[0].keys())
        rows = []
        for i, metrics in enumerate(metrics_summary):
            row = [run_ids[i][:8]] + [
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()
            ]
            rows.append(row)
        blocks.append(create_table(headers, rows))

    # Best Configuration
    blocks.append(create_heading("Best Configuration", 2))
    for k, v in best_config.items():
        blocks.append(create_bullet_list_item(f"{k}: {v}"))

    # Mathematical Formulation
    blocks.append(create_heading("Mathematical Formulation", 2))

    blocks.append(create_paragraph("Cross-Entropy Loss:", bold=True))
    blocks.append(create_math_block(r"$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$"))
    blocks.append(
        create_paragraph(
            "Where C is the number of classes, y_i is the true label, and ŷ_i is the predicted probability."
        )
    )

    blocks.append(create_paragraph("Accuracy:", bold=True))
    blocks.append(create_math_block(r"$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$"))

    blocks.append(create_paragraph("F1 Score:", bold=True))
    blocks.append(
        create_math_block(
            r"$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$"
        )
    )

    # Code Example
    blocks.append(create_heading("Example Usage", 2))
    code = f"""import artifacta as ds

# Initialize run with best config
run = ds.init(config={best_config})

# Train model
model = train_model(run.config)

# Log metrics
run.log({{"accuracy": 0.95, "loss": 0.123}})"""
    blocks.append(create_code_block(code))

    return create_notebook_entry(project_id, f"{experiment_name} Results", blocks, run_ids, api_url)


def create_computer_vision_notebook(
    project_id: str,
    run_ids: List[str],
    metrics: Dict[str, float],
    api_url: str,
    confusion_data: Optional[tuple] = None,
) -> int:
    """Create computer vision notebook with LaTeX equations"""

    blocks = []

    blocks.append(create_heading("Computer Vision Experiment", 1))
    blocks.append(create_heading("Model Performance", 2))

    # Metrics table
    headers = ["Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in metrics.items()]
    blocks.append(create_table(headers, rows))

    # Confusion Matrix Analysis
    blocks.append(create_heading("Confusion Matrix Analysis", 2))
    blocks.append(
        create_paragraph(
            "The confusion matrix visualizes classification performance across all classes."
        )
    )

    blocks.append(create_paragraph("Precision:", bold=True))
    blocks.append(create_math_block(r"$$\text{Precision} = \frac{TP}{TP + FP}$$"))

    blocks.append(create_paragraph("Recall:", bold=True))
    blocks.append(create_math_block(r"$$\text{Recall} = \frac{TP}{TP + FN}$$"))

    blocks.append(create_paragraph("Softmax Output:", bold=True))
    blocks.append(create_math_block(r"$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$"))

    # Create note
    note_id = create_notebook_entry(project_id, "Computer Vision Results", blocks, run_ids, api_url)

    # Add confusion matrix if provided
    if confusion_data:
        y_true, y_pred, labels = confusion_data
        cm_image = generate_confusion_matrix_image(y_true, y_pred, labels)
        upload_attachment(note_id, "confusion_matrix.png", cm_image, api_url)

    return note_id
