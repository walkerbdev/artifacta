"""
Helper functions for creating rich notebook entries using Markdown
BlockNote supports Markdown natively, so this is more reliable than JSON
"""

import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


def create_notebook_entry(
    project_id: str,
    title: str,
    markdown_content: str,
    api_url: str,
) -> int:
    """
    Create a rich notebook entry with Markdown content

    Args:
        project_id: Project ID
        title: Note title
        markdown_content: Markdown formatted content
        api_url: API base URL

    Returns:
        Note ID
    """
    response = requests.post(
        f"{api_url}/api/projects/{project_id}/notes",
        json={
            "title": title,
            "content": markdown_content,
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
    metrics_summary: List[Dict[str, Any]],
    best_config: Dict[str, Any],
    api_url: str,
) -> int:
    """Create experiment summary notebook using Markdown"""

    # Build markdown content
    md_parts = []

    # Title
    md_parts.append(f"# {experiment_name} - Experiment Summary\n")

    # Overview
    md_parts.append("## Overview\n")
    md_parts.append(
        f"This notebook contains results from {len(metrics_summary)} experimental runs "
        f"exploring different hyperparameter configurations.\n"
    )

    # Key Findings
    md_parts.append("## Key Findings\n")
    md_parts.append(f"- Total runs completed: {len(metrics_summary)}\n")
    md_parts.append(f"- Best configuration identified: {list(best_config.keys())}\n")
    md_parts.append("- All experiments logged with full provenance tracking\n")

    # Results Table
    md_parts.append("## Results Summary\n")
    if metrics_summary:
        # Table header
        headers = list(metrics_summary[0].keys())
        md_parts.append("| " + " | ".join(headers) + " |\n")
        md_parts.append("|" + "---|" * len(headers) + "\n")

        # Table rows
        for metrics in enumerate(metrics_summary):
            row_values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]
            md_parts.append("| " + " | ".join(row_values) + " |\n")

    # Best Configuration
    md_parts.append("\n## Best Configuration\n")
    for k, v in best_config.items():
        md_parts.append(f"- {k}: {v}\n")

    # Mathematical Formulation
    md_parts.append("\n## Mathematical Formulation\n\n")
    md_parts.append("**Cross-Entropy Loss:**\n\n")
    md_parts.append("$$L = -\\sum_{i=1}^{C} y_i \\log(\\hat{y}_i)$$\n\n")
    md_parts.append(
        "Where C is the number of classes, $y_i$ is the true label, and $\\hat{y}_i$ is the predicted probability.\n\n"
    )

    md_parts.append("**Accuracy:**\n\n")
    md_parts.append("$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$\n\n")

    md_parts.append("**F1 Score:**\n\n")
    md_parts.append(
        "$$F1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$\n\n"
    )

    # Code Example
    md_parts.append("\n## Example Usage\n")
    md_parts.append("```python\n")
    md_parts.append("import artifacta as ds\n\n")
    md_parts.append("# Initialize run with best config\n")
    md_parts.append(f"run = ds.init(config={best_config})\n\n")
    md_parts.append("# Train model\n")
    md_parts.append("model = train_model(run.config)\n\n")
    md_parts.append("# Log metrics\n")
    md_parts.append('run.log({"accuracy": 0.95, "loss": 0.123})\n')
    md_parts.append("```\n")

    markdown_content = "".join(md_parts)

    return create_notebook_entry(
        project_id, f"{experiment_name} Results", markdown_content, api_url
    )


def create_computer_vision_notebook(
    project_id: str,
    metrics: Dict[str, float],
    api_url: str,
    confusion_data: Optional[tuple] = None,
) -> int:
    """Create computer vision notebook using Markdown"""

    md_parts = []

    md_parts.append("# Computer Vision Experiment\n")
    md_parts.append("## Model Performance\n")

    # Metrics table
    md_parts.append("| Metric | Value |\n")
    md_parts.append("|--------|-------|\n")
    for name, value in metrics.items():
        md_parts.append(f"| {name} | {value:.4f} |\n")

    # Add confusion matrix explanation with equations
    md_parts.append("\n## Confusion Matrix Analysis\n\n")
    md_parts.append(
        "The confusion matrix visualizes classification performance across all classes.\n\n"
    )

    md_parts.append("**Precision:**\n\n")
    md_parts.append("$$\\text{Precision} = \\frac{TP}{TP + FP}$$\n\n")

    md_parts.append("**Recall:**\n\n")
    md_parts.append("$$\\text{Recall} = \\frac{TP}{TP + FN}$$\n\n")

    md_parts.append("**Softmax Output:**\n\n")
    md_parts.append("$$\\hat{y}_i = \\frac{e^{z_i}}{\\sum_{j=1}^{C} e^{z_j}}$$\n\n")

    markdown_content = "".join(md_parts)

    # Create note
    note_id = create_notebook_entry(
        project_id, "Computer Vision Results", markdown_content, api_url
    )

    # Add confusion matrix if provided
    if confusion_data:
        y_true, y_pred, labels = confusion_data
        cm_image = generate_confusion_matrix_image(y_true, y_pred, labels)
        upload_attachment(note_id, "confusion_matrix.png", cm_image, api_url)

    return note_id
