"""
Helper functions for creating rich notebook entries with tables, images, and formatting
"""

import io
import json
import uuid
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


def add_ids_recursively(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively add unique IDs to all blocks and nested content blocks

    BlockNote requires IDs on ALL nodes including text nodes

    Args:
        block: Block dictionary to add IDs to

    Returns:
        Block with IDs added
    """
    # Add ID to this block if it doesn't have one (including text nodes!)
    if "id" not in block and "type" in block:
        block["id"] = str(uuid.uuid4())

    # Recursively add IDs to nested content
    if "content" in block and isinstance(block["content"], list):
        for item in block["content"]:
            if isinstance(item, dict):
                add_ids_recursively(item)

    return block


def create_notebook_entry(
    project_id: str,
    title: str,
    sections: List[Dict[str, Any]],
    api_url: str,
) -> int:
    """
    Create a rich notebook entry with formatted content

    Args:
        project_id: Project ID
        title: Note title
        sections: List of content sections (see create_*_section functions)
        api_url: API base URL

    Returns:
        Note ID
    """
    # Build BlockNote content structure as array of blocks with IDs
    blocks = []

    for section in sections:
        for block in section:
            # Recursively add IDs to this block and all nested content
            add_ids_recursively(block)
            blocks.append(block)

    # Create note
    response = requests.post(
        f"{api_url}/api/projects/{project_id}/notes",
        json={
            "title": title,
            "content": json.dumps(blocks),  # Send array of blocks, not doc object
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


# ============================================================================
# Content Section Builders
# ============================================================================


def create_heading_section(text: str, level: int = 1) -> List[Dict[str, Any]]:
    """Create a heading section"""
    return [
        {
            "type": "heading",
            "attrs": {"level": level},
            "content": [{"type": "text", "text": text}],
        }
    ]


def create_paragraph_section(text: str, bold: bool = False) -> List[Dict[str, Any]]:
    """Create a paragraph section"""
    marks = [{"type": "bold"}] if bold else []
    return [
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": text, "marks": marks}]
            if marks
            else [{"type": "text", "text": text}],
        }
    ]


def create_bullet_list_section(items: List[str]) -> List[Dict[str, Any]]:
    """Create a bullet list section"""
    list_items = []
    for item in items:
        list_items.append(
            {
                "type": "listItem",
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": item}]}],
            }
        )

    return [{"type": "bulletList", "content": list_items}]


def create_table_section(headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Create a table section

    Args:
        headers: Column headers
        rows: Data rows (list of lists)

    Returns:
        BlockNote table structure
    """
    # Build table rows
    table_rows = []

    # Header row
    header_cells = []
    for header in headers:
        header_cells.append(
            {
                "type": "tableCell",
                "attrs": {"headerCell": True},
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": header, "marks": [{"type": "bold"}]}],
                    }
                ],
            }
        )
    table_rows.append({"type": "tableRow", "content": header_cells})

    # Data rows
    for row in rows:
        cells = []
        for cell in row:
            cells.append(
                {
                    "type": "tableCell",
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": str(cell)}]}
                    ],
                }
            )
        table_rows.append({"type": "tableRow", "content": cells})

    return [{"type": "table", "content": table_rows}]


def create_code_block_section(code: str, language: str = "python") -> List[Dict[str, Any]]:
    """Create a code block section"""
    return [
        {
            "type": "codeBlock",
            "attrs": {"language": language},
            "content": [{"type": "text", "text": code}],
        }
    ]


# ============================================================================
# Demo Notebook Generators
# ============================================================================


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

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def generate_metrics_plot(
    metrics_data: Dict[str, List[float]], title: str = "Training Metrics"
) -> bytes:
    """Generate metrics line plot"""
    plt.figure(figsize=(12, 6))

    for metric_name, values in metrics_data.items():
        plt.plot(values, marker="o", label=metric_name, linewidth=2)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
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
    """
    Create a comprehensive experiment summary notebook

    Args:
        project_id: Project ID
        experiment_name: Name of experiment
        run_ids: List of run IDs (used for display only)
        metrics_summary: List of dicts with run metrics
        best_config: Best configuration found
        api_url: API base URL

    Returns:
        Note ID
    """
    sections = []

    # Title
    sections.append(create_heading_section(f"{experiment_name} - Experiment Summary", level=1))

    # Overview
    sections.append(create_heading_section("Overview", level=2))
    sections.append(
        create_paragraph_section(
            f"This notebook contains results from {len(run_ids)} experimental runs "
            f"exploring different hyperparameter configurations."
        )
    )

    # Key Findings
    sections.append(create_heading_section("Key Findings", level=2))
    findings = [
        f"Total runs completed: {len(run_ids)}",
        f"Best configuration identified: {list(best_config.keys())}",
        "All experiments logged with full provenance tracking",
    ]
    sections.append(create_bullet_list_section(findings))

    # Results Table
    sections.append(create_heading_section("Results Summary", level=2))

    if metrics_summary:
        # Extract headers from first result
        headers = ["Run ID"] + list(metrics_summary[0].keys())
        rows = []
        for i, metrics in enumerate(metrics_summary):
            row = [run_ids[i][:8]] + [
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()
            ]
            rows.append(row)

        sections.append(create_table_section(headers, rows))

    # Best Configuration
    sections.append(create_heading_section("Best Configuration", level=2))
    config_items = [f"{k}: {v}" for k, v in best_config.items()]
    sections.append(create_bullet_list_section(config_items))

    # Code snippet
    sections.append(create_heading_section("Example Usage", level=2))
    code = f"""import artifacta as ds

# Initialize run with best config
run = ds.init(config={json.dumps(best_config, indent=2)})

# Train model
model = train_model(run.config)

# Log metrics
run.log({{"accuracy": 0.95, "loss": 0.123}})
"""
    sections.append(create_code_block_section(code))

    # Create note
    note_id = create_notebook_entry(project_id, f"{experiment_name} Results", sections, api_url)

    return note_id


def create_computer_vision_notebook(
    project_id: str,
    metrics: Dict[str, float],
    api_url: str,
    confusion_data: Optional[tuple] = None,
) -> int:
    """
    Create a computer vision experiment notebook with confusion matrix

    Args:
        project_id: Project ID
        metrics: Dictionary of metric name -> value
        confusion_data: Optional tuple of (y_true, y_pred, labels)
        api_url: API base URL

    Returns:
        Note ID
    """
    sections = []

    sections.append(create_heading_section("Computer Vision Experiment", level=1))

    sections.append(create_heading_section("Model Performance", level=2))

    # Metrics table
    headers = ["Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in metrics.items()]
    sections.append(create_table_section(headers, rows))

    # Create note
    note_id = create_notebook_entry(project_id, "Computer Vision Results", sections, api_url)

    # Add confusion matrix if provided
    if confusion_data:
        y_true, y_pred, labels = confusion_data
        cm_image = generate_confusion_matrix_image(y_true, y_pred, labels)
        upload_attachment(note_id, "confusion_matrix.png", cm_image, api_url)

    return note_id
