"""
Helper functions for creating rich notebook entries using HTML format for TipTap editor
This allows us to use LaTeX with the @aarkue/tiptap-math-extension
"""

import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns


def create_heading(text: str, level: int = 1) -> str:
    """Create an HTML heading"""
    return f"<h{level}>{text}</h{level}>"


def create_paragraph(text: str, bold: bool = False) -> str:
    """Create an HTML paragraph"""
    if bold:
        text = f"<strong>{text}</strong>"
    return f"<p>{text}</p>"


def create_bullet_list(items: List[str]) -> str:
    """Create an HTML bullet list"""
    list_items = "".join([f"<li>{item}</li>" for item in items])
    return f"<ul>{list_items}</ul>"


def create_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create an HTML table"""
    header_cells = "".join([f"<th>{h}</th>" for h in headers])
    header_row = f"<tr>{header_cells}</tr>"

    body_rows = []
    for row in rows:
        cells = "".join([f"<td>{str(cell)}</td>" for cell in row])
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""<table>
<thead>{header_row}</thead>
<tbody>{"".join(body_rows)}</tbody>
</table>"""


def create_code_block(code: str, language: str = "python") -> str:
    """Create an HTML code block"""
    # TipTap uses <pre><code> for code blocks
    return f'<pre><code class="language-{language}">{code}</code></pre>'


def create_blockquote(text: str) -> str:
    """Create a blockquote"""
    return f"<blockquote><p>{text}</p></blockquote>"


def create_horizontal_rule() -> str:
    """Create a horizontal rule separator"""
    return "<hr>"


def create_link(text: str, url: str) -> str:
    """Create a hyperlink"""
    return f'<a href="{url}">{text}</a>'


def create_inline_code(text: str) -> str:
    """Create inline code"""
    return f"<code>{text}</code>"


def create_strikethrough(text: str) -> str:
    """Create strikethrough text"""
    return f"<s>{text}</s>"


def create_math_inline(latex: str) -> str:
    """Create inline LaTeX math using TipTap math node format"""
    return f'<span data-type="inlineMath" data-latex="{latex}" data-display="no" data-evaluate="no"></span>'


def create_math_display(latex: str) -> str:
    """Create display LaTeX math using TipTap math node format"""
    return f'<p><span data-type="inlineMath" data-latex="{latex}" data-display="yes" data-evaluate="no"></span></p>'


def create_notebook_entry(
    project_id: str,
    title: str,
    html_content: str,
    api_url: str,
) -> int:
    """Create a notebook entry with HTML content using artifacta SDK"""
    import artifacta as ds

    # Get current run or create a temporary emitter to create note
    run = ds.get_run()
    if run:
        return run.create_note(title, html_content)
    else:
        # No active run - create note via emitter directly
        from artifacta.emitter import HTTPEmitter

        emitter = HTTPEmitter("temp")
        return emitter.emit_note(project_id, title, html_content)


def upload_attachment(
    note_id: int,
    filename: str,
    file_content: bytes,
    api_url: str,
    mime_type: str = None,
) -> Dict[str, Any]:
    """Upload file attachment to a note"""
    import mimetypes

    # Guess MIME type from filename if not provided
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

    response = requests.post(
        f"{api_url}/api/notes/{note_id}/attachments",
        files={"file": (filename, file_content, mime_type)},
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
    """Create experiment summary notebook with LaTeX equations"""

    html_parts = []

    # Title
    html_parts.append(create_heading(f"{experiment_name} - Experiment Summary", 1))

    # Overview
    html_parts.append(create_heading("Overview", 2))
    html_parts.append(
        create_paragraph(
            f"This notebook contains results from {len(metrics_summary)} experimental runs "
            f"exploring different hyperparameter configurations."
        )
    )

    # Key Findings
    html_parts.append(create_heading("Key Findings", 2))
    html_parts.append(
        create_bullet_list(
            [
                f"Total runs completed: {len(metrics_summary)}",
                f"Best configuration identified: {list(best_config.keys())}",
                "All experiments logged with full provenance tracking",
            ]
        )
    )

    # Results Table
    html_parts.append(create_heading("Results Summary", 2))
    if metrics_summary:
        headers = ["Run"] + list(metrics_summary[0].keys())
        rows = []
        for i, metrics in enumerate(metrics_summary):
            row = [f"Run {i + 1}"] + [
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()
            ]
            rows.append(row)
        html_parts.append(create_table(headers, rows))

    # Best Configuration
    html_parts.append(create_heading("Best Configuration", 2))
    html_parts.append(create_bullet_list([f"{k}: {v}" for k, v in best_config.items()]))

    # Mathematical Formulation
    html_parts.append(create_heading("Mathematical Formulation", 2))

    html_parts.append(
        create_blockquote(
            "Understanding the mathematical foundations helps us interpret model behavior and make informed decisions about hyperparameter tuning."
        )
    )

    html_parts.append(create_paragraph("Cross-Entropy Loss:", bold=True))
    html_parts.append(create_math_display(r"L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)"))
    html_parts.append(
        create_paragraph(
            f"Where {create_inline_code('C')} is the number of classes, "
            f"{create_inline_code('y_i')} is the true label, and "
            f"{create_inline_code('ŷ_i')} is the predicted probability."
        )
    )

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_paragraph("Accuracy:", bold=True))
    html_parts.append(create_math_display(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}"))
    html_parts.append(
        create_paragraph(
            f"Accuracy can be misleading with imbalanced datasets. "
            f"See {create_link('this article', 'https://developers.google.com/machine-learning/crash-course/classification/accuracy')} for more details."
        )
    )

    html_parts.append(create_paragraph("F1 Score:", bold=True))
    html_parts.append(
        create_math_display(
            r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}"
        )
    )

    # Code Example
    html_parts.append(create_heading("Example Usage", 2))
    code = f"""import artifacta as ds

# Initialize run with best config
run = ds.init(config={best_config})

# Train model
model = train_model(run.config)

# Log metrics
run.log({{"accuracy": 0.95, "loss": 0.123}})"""
    html_parts.append(create_code_block(code))

    html_content = "\n".join(html_parts)
    return create_notebook_entry(project_id, f"{experiment_name} Results", html_content, api_url)


def create_computer_vision_notebook(
    project_id: str,
    metrics: Dict[str, float],
    api_url: str,
    confusion_data: Optional[tuple] = None,
) -> int:
    """Create computer vision notebook with LaTeX equations"""

    html_parts = []

    html_parts.append(create_heading("Computer Vision Experiment", 1))

    html_parts.append(
        create_blockquote(
            "This notebook documents our ResNet50 training on CIFAR-10 dataset, "
            "including detailed performance metrics and confusion matrix analysis."
        )
    )

    html_parts.append(create_heading("Model Performance", 2))

    # Metrics table
    headers = ["Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in metrics.items()]
    html_parts.append(create_table(headers, rows))

    html_parts.append(create_horizontal_rule())

    # Confusion Matrix Analysis
    html_parts.append(create_heading("Confusion Matrix Analysis", 2))
    html_parts.append(
        create_paragraph(
            f"The confusion matrix visualizes classification performance across all classes. "
            f"{create_strikethrough('Simple accuracy alone')} Using multiple metrics provides better insight."
        )
    )

    html_parts.append(create_paragraph("Precision:", bold=True))
    html_parts.append(create_math_display(r"\text{Precision} = \frac{TP}{TP + FP}"))
    html_parts.append(
        create_paragraph(
            f"High precision means few {create_inline_code('false positives')} - "
            f"when the model predicts a class, it's usually correct."
        )
    )

    html_parts.append(create_paragraph("Recall:", bold=True))
    html_parts.append(create_math_display(r"\text{Recall} = \frac{TP}{TP + FN}"))
    html_parts.append(
        create_paragraph(
            f"High recall means few {create_inline_code('false negatives')} - "
            f"the model finds most instances of each class."
        )
    )

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Model Architecture", 2))
    html_parts.append(create_paragraph("Softmax Output Layer:", bold=True))
    html_parts.append(create_math_display(r"\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}"))
    html_parts.append(
        create_paragraph(
            f"The softmax function converts raw logits into probabilities. "
            f"Learn more in the {create_link('PyTorch docs', 'https://pytorch.org/docs/stable/nn.html#softmax')}."
        )
    )

    html_parts.append(create_heading("Next Steps", 2))
    html_parts.append(
        create_bullet_list(
            [
                f"Experiment with data augmentation to improve {create_inline_code('test_accuracy')}",
                f"Try learning rate scheduling {create_strikethrough('(constant LR performed poorly)')}",
                "Analyze per-class performance using the confusion matrix attachment below",
            ]
        )
    )

    html_parts.append(create_heading("Supporting Materials", 2))
    html_parts.append(create_paragraph("Additional files attached to this notebook include:"))
    html_parts.append(
        create_bullet_list(
            [
                "Confusion Matrix visualization (PNG)",
                "Detailed training report (PDF)",
                "Model training progress audio notification (WAV)",
                "Training animation video (MP4)",
            ]
        )
    )

    html_content = "\n".join(html_parts)

    # Create note
    note_id = create_notebook_entry(project_id, "Computer Vision Results", html_content, api_url)

    # Add confusion matrix if provided (only if note_id was successfully created)
    if note_id and confusion_data:
        y_true, y_pred, labels = confusion_data
        cm_image = generate_confusion_matrix_image(y_true, y_pred, labels)
        upload_attachment(note_id, "confusion_matrix.png", cm_image, api_url)

    # Add PDF report (only if note_id was successfully created)
    if note_id:
        try:
            from tests.helpers.pdf import create_test_pdf_report

            pdf_content = create_test_pdf_report("ResNet-50 Training Report")
            upload_attachment(note_id, "training_report.pdf", pdf_content, api_url)
        except Exception as e:
            print(f"Failed to generate PDF: {e}")

    # Add audio file (only if note_id was successfully created)
    if note_id:
        try:
            from tests.helpers.audio import create_test_audio_tone

            audio_path = create_test_audio_tone(
                frequency=440, duration=1.0, filename="training_complete.wav"
            )
            with open(audio_path, "rb") as f:
                audio_content = f.read()
            upload_attachment(note_id, "training_complete.wav", audio_content, api_url)
        except Exception as e:
            print(f"Failed to generate audio: {e}")

    # Add video file (only if note_id was successfully created)
    if note_id:
        try:
            from tests.helpers.video import create_test_video_animation

            video_path = create_test_video_animation(
                width=320, height=240, duration_seconds=2, fps=10, filename="training_animation.mp4"
            )
            with open(video_path, "rb") as f:
                video_content = f.read()
            upload_attachment(note_id, "training_animation.mp4", video_content, api_url)
        except Exception as e:
            print(f"Failed to generate video: {e}")

    return note_id


def create_ab_testing_notebook(
    project_id: str,
    variant_a_metrics: Dict[str, float],
    variant_b_metrics: Dict[str, float],
    api_url: str,
) -> int:
    """Create A/B testing notebook with statistical analysis"""
    html_parts = []

    html_parts.append(create_heading("A/B Testing Experiment", 1))
    html_parts.append(
        create_blockquote(
            "Statistical comparison of two product variants to determine which performs better. "
            "Always validate with proper hypothesis testing before making decisions."
        )
    )

    html_parts.append(create_heading("Variant Comparison", 2))
    headers = ["Metric", "Variant A", "Variant B", "Δ"]
    rows = []
    for metric in variant_a_metrics:
        a_val = variant_a_metrics[metric]
        b_val = variant_b_metrics[metric]
        delta = ((b_val - a_val) / a_val * 100) if a_val != 0 else 0
        rows.append([metric, f"{a_val:.4f}", f"{b_val:.4f}", f"{delta:+.2f}%"])
    html_parts.append(create_table(headers, rows))

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Statistical Significance", 2))
    html_parts.append(create_paragraph("T-Test Statistic:", bold=True))
    html_parts.append(
        create_math_display(
            r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}"
        )
    )
    html_parts.append(
        create_paragraph(
            f"Use a t-test when comparing means between two groups. "
            f"See {create_link('this guide', 'https://en.wikipedia.org/wiki/Student%27s_t-test')} for details."
        )
    )

    html_parts.append(create_paragraph("P-Value Interpretation:", bold=True))
    html_parts.append(
        create_bullet_list(
            [
                f"{create_inline_code('p < 0.01')}: Highly significant - strong evidence for difference",
                f"{create_inline_code('p < 0.05')}: Significant - moderate evidence for difference",
                f"{create_inline_code('p >= 0.05')}: Not significant {create_strikethrough('(but may still be practically important)')}",
            ]
        )
    )

    html_parts.append(create_heading("Recommendation", 2))
    html_parts.append(
        create_paragraph(
            f"Based on the results, Variant B shows improvement in key metrics. "
            f"Consider running for {create_inline_code('2-3 more weeks')} to ensure stability."
        )
    )

    html_content = "\n".join(html_parts)
    return create_notebook_entry(project_id, "A/B Test Results", html_content, api_url)


def create_finance_notebook(
    project_id: str,
    portfolio_metrics: Dict[str, float],
    api_url: str,
) -> int:
    """Create finance/portfolio analysis notebook"""
    html_parts = []

    html_parts.append(create_heading("Portfolio Risk Analysis", 1))
    html_parts.append(
        create_blockquote(
            "Quantitative analysis of portfolio performance and risk metrics using Modern Portfolio Theory."
        )
    )

    html_parts.append(create_heading("Performance Metrics", 2))
    headers = ["Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in portfolio_metrics.items()]
    html_parts.append(create_table(headers, rows))

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Risk Metrics", 2))
    html_parts.append(create_paragraph("Sharpe Ratio:", bold=True))
    html_parts.append(create_math_display(r"S = \frac{R_p - R_f}{\sigma_p}"))
    html_parts.append(
        create_paragraph(
            f"Where {create_inline_code('R_p')} is portfolio return, "
            f"{create_inline_code('R_f')} is risk-free rate, and "
            f"{create_inline_code('σ_p')} is portfolio standard deviation."
        )
    )

    html_parts.append(create_paragraph("Value at Risk (VaR):", bold=True))
    html_parts.append(
        create_math_display(r"VaR_{\alpha} = -\inf\{x \in \mathbb{R} : P(X \leq x) > \alpha\}")
    )
    html_parts.append(
        create_paragraph(
            f"VaR estimates maximum loss over a given time period at a confidence level. "
            f"Learn more: {create_link('Investopedia VaR', 'https://www.investopedia.com/terms/v/var.asp')}"
        )
    )

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Next Steps", 2))
    html_parts.append(
        create_bullet_list(
            [
                "Rebalance portfolio to target allocation",
                f"Monitor {create_inline_code('drawdown')} during volatile periods",
                f"{create_strikethrough('Ignore daily fluctuations')} Focus on long-term trends",
            ]
        )
    )

    html_content = "\n".join(html_parts)
    return create_notebook_entry(project_id, "Portfolio Analysis", html_content, api_url)


def create_genomics_notebook(
    project_id: str,
    sequence_stats: Dict[str, Any],
    api_url: str,
) -> int:
    """Create genomics analysis notebook"""
    html_parts = []

    html_parts.append(create_heading("Genomic Sequence Analysis", 1))
    html_parts.append(
        create_blockquote(
            "DNA sequence analysis using bioinformatics algorithms for gene prediction and functional annotation."
        )
    )

    html_parts.append(create_heading("Sequence Statistics", 2))
    headers = ["Property", "Value"]
    rows = [[name, str(value)] for name, value in sequence_stats.items()]
    html_parts.append(create_table(headers, rows))

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Alignment Scoring", 2))
    html_parts.append(create_paragraph("Needleman-Wunsch Algorithm:", bold=True))
    html_parts.append(
        create_math_display(
            r"F(i,j) = \max\begin{cases}F(i-1,j-1) + s(x_i, y_j)\\F(i-1,j) + d\\F(i,j-1) + d\end{cases}"
        )
    )
    html_parts.append(
        create_paragraph(
            f"Dynamic programming for global sequence alignment. "
            f"{create_inline_code('s(x,y)')} is substitution score, {create_inline_code('d')} is gap penalty."
        )
    )

    html_parts.append(create_paragraph("GC Content:", bold=True))
    html_parts.append(create_math_display(r"GC\% = \frac{G + C}{A + T + G + C} \times 100"))
    html_parts.append(
        create_paragraph(
            f"Higher GC content indicates stronger thermal stability. "
            f"Reference: {create_link('NCBI Guide', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2909565/')}"
        )
    )

    html_parts.append(create_heading("Key Findings", 2))
    html_parts.append(
        create_bullet_list(
            [
                "Identified conserved domains in target region",
                f"GC content: {create_inline_code('45.2%')} (typical for human genome)",
                f"{create_strikethrough('Initial alignment failed')} Optimized gap penalty improved results",
            ]
        )
    )

    html_content = "\n".join(html_parts)
    return create_notebook_entry(project_id, "Genomics Analysis", html_content, api_url)


def create_climate_notebook(
    project_id: str,
    climate_metrics: Dict[str, float],
    api_url: str,
) -> int:
    """Create climate modeling notebook"""
    html_parts = []

    html_parts.append(create_heading("Climate Model Simulation", 1))
    html_parts.append(
        create_blockquote(
            "Physical climate model analyzing temperature trends and atmospheric CO₂ concentrations using numerical methods."
        )
    )

    html_parts.append(create_heading("Simulation Results", 2))
    headers = ["Parameter", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in climate_metrics.items()]
    html_parts.append(create_table(headers, rows))

    html_parts.append(create_horizontal_rule())

    html_parts.append(create_heading("Physical Models", 2))
    html_parts.append(create_paragraph("Stefan-Boltzmann Law:", bold=True))
    html_parts.append(create_math_display(r"E = \sigma T^4"))
    html_parts.append(
        create_paragraph(
            f"Relates radiative energy to temperature. "
            f"{create_inline_code('σ = 5.67×10⁻⁸ W·m⁻²·K⁻⁴')} is the Stefan-Boltzmann constant."
        )
    )

    html_parts.append(create_paragraph("CO₂ Forcing:", bold=True))
    html_parts.append(create_math_display(r"\Delta F = 5.35 \ln\left(\frac{C}{C_0}\right)"))
    html_parts.append(
        create_paragraph(
            f"Radiative forcing from CO₂ concentration changes. "
            f"More details: {create_link('IPCC Report', 'https://www.ipcc.ch/report/ar6/wg1/')}"
        )
    )

    html_parts.append(create_heading("Predictions", 2))
    html_parts.append(
        create_bullet_list(
            [
                f"Temperature increase: {create_inline_code('+2.1°C')} by 2100 (RCP 4.5 scenario)",
                "Sea level rise strongly correlated with thermal expansion",
                f"{create_strikethrough('Linear extrapolation')} Non-linear feedback loops included",
            ]
        )
    )

    html_content = "\n".join(html_parts)
    return create_notebook_entry(project_id, "Climate Simulation", html_content, api_url)
