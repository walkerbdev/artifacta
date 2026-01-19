"""Evaluation metrics."""

from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, class_names=None):
    """Compute classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def top_k_accuracy(outputs, targets, k=5):
    """Compute top-k accuracy."""
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / targets.size(0)).item()
