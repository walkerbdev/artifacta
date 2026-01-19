"""ML Metrics - ROC and Precision-Recall Curves"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def train_and_evaluate(X_train, y_train, X_test, y_test):  # noqa: N803
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Get predictions
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }
