"""
Visualization Module
"""

from sklearn.metrics import (
    PrecisionRecallDisplay,
)

def display_precision_recall_curve(
    model,
    X_test,
    y_test,
    title = "Precision-Recall Curve"
):
    display = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test)
    _ = display.ax_.set_title(title)