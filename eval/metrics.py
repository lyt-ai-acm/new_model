"""
Evaluation metrics for three-class sentiment analysis.
"""
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


LABELS = ["neg", "neu", "pos"]
LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Compute Accuracy, Macro-F1, and per-class P/R/F1.
    
    Args:
        y_true: ground truth labels (strings: neg/neu/pos)
        y_pred: predicted labels (strings: neg/neu/pos)
    
    Returns:
        dict with accuracy, macro_f1, and per-class metrics
    """
    y_true_ids = [LABEL2ID[l] for l in y_true]
    y_pred_ids = [LABEL2ID[l] for l in y_pred]
    
    acc = accuracy_score(y_true_ids, y_pred_ids)
    macro_f1 = f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
    
    per_class = {}
    for label in LABELS:
        lid = LABEL2ID[label]
        y_t_bin = [1 if y == lid else 0 for y in y_true_ids]
        y_p_bin = [1 if y == lid else 0 for y in y_pred_ids]
        per_class[label] = {
            "precision": precision_score(y_t_bin, y_p_bin, zero_division=0),
            "recall": recall_score(y_t_bin, y_p_bin, zero_division=0),
            "f1": f1_score(y_t_bin, y_p_bin, zero_division=0),
        }
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "report": classification_report(y_true_ids, y_pred_ids, target_names=LABELS, zero_division=0),
    }


def print_metrics(metrics: Dict, experiment_name: str = ""):
    if experiment_name:
        print(f"\n=== {experiment_name} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    for label, m in metrics["per_class"].items():
        print(f"  [{label}] P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])
