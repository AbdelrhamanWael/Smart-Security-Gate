"""
evaluation.py - Biometric Evaluation & Metrics Engine
Part of: Hierarchical Multimodal Biometric Verification System

Interface Contract (Section 4):
evaluate(y_true, y_score, threshold=0.5) -> dict
  acc, prec, rec, F1, FAR, FRR, TAR, EER, eer_threshold, AUC

Also provides helper functions for generating:
- Confusion Matrix plots
- ROC curves with EER point marked
- Genuine vs. Impostor score distribution plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Equal Error Rate (EER) and the optimal EER threshold
    where FAR equals FRR.

    Returns:
      eer: float (in [0, 1])
      eer_threshold: float
      fpr (FAR curve): np.ndarray
      fnr (FRR curve): np.ndarray
      thresholds: np.ndarray
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr  # FRR

    # Find the threshold where FAR and FRR are closest
    diff = np.abs(fpr - fnr)
    idx = np.nanargmin(diff)

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold, fpr, fnr, thresholds


def evaluate(
    y_true: Union[list, np.ndarray],
    y_score: Union[list, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, list]]:
    """
    Standard interface contract:
    evaluate(y_true, y_score, threshold) -> dict

    Computes:
      - TP, FP, TN, FN
      - Accuracy, Precision, Recall, F1
      - FAR (False Acceptance Rate) = FP / (FP + TN)
      - FRR (False Rejection Rate) = FN / (FN + TP)
      - TAR (True Acceptance Rate) = 1 - FRR
      - AUC (Area Under ROC Curve)
      - EER (Equal Error Rate) and eer_threshold
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Hard decisions based on specified threshold
    y_pred = (y_score >= threshold).astype(int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Biometric security metrics
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    tar = float(1.0 - frr)

    # ROC AUC and EER
    if len(np.unique(y_true)) > 1:
        auc_val = float(roc_auc_score(y_true, y_score))
        eer, eer_threshold, _, _, _ = compute_eer(y_true, y_score)
    else:
        auc_val = 1.0
        eer = 0.0
        eer_threshold = threshold

    results = {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "FAR": far,
        "FRR": frr,
        "TAR": tar,
        "AUC": auc_val,
        "EER": eer,
        "eer_threshold": eer_threshold,
    }
    return results


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate and save ROC curve with EER point marked.
    """
    eer, eer_thresh, fpr, fnr, _ = compute_eer(y_true, y_score)
    tpr = 1.0 - fnr
    auc_val = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Chance")

    # Mark EER point (FAR = EER, TPR = 1 - EER)
    ax.scatter([eer], [1.0 - eer], color="red", s=70, zorder=5,
               label=f"EER = {eer:.2%} (T={eer_thresh:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=11)
    ax.set_ylabel("True Acceptance Rate (TAR = 1 - FRR)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate and save formatted confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:d}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float = 0.5,
    title: str = "Match Score Distributions",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot genuine vs. impostor score histograms with decision threshold.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    bins = np.linspace(0.0, 1.0, 30)
    ax.hist(genuine_scores, bins=bins, alpha=0.6, color="green", label="Genuine", density=True)
    ax.hist(impostor_scores, bins=bins, alpha=0.6, color="red", label="Impostor", density=True)

    ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Verification Match Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig, ax



# ---------------------------------------------------------------------------
# Threshold selection helpers (Member 3) - used to FREEZE thresholds on
# validation data only, then apply them unchanged on the held-out test set.
# ---------------------------------------------------------------------------

def threshold_at_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold where FAR ~= FRR (computed on the given validation scores)."""
    return compute_eer(y_true, y_score)[1]


def threshold_at_far(y_true: np.ndarray, y_score: np.ndarray, target_far: float = 0.01) -> float:
    """
    Lowest threshold whose FAR <= target_far (i.e. the most convenient
    operating point that still satisfies the security requirement).
    """
    fpr, tpr, thr = roc_curve(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), pos_label=1)
    ok = np.where(fpr <= target_far)[0]
    idx = ok[-1] if len(ok) else 0  # highest TPR among points that satisfy FAR budget
    t = float(thr[idx])
    return t if np.isfinite(t) else float(np.max(y_score)) + 1e-6


def threshold_at_frr(y_true: np.ndarray, y_score: np.ndarray, target_frr: float = 0.02) -> float:
    """
    Highest threshold whose FRR <= target_frr. Used for a lenient Level-1
    gate: almost no genuine user is rejected early, the later levels do the
    heavy lifting against impostors.
    """
    fpr, tpr, thr = roc_curve(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), pos_label=1)
    fnr = 1.0 - tpr
    ok = np.where(fnr <= target_frr)[0]
    idx = ok[0] if len(ok) else len(thr) - 1  # lowest FAR among points that satisfy FRR budget
    t = float(thr[idx])
    return t if np.isfinite(t) else float(np.max(y_score)) + 1e-6


def plot_roc_comparison(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay ROC curves of several systems: {name: (y_true, y_score)}."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for name, (y, s) in curves.items():
        y = np.asarray(y, dtype=int)
        s = np.asarray(s, dtype=float)
        if len(np.unique(y)) < 2:
            continue
        eer, _, fpr, fnr, _ = compute_eer(y, s)
        auc_val = roc_auc_score(y, s)
        ax.plot(fpr, 1.0 - fnr, lw=2, label=f"{name} (AUC={auc_val:.3f}, EER={eer:.1%})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=11)
    ax.set_ylabel("True Acceptance Rate (TAR = 1 - FRR)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig, ax


def confusion_from_metrics(m: Dict[str, Union[float, int]]) -> np.ndarray:
    """2x2 matrix [[TN, FP], [FN, TP]] from an evaluate() result."""
    return np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
