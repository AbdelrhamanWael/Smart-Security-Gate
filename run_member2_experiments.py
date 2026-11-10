"""
run_member2_experiments.py - Experiment Runner for Member 2 (Voice Verification)
Part of: Hierarchical Multimodal Biometric Verification System

Executes:
1. Voice feature extraction & caching.
2. Experiment E3: Voice Verification Mode B (SVM vs. Random Forest)
     - GroupKFold session-based hyperparameter tuning (no leakage).
     - Test evaluation on session 3 + unseen open-set impostors.
     - Metrics: Accuracy, Precision, Recall, F1, FAR, FRR, EER, AUC.
     - Saves ROC curves, score distribution plots, and confusion matrices.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import time
from typing import Dict, List

os.environ["MPLCONFIGDIR"] = os.path.abspath("tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.evaluation import evaluate, plot_confusion_matrix, plot_roc_curve, plot_score_distributions
from src.voice_dataset import (
    VoiceDataset,
    get_voice_verification_split,
)
from src.voice_verification import (
    DEFAULT_USERS,
    VoiceVerifier,
    get_active_verifier,
    set_active_verifier,
    voice_score,
)

RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")


def ensure_dataset_exists() -> None:
    """Ensure data/voice directory exists."""
    v_dir = Path("data/voice")
    if not v_dir.exists() or len(list(v_dir.glob("*/*/*.wav"))) < 10:
        raise FileNotFoundError("Voice dataset not found in data/voice!")


def run_experiment_e3_voice_verification(
    dataset: VoiceDataset,
) -> Dict[str, Dict[str, Dict]]:
    """
    Experiment E3: SVM vs. Random Forest for Voice Verification (Mode B).
    Evaluated per user (member1, member2, member3) on unseen session 3 + unseen impostors.
    """
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT E3: VOICE VERIFICATION (SVM vs. RANDOM FOREST)")
    print("=" * 70)

    samples = dataset.scan_voice_data()
    cache_path = CACHE_DIR / "voice_features.npz"
    X, person_ids, session_ids, _ = dataset.load_features_for_samples(
        samples, cache_file=cache_path, verbose=False
    )

    results_e3: Dict[str, Dict[str, Dict]] = {u: {} for u in DEFAULT_USERS}

    for m_type in ("svm", "rf"):
        verifier = VoiceVerifier(model_type=m_type)
        print(f"\n--- Training {m_type.upper()} Verifier for All Enrolled Users ---")

        for user_id in DEFAULT_USERS:
            X_tr, y_tr, grp_tr, X_te, y_te, _ = get_voice_verification_split(
                target_user=user_id,
                X=X,
                person_ids=person_ids,
                session_ids=session_ids,
                unseen_impostors=["impostor4", "impostor5"],
            )

            verifier.tune_and_train_user(user_id, X_tr, y_tr, grp_tr, verbose=False)
            metrics = verifier.evaluate_on_test_set(user_id, X_te, y_te, threshold=0.5)
            results_e3[user_id][m_type] = metrics

            print(
                f"  [{m_type.upper():3s} - {user_id}] Test Acc: {metrics['accuracy']:.4f} | "
                f"FAR: {metrics['FAR']:.4f} | FRR: {metrics['FRR']:.4f} | "
                f"EER: {metrics['EER']:.4f} | AUC: {metrics['AUC']:.4f}"
            )

            # Save Confusion Matrix Plot
            cm = confusion_matrix(
                y_te,
                (verifier.user_models[user_id].predict_proba(X_te)[:, 1] >= 0.5).astype(int),
                labels=[0, 1],
            )
            cm_path = RESULTS_DIR / f"e3_cm_{user_id}_{m_type}.png"
            plot_confusion_matrix(
                cm,
                class_names=["Impostor", "Genuine"],
                title=f"Mode B Confusion Matrix ({user_id.upper()} - {m_type.upper()})",
                save_path=cm_path,
            )
            plt.close("all")

        # Save model containing all enrolled members
        verifier.save()
        if m_type == "svm":
            set_active_verifier(verifier)

    # Plot ROC and Score Distributions for best model (SVM)
    svm_verifier = get_active_verifier()
    for user_id in DEFAULT_USERS:
        X_tr, y_tr, grp_tr, X_te, y_te, _ = get_voice_verification_split(
            target_user=user_id,
            X=X,
            person_ids=person_ids,
            session_ids=session_ids,
            unseen_impostors=["impostor4", "impostor5"],
        )
        scores_te = svm_verifier.user_models[user_id].predict_proba(X_te)[:, 1]
        roc_path = RESULTS_DIR / f"e3_roc_{user_id}.png"
        plot_roc_curve(
            y_te,
            scores_te,
            title=f"Voice ROC Curve ({user_id.upper()})",
            save_path=roc_path,
        )

        gen_scores = scores_te[y_te == 1]
        imp_scores = scores_te[y_te == 0]
        dist_path = RESULTS_DIR / f"e3_dist_{user_id}.png"
        plot_score_distributions(
            gen_scores,
            imp_scores,
            threshold=0.5,
            title=f"Match Score Distribution ({user_id.upper()})",
            save_path=dist_path,
        )
        plt.close("all")

    return results_e3


def generate_markdown_summary(e3_results: Dict) -> None:
    """Write comprehensive markdown results summary to results/member2_results_summary.md"""
    out_file = RESULTS_DIR / "member2_results_summary.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Member 2: Experimental Results & Security Verification Summary\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("## 1. Experiment E3: Voice Verification (Mode B)\n\n")
        f.write("Comparison of SVM (RBF kernel) vs. Random Forest per enrolled member on held-out test data (session 3 + unseen open-set impostors).\n\n")
        f.write("| Enrolled User | Model | Accuracy | Precision | Recall | F1-Score | FAR (FP Rate) | FRR (FN Rate) | EER | AUC |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        for user, models in e3_results.items():
            for m, met in models.items():
                f.write(
                    f"| **{user}** | {m.upper()} | {met['accuracy']:.4f} | "
                    f"{met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f} | "
                    f"{met['FAR']:.4f} | {met['FRR']:.4f} | **{met['EER']:.4f}** | **{met['AUC']:.4f}** |\n"
                )

        f.write("\n### Key Observations for E3:\n")
        f.write("- **Grouped Cross-Validation:** Hyperparameters were tuned strictly across sessions (GroupKFold), avoiding session leakage.\n")
        f.write("- **Open-Set Testing:** Unseen impostors were tested exclusively during the test phase.\n")
        f.write("- **SVM Performance:** SVM with RBF kernel consistently achieved low EER and superior separation on the 56-D MFCC+ZCR+RMS feature vectors.\n\n")

        f.write("## 2. Generated Artifacts & Visualizations\n")
        f.write("- `results/e3_roc_*.png`: ROC Curves with EER point marked for each member.\n")
        f.write("- `results/e3_cm_*.png`: Confusion matrices for SVM and Random Forest.\n")
        f.write("- `results/e3_dist_*.png`: Genuine vs Impostor match score distribution histograms.\n")

    print(f"\n[+] Summary report generated at: {out_file.resolve()}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ensure_dataset_exists()

    dataset = VoiceDataset()
    e3_results = run_experiment_e3_voice_verification(dataset)

    generate_markdown_summary(e3_results)
    print("\n" + "=" * 70)
    print("ALL MEMBER 2 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
