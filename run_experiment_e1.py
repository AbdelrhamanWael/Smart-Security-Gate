import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dataset_loader import load_face_dataset
from src.face_models import calculate_biometric_metrics

def evaluate_configuration(X, y, groups, use_pca=True, n_components=30):
    """Builds pipeline and evaluates via 3-Fold GroupKFold."""
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=min(n_components, len(X) - 1), random_state=42)))
    steps.append(("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)))
    
    pipeline = Pipeline(steps)
    cv = GroupKFold(n_splits=3)
    y_scores = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    metrics = calculate_biometric_metrics(y, y_scores)
    return metrics, y_scores

def main():
    print("=" * 70)
    print("      EXPERIMENT E1: ABLATION STUDY OF FACE FEATURES & PCA        ")
    print("=" * 70)
    
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load data under all 3 feature modes
    print("\n[Step 1] Extracting feature representations...")
    X_hog, y, groups, _ = load_face_dataset(target_user="member1", feature_mode="hog")
    X_lbp, _, _, _      = load_face_dataset(target_user="member1", feature_mode="lbp")
    X_both, _, _, _     = load_face_dataset(target_user="member1", feature_mode="both")
    
    experiments = [
        ("HOG only",               X_hog,  False),
        ("HOG only + PCA",         X_hog,  True),
        ("LBP only",               X_lbp,  False),
        ("LBP only + PCA",         X_lbp,  True),
        ("HOG + LBP",              X_both, False),
        ("HOG + LBP + PCA (Ours)", X_both, True),
    ]
    
    results = []
    roc_data = {}
    
    print("\n[Step 2] Running GroupKFold Cross-Validation for each setting...")
    for name, feat_matrix, use_pca in experiments:
        metrics, scores = evaluate_configuration(feat_matrix, y, groups, use_pca=use_pca)
        dim = 30 if use_pca else feat_matrix.shape[1]
        
        results.append({
            "Config": name,
            "Dim": dim,
            "Accuracy": metrics["accuracy"] * 100,
            "Precision": metrics["precision"] * 100,
            "Recall": metrics["recall"] * 100,
            "FAR": metrics["FAR"] * 100,
            "FRR": metrics["FRR"] * 100,
            "EER": metrics["EER"] * 100,
            "EER_Thresh": metrics["EER_threshold"]
        })
        roc_data[name] = (y, scores)
        print(f" -> Finished: {name:25s} | EER: {metrics['EER'] * 100:.2f}% | Acc: {metrics['accuracy'] * 100:.2f}%")
        
    # 2. Print Markdown Table for Report
    print("\n" + "=" * 70)
    print("### EXPERIMENT E1 SUMMARY TABLE (Copy this into your Report):")
    print("=" * 70)
    print("| Configuration | Feature Dims | Accuracy | FAR (%) | FRR (%) | EER (%) | EER Threshold |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        print(f"| {r['Config']} | {r['Dim']} | {r['Accuracy']:.2f}% | {r['FAR']:.2f}% | {r['FRR']:.2f}% | **{r['EER']:.2f}%** | {r['EER_Thresh']:.4f} |")
    print("=" * 70)
    
    # 3. Generate ROC Comparison Curve Plot
    print("\n[Step 3] Plotting ROC curves to 'reports/experiment_e1_roc.png'...")
    plt.figure(figsize=(9, 7))
    for name, (y_true, y_scores) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Acceptance Rate (FAR / FPR)", fontsize=11)
    plt.ylabel("True Acceptance Rate (TAR / TPR / Recall)", fontsize=11)
    plt.title("Experiment E1: ROC Comparison Across Feature Configurations", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/experiment_e1_roc.png", dpi=300)
    plt.close()
    
    print("[SUCCESS] ROC plot saved to 'reports/experiment_e1_roc.png'!")
    print("Experiment E1 is fully completed.")

if __name__ == "__main__":
    main()

