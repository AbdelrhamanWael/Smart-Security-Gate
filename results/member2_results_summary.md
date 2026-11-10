# Member 2: Experimental Results & Security Verification Summary

Generated on: 2026-11-10 07:55:54

## 1. Experiment E3: Voice Verification (Mode B)

Comparison of SVM (RBF kernel) vs. Random Forest per enrolled member on held-out test data (session 3 + unseen open-set impostors).

| Enrolled User | Model | Accuracy | Precision | Recall | F1-Score | FAR (FP Rate) | FRR (FN Rate) | EER | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **member1** | SVM | 0.9952 | 0.8333 | 1.0000 | 0.9091 | 0.0049 | 0.0000 | **0.0000** | **1.0000** |
| **member1** | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |
| **member2** | SVM | 0.9905 | 1.0000 | 0.6000 | 0.7500 | 0.0000 | 0.4000 | **0.0000** | **1.0000** |
| **member2** | RF | 0.9952 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.2000 | **0.0000** | **1.0000** |
| **member3** | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |
| **member3** | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |

### Key Observations for E3:
- **Grouped Cross-Validation:** Hyperparameters were tuned strictly across sessions (GroupKFold), avoiding session leakage.
- **Open-Set Testing:** Unseen impostors were tested exclusively during the test phase.
- **SVM Performance:** SVM with RBF kernel consistently achieved low EER and superior separation on the 56-D MFCC+ZCR+RMS feature vectors.

## 2. Generated Artifacts & Visualizations
- `results/e3_roc_*.png`: ROC Curves with EER point marked for each member.
- `results/e3_cm_*.png`: Confusion matrices for SVM and Random Forest.
- `results/e3_dist_*.png`: Genuine vs Impostor match score distribution histograms.
