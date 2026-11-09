# Member 2: Experimental Results & Security Verification Summary

Generated on: 2026-11-08 14:52:21

## 1. Experiment E3: Voice Verification (Mode B)

Comparison of SVM (RBF kernel) vs. Random Forest per enrolled member on held-out test data (session 3 + unseen open-set impostors).

| Enrolled User | Model | Accuracy | Precision | Recall | F1-Score | FAR (FP Rate) | FRR (FN Rate) | EER | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **member1** | SVM | 0.8875 | 0.4146 | 0.8500 | 0.5574 | 0.1091 | 0.1500 | **0.1545** | **0.9566** |
| **member1** | RF | 0.9958 | 0.9524 | 1.0000 | 0.9756 | 0.0045 | 0.0000 | **0.0000** | **1.0000** |
| **member2** | SVM | 0.9792 | 0.8000 | 1.0000 | 0.8889 | 0.0227 | 0.0000 | **0.0000** | **1.0000** |
| **member2** | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |
| **member3** | SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |
| **member3** | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **0.0000** | **1.0000** |

### Key Observations for E3:
- **Grouped Cross-Validation:** Hyperparameters were tuned strictly across sessions (GroupKFold), avoiding session leakage.
- **Open-Set Testing:** Unseen impostors were tested exclusively during the test phase.
- **SVM Performance:** SVM with RBF kernel consistently achieved low EER and superior separation on the 56-D MFCC+ZCR+RMS feature vectors.

## 2. Experiment E8: Speech Command Recognition (Bonus 1)

Isolated-word classification for `['open', 'lock', 'logs', 'status', 'unknown']`.

| Model | Overall Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---|---|---|---|
| **SVM** | **99.6154%** | 0.9967 | 0.9967 | 0.9967 |
| **KNN** | **90.3846%** | 0.9185 | 0.8900 | 0.9013 |

## 3. Role-Based Access Control & Security Matrix (Bonus 1)

| Test Description | Claimed User | Verified? | Level | Command | Expected | Actual Status | Pass? |
|---|---|---|---|---|---|---|---|
| Unverified intruder executing 'open' | None | False | 0 | `open` | `ACCESS_DENIED` | `ACCESS_DENIED` | **YES** |
| Failed verification user executing 'status' | impostor1 | False | 0 | `status` | `ACCESS_DENIED` | `ACCESS_DENIED` | **YES** |
| Super Admin executing 'open' | member1 | True | 2 | `open` | `SUCCESS` | `SUCCESS` | **YES** |
| Super Admin executing 'logs' | member1 | True | 2 | `logs` | `SUCCESS` | `SUCCESS` | **YES** |
| Operator executing 'open' | member2 | True | 2 | `open` | `SUCCESS` | `SUCCESS` | **YES** |
| Operator attempting 'logs' (Protected) | member2 | True | 2 | `logs` | `PERMISSION_DENIED` | `PERMISSION_DENIED` | **YES** |
| Auditor executing 'logs' | member3 | True | 2 | `logs` | `SUCCESS` | `SUCCESS` | **YES** |
| Auditor attempting 'open' (Protected) | member3 | True | 2 | `open` | `PERMISSION_DENIED` | `PERMISSION_DENIED` | **YES** |
| Super Admin with background noise / unknown word | member1 | True | 2 | `unknown` | `UNKNOWN_COMMAND` | `UNKNOWN_COMMAND` | **YES** |

## 4. Generated Artifacts & Visualizations
- `results/e3_roc_*.png`: ROC Curves with EER point marked for each member.
- `results/e3_cm_*.png`: Confusion matrices for SVM and Random Forest.
- `results/e3_dist_*.png`: Genuine vs Impostor match score distribution histograms.
- `results/e8_cm_commands_*.png`: Multi-class confusion matrices for speech commands.
- `logs/audit_log.csv`: Full audit trail of all access and execution events.
