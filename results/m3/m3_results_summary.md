# Member 3: Early Fusion, Hierarchy & Evaluation Results
Generated: 2026-11-10 07:59:05  |  users evaluated: member1, member2  |  **skipped (missing modality data): member3**
## Protocol
- **DEV** (training, tuning, threshold freezing): genuine sessions 1-2 + *seen* impostor identities.
- **TEST** (all numbers below): genuine session 3 (raw images only) + *unseen* impostor identities (open set).
- Impostor pairs are *virtual identities* (face / voice / fingerprint come from different public sources); every person belongs to exactly one virtual identity, so no person or session crosses DEV/TEST (asserted in code).
- Scalers / PCA / block-normalisation sit inside sklearn Pipelines and are fitted on training folds only. Grouped CV = `StratifiedGroupKFold` by session / identity.
- Thresholds are frozen on DEV out-of-fold scores (EER point) and applied unchanged on TEST. EER and AUC are threshold-free and computed on TEST scores.
- member1: DEV 190 rows (30 genuine), TEST 90 rows (15 genuine); unseen impostors: unseen_chim0, unseen_chim1, unseen_chim2
- member2: DEV 190 rows (30 genuine), TEST 90 rows (15 genuine); unseen impostors: unseen_chim0, unseen_chim1, unseen_chim2

> ⚠ **Small-sample warning.** TEST contains only a handful of genuine attempts per user (see above). One error moves FRR by several points, and EER/AUC of 0% / 100% mean *no error was observed on this small test*, not that the system is perfect. Report them with this caveat.

## 1. Main comparison table (mean ± std over enrolled users)
| System | Model | Accuracy | Precision | Recall | F1 | FAR | FRR | EER | AUC |
|---|---|---|---|---|---|---|---|---|---|
| Face-only (Mode A) | SVM | 97.2% ± 2.8 | 100.0% ± 0.0 | 83.3% ± 16.7 | 90.0% ± 10.0 | 0.0% ± 0.0 | 16.7% ± 16.7 | 1.7% ± 1.7 | 98.9% ± 1.1 |
| Face-only (Mode A) | KNN | 95.6% ± 3.3 | 88.5% ± 5.2 | 83.3% ± 16.7 | 85.4% ± 11.4 | 2.0% ± 0.7 | 16.7% ± 16.7 | 1.0% ± 0.3 | 99.2% ± 0.1 |
| Voice-only (Mode B) | SVM | 95.0% ± 1.7 | 100.0% ± 0.0 | 70.0% ± 10.0 | 81.9% ± 6.9 | 0.0% ± 0.0 | 30.0% ± 10.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Voice-only (Mode B) | RF | 90.0% ± 3.3 | 100.0% ± 0.0 | 40.0% ± 20.0 | 54.2% ± 20.8 | 0.0% ± 0.0 | 60.0% ± 20.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Early Fusion (Mode C) | SVM | 97.2% ± 1.7 | 100.0% ± 0.0 | 83.3% ± 10.0 | 90.6% ± 6.0 | 0.0% ± 0.0 | 16.7% ± 10.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Early Fusion (Mode C) | RF | 95.0% ± 1.7 | 100.0% ± 0.0 | 70.0% ± 10.0 | 81.9% ± 6.9 | 0.0% ± 0.0 | 30.0% ± 10.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Hierarchical (Mode D: face→fusion) | Best pipeline | 95.6% ± 3.3 | 100.0% ± 0.0 | 73.3% ± 20.0 | 83.1% ± 13.5 | 0.0% ± 0.0 | 26.7% ± 20.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Hierarchical + Fingerprint | Full system | 95.6% ± 3.3 | 100.0% ± 0.0 | 73.3% ± 20.0 | 83.1% ± 13.5 | 0.0% ± 0.0 | 26.7% ± 20.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Fingerprint-only (bonus) | SVM | 94.4% ± 5.6 | 80.0% ± 20.0 | 100.0% ± 0.0 | 87.5% ± 12.5 | 6.7% ± 6.7 | 0.0% ± 0.0 | 0.0% ± 0.0 | 100.0% ± 0.0 |
| Fingerprint-only (bonus) | KNN | 88.9% ± 11.1 | 71.4% ± 28.6 | 100.0% ± 0.0 | 80.0% ± 20.0 | 13.3% ± 13.3 | 0.0% ± 0.0 | 0.7% ± 0.7 | 98.9% ± 1.1 |

## 2. Per-user results
| User | System | Acc | Prec | Rec | F1 | FAR | FRR | EER | AUC | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|---|---|
| member1 | Face-only SVM | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 15/0/75/0 |
| member1 | Face-only KNN | 98.9% | 93.8% | 100.0% | 96.8% | 1.3% | 0.0% | 0.7% | 99.3% | 15/1/74/0 |
| member1 | Voice-only SVM | 96.7% | 100.0% | 80.0% | 88.9% | 0.0% | 20.0% | 0.0% | 100.0% | 12/0/75/3 |
| member1 | Voice-only RF | 86.7% | 100.0% | 20.0% | 33.3% | 0.0% | 80.0% | 0.0% | 100.0% | 3/0/75/12 |
| member1 | Early Fusion SVM | 98.9% | 100.0% | 93.3% | 96.6% | 0.0% | 6.7% | 0.0% | 100.0% | 14/0/75/1 |
| member1 | Early Fusion RF | 96.7% | 100.0% | 80.0% | 88.9% | 0.0% | 20.0% | 0.0% | 100.0% | 12/0/75/3 |
| member1 | Fingerprint-only SVM | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 15/0/75/0 |
| member1 | Fingerprint-only KNN | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 15/0/75/0 |
| member1 | Hierarchical (face→fusion) | 98.9% | 100.0% | 93.3% | 96.6% | 0.0% | 6.7% | 0.0% | 100.0% | 14/0/75/1 |
| member1 | Hierarchical + Fingerprint | 98.9% | 100.0% | 93.3% | 96.6% | 0.0% | 6.7% | 0.0% | 100.0% | 14/0/75/1 |
| member2 | Face-only SVM | 94.4% | 100.0% | 66.7% | 80.0% | 0.0% | 33.3% | 3.3% | 97.8% | 10/0/75/5 |
| member2 | Face-only KNN | 92.2% | 83.3% | 66.7% | 74.1% | 2.7% | 33.3% | 1.3% | 99.1% | 10/2/73/5 |
| member2 | Voice-only SVM | 93.3% | 100.0% | 60.0% | 75.0% | 0.0% | 40.0% | 0.0% | 100.0% | 9/0/75/6 |
| member2 | Voice-only RF | 93.3% | 100.0% | 60.0% | 75.0% | 0.0% | 40.0% | 0.0% | 100.0% | 9/0/75/6 |
| member2 | Early Fusion SVM | 95.6% | 100.0% | 73.3% | 84.6% | 0.0% | 26.7% | 0.0% | 100.0% | 11/0/75/4 |
| member2 | Early Fusion RF | 93.3% | 100.0% | 60.0% | 75.0% | 0.0% | 40.0% | 0.0% | 100.0% | 9/0/75/6 |
| member2 | Fingerprint-only SVM | 88.9% | 60.0% | 100.0% | 75.0% | 13.3% | 0.0% | 0.0% | 100.0% | 15/10/65/0 |
| member2 | Fingerprint-only KNN | 77.8% | 42.9% | 100.0% | 60.0% | 26.7% | 0.0% | 1.3% | 97.9% | 15/20/55/0 |
| member2 | Hierarchical (face→fusion) | 92.2% | 100.0% | 53.3% | 69.6% | 0.0% | 46.7% | 0.0% | 100.0% | 8/0/75/7 |
| member2 | Hierarchical + Fingerprint | 92.2% | 100.0% | 53.3% | 69.6% | 0.0% | 46.7% | 0.0% | 100.0% | 8/0/75/7 |

## 3. E4 - Early fusion (Mode C): selected models and tuned parameters
| User | Face model | Voice model | Fusion model | Fusion params (tuned on DEV) |
|---|---|---|---|---|
| member1 | face_knn | voice_svm | fusion_svm | `{'n_pca': 20, 'w_voice': 0.5, 'C': 1.0}` |
| member2 | face_knn | voice_svm | fusion_svm | `{'n_pca': 20, 'w_voice': 0.5, 'C': 1.0}` |

## 4. E5 - Hierarchical pipeline (Mode D)
| User | T_low | T_fusion | T_fp | Impostors stopped L1 / L2 / L3 / accepted | Genuine rejected L1 / L2 / L3 / reach L3 |
|---|---|---|---|---|---|
| member1 | 0.828 | 0.458 | 0.200 | 99% / 1% / 0% / 0% | 0% / 7% / 0% / 93% |
| member2 | 0.721 | 0.784 | 0.166 | 97% / 3% / 0% / 0% | 33% / 13% / 0% / 53% |

### 4.1 Processing time: hierarchy vs flat (average per attempt, seconds)
`compute` = measured feature extraction + inference on this machine; `with capture` adds the human time assumed for speaking the passphrase (3.0 s) and presenting a fingerprint (2.0 s) - **these two are assumptions, not measurements**.

| User | Measured ms (face / voice+fusion / fp) | Flat face+voice | Hier. D (intruder / genuine / 50-50) | Flat all 3 | Hier. D+FP (intruder / genuine / 50-50) |
|---|---|---|---|---|---|
| member1 (compute) | 140 / 6 / 31 | 0.146 | 0.140 / 0.146 / 0.143 | 0.176 | 0.140 / 0.174 / 0.157 |
| member1 (with capture) | 140 / 6 / 31 | 3.146 | 0.180 / 3.146 / 1.663 | 5.176 | 0.180 / 5.041 / 2.610 |
| member2 (compute) | 42 / 5 / 21 | 0.047 | 0.042 / 0.045 / 0.044 | 0.069 | 0.042 / 0.057 / 0.049 |
| member2 (with capture) | 42 / 5 / 21 | 3.047 | 0.122 / 2.045 / 1.084 | 5.069 | 0.122 / 3.124 / 1.623 |

### 4.2 Level-1 threshold policy ablation (frozen on DEV)
| User | T_low policy | T_low | FAR | FRR | Accuracy | Impostors stopped at L1 | Avg intruder time with capture (s) |
|---|---|---|---|---|---|---|---|
| member1 | lenient | 0.828 | 0.0% | 6.7% | 98.9% | 99% | 0.18 |
| member1 | eer | 0.828 | 0.0% | 6.7% | 98.9% | 99% | 0.18 |
| member1 | far1 | 1.000 | 0.0% | 100.0% | 83.3% | 100% | 0.14 |
| member2 | lenient | 0.000 | 0.0% | 26.7% | 95.6% | 0% | 3.05 |
| member2 | eer | 0.721 | 0.0% | 46.7% | 92.2% | 97% | 0.12 |
| member2 | far1 | 1.000 | 0.0% | 100.0% | 83.3% | 100% | 0.04 |

`lenient` = highest threshold with DEV FRR <= 2% (almost no genuine user rejected early), `eer` = EER point, `far1` = lowest threshold with DEV FAR <= 1%.

## 5. E6 - Fusion design choices
### 5.1 Block weight on the voice block (SVM fusion, DEV grouped-CV, best of the other params)
| User | w_voice | CV AUC | CV EER |
|---|---|---|---|
| member1 | 0.5 | 1.000 | 0.0% |
| member1 | 1.0 | 1.000 | 0.0% |
| member1 | 2.0 | 1.000 | 0.0% |
| member2 | 0.5 | 1.000 | 0.0% |
| member2 | 1.0 | 1.000 | 0.0% |
| member2 | 2.0 | 1.000 | 0.0% |

### 5.2 Early fusion vs late fusion (mean of best face score and best voice score)
| User | Early fusion FAR / FRR / EER / AUC | Late fusion FAR / FRR / EER / AUC |
|---|---|---|
| member1 | 0.0% / 6.7% / 0.0% / 1.000 | 0.0% / 20.0% / 0.0% / 1.000 |
| member2 | 0.0% / 26.7% / 0.0% / 1.000 | 0.0% / 6.7% / 0.0% / 1.000 |

## 6. E7 - Robustness (frozen models and thresholds, perturbed TEST inputs)
Low light = pixel values x0.35, over-exposed = x1.8 (before Haar detection + CLAHE); noise = white Gaussian noise at the given SNR added to the passphrase. Face not detected => score 0 (fail closed). Mean over users.

| Condition | Face not detected | Face FAR / FRR | Voice FAR / FRR | Fusion FAR / FRR | Hierarchical FAR / FRR |
|---|---|---|---|---|---|
| nominal | 0% | 2.0% / 16.7% | 0.0% / 30.0% | 0.0% / 16.7% | 0.0% / 26.7% |
| low light (x0.35) | 0% | 2.7% / 16.7% | 0.0% / 30.0% | 0.0% / 16.7% | 0.0% / 26.7% |
| over-exposed (x1.8) | 0% | 6.0% / 16.7% | 0.0% / 30.0% | 0.0% / 13.3% | 0.0% / 26.7% |
| noise 20 dB SNR | 0% | 2.0% / 16.7% | 0.0% / 60.0% | 0.0% / 60.0% | 0.0% / 60.0% |
| noise 10 dB SNR | 0% | 2.0% / 16.7% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% |
| low light + noise 10 dB | 0% | 2.7% / 16.7% | 0.0% / 100.0% | 0.0% / 100.0% | 0.0% / 100.0% |

## 7. Metrics: what they mean and why they suit biometrics
| Metric | Definition | Why it matters for biometric verification |
|---|---|---|
| Confusion matrix | counts of TP, FP, TN, FN (genuine = positive) | shows *which* error happens, not just how many |
| Accuracy | (TP+TN)/all | easy overview but misleading when impostor attempts outnumber genuine ones (ours: 5:1), so it is never used alone |
| Precision | TP/(TP+FP) | of everyone the gate let in, how many were really the claimed user |
| Recall = TAR | TP/(TP+FN) | share of genuine attempts that are accepted |
| F1 | harmonic mean of precision and recall | one number that stays honest under class imbalance |
| **FAR** | FP/(FP+TN) | an intruder is accepted: **the dangerous (security) error** |
| **FRR** | FN/(FN+TP) | the real admin is rejected: the usability error |
| ROC / AUC | TAR vs FAR for every possible threshold | compares systems independently of one threshold; AUC = probability a genuine attempt scores above an impostor one |
| **EER** | operating point where FAR = FRR (lower is better) | standard single-number summary for comparing biometric systems; threshold-free |

## 8. Figures
`results/m3/figures/`: `roc_<user>.png`, `roc_pooled.png`, `cm_<user>_<system>.png`, `cm_pooled_*.png`, `dist_<user>_*.png`.
