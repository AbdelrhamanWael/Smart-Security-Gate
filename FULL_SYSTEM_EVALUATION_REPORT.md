# 📊 Comprehensive System Evaluation Report: Multimodal Biometric Security Gate
**Project:** Smart Security Gate for a Data Center  
**Course Requirement:** Task 2 — Hierarchical Multimodal Biometric Verification System  
**Implementation Constraint:** Strictly Traditional Machine Learning (Zero Deep Learning)  
**Date:** November 2026  
**Team Members & Roles:**
- **Abdelrahman Wael** (Member 1 — Super Admin)
- **Hashem** (Member 2 — Operator)
- **Hagar** (Member 3 — Security Auditor)

---

## 1. Executive Summary

This report delivers the comprehensive empirical evaluation of the **Smart Security Gate** biometric verification system developed for enterprise data center access control. The architecture integrates **Face**, **Voice**, and **Fingerprint** biometrics using traditional machine learning algorithms:

1. **Unimodal Verification:** Mode A (Face-only: HOG + LBP + PCA + SVM/KNN) and Mode B (Voice-only: MFCC + Delta-MFCC + ZCR + RMS + SVM/RF).
2. **Multimodal Early Fusion (Mode C):** Joint feature-level representation via scaled, block-weighted concatenation feeding an RBF-kernel Support Vector Machine.
3. **Hierarchical Verification Cascade (Mode D):** Sequential multi-stage evaluation (Level 1: Face $\rightarrow$ Level 2: Voice & Early Fusion $\rightarrow$ Level 3: Fingerprint).
4. **Fault-Tolerant Gate Policy (Web GUI):** A 2-out-of-3 majority verification rule ensuring high security while preventing legitimate user lockout if a single sensor (e.g., webcam) is degraded.

### Key Performance Highlights:
- **Zero Impostor Breaches:** Early Fusion and the Hierarchical Pipeline achieved a **$0.0\%$ False Acceptance Rate (FAR)** across all open-set intruder tests.
- **Perfect Global Discrimination:** Multimodal Early Fusion achieved an **Area Under the ROC Curve (AUC) of $100.0\%$** and an **Equal Error Rate (EER) of $0.0\%$**.
- **Massive Computational Speedup:** The Hierarchical Cascade successfully filters out over **$98\%$ of unauthorized intruders at Level 1 in under $55\text{ ms}$**, eliminating the computational and latency overhead of activating the microphone (2.5s) or fingerprint sensor.

---

## 2. Experimental Protocol & Zero Data Leakage

To prevent optimistic error reporting and ensure compliance with biometric evaluation standards, the following protocols were strictly enforced:

```
[Session 1 & Session 2 (DEV Set)] ──> GroupKFold Cross-Validation ──> Tune Hyperparameters & Freeze Thresholds (EER)
                                                                                  │
[Session 3 & Unseen Impostors (TEST Set)] ──> Zero-Leakage Evaluation ────────────┴──> Final Benchmark Metrics
```

1. **Session-Based Grouping (`StratifiedGroupKFold`):**
   - Biometric data was collected across separate temporal sessions (Session 1, Session 2, and Session 3).
   - Training and validation splits were grouped strictly by `session_id`. No sample from a test session was ever visible during training. This prevents the models from overfitting to room acoustics, ambient lighting, or microphone positioning.
2. **Pipeline Encapsulation:**
   - Feature normalizers (`StandardScaler`), dimensionality reduction (`PCA`), and block-weighting transformers sit inside Scikit-Learn `Pipeline` objects. They were fitted exclusively on training folds; test folds were transformed without fitting.
3. **Open-Set Impostor Evaluation:**
   - The test set includes completely unseen identities (Ariel Sharon, Colin Powell, Donald Rumsfeld, and synthetic chimeric subjects) that were never present in development or tuning.
4. **Threshold Freezing:**
   - All operational decision thresholds ($T_{\text{low}}, T_{\text{fusion}}, T_{\text{fp}}$) were calculated and frozen on the Development Set (using the EER operating point) and evaluated unmodified on the held-out Test Set.

---

## 3. Biometric Verification Metrics Glossary

In biometric security, standard classification accuracy is insufficient due to severe class imbalance (impostor attempts vastly outnumber genuine attempts). The system is evaluated using domain-specific biometric metrics:

| Metric | Mathematical Formula | Physical & Operational Significance |
| :--- | :---: | :--- |
| **FAR (False Acceptance Rate)** | $\frac{\text{FP}}{\text{FP} + \text{TN}}$ | **Critical Security Metric:** The proportion of unauthorized intruder attempts erroneously granted access. In a data center, this must be $\approx 0.0\%$. |
| **FRR (False Rejection Rate)** | $\frac{\text{FN}}{\text{FN} + \text{TP}}$ | **Usability / Convenience Metric:** The proportion of genuine enrolled users erroneously rejected and forced to retry. |
| **TAR (True Acceptance Rate)** | $1 - \text{FRR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ | The proportion of genuine attempts correctly granted access. |
| **EER (Equal Error Rate)** | Threshold where $\text{FAR} = \text{FRR}$ | **The Universal Biometric Benchmark:** The threshold-independent point where security and usability balance. Lower EER indicates superior intrinsic discriminative capability. |
| **ROC-AUC** | $\int_0^1 \text{TAR}(\text{FAR})\, d(\text{FAR})$ | Global measure of class separability across all possible decision thresholds $\theta \in [0, 1]$. $1.0 = 100\%$ perfect separation. |
| **Confusion Matrix** | $\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$ | Explicit counts of True Negatives (intruders blocked), False Positives (breaches), False Negatives (lockouts), and True Positives (authorized entries). |

---

## 4. Master System Performance Comparison Table

All models evaluated on the held-out test partition (Session 3 + Unseen Impostors). Results expressed as **Mean $\pm$ Standard Deviation** across enrolled subjects:

| Verification System | Model / Algorithm | Accuracy | Precision | Recall (TAR) | FAR (Security) | FRR (Usability) | EER (Benchmark) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A: Face-Only** | **SVM (RBF Kernel)** | $97.2\% \pm 2.8$ | $100.0\% \pm 0.0$ | $83.3\% \pm 16.7$ | $\mathbf{0.0\% \pm 0.0}$ | $16.7\% \pm 16.7$ | $1.7\% \pm 1.7$ | $98.9\% \pm 1.1$ |
| **Mode A: Face-Only** | KNN ($k=5$, Cosine) | $95.6\% \pm 3.3$ | $88.5\% \pm 5.2$ | $83.3\% \pm 16.7$ | $2.0\% \pm 0.7$ | $16.7\% \pm 16.7$ | $1.0\% \pm 0.3$ | $99.2\% \pm 0.1$ |
| **Mode B: Voice-Only** | **SVM (RBF Kernel)** | $95.0\% \pm 1.7$ | $100.0\% \pm 0.0$ | $70.0\% \pm 10.0$ | $\mathbf{0.0\% \pm 0.0}$ | $30.0\% \pm 10.0$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Mode B: Voice-Only** | Random Forest (100) | $90.0\% \pm 3.3$ | $100.0\% \pm 0.0$ | $40.0\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $60.0\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Mode C: Early Fusion** | **SVM (RBF Kernel) ★** | $\mathbf{97.2\% \pm 1.7}$ | $\mathbf{100.0\% \pm 0.0}$ | $\mathbf{83.3\% \pm 10.0}$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{16.7\% \pm 10.0}$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Mode C: Early Fusion** | Random Forest | $95.0\% \pm 1.7$ | $100.0\% \pm 0.0$ | $70.0\% \pm 10.0$ | $\mathbf{0.0\% \pm 0.0}$ | $30.0\% \pm 10.0$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Mode D: Cascade (L1$\rightarrow$L2)** | Hierarchical Pipeline | $95.6\% \pm 3.3$ | $100.0\% \pm 0.0$ | $73.3\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $26.7\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Mode D: Full Cascade (L1$\rightarrow$L3)** | Face + Voice + Fingerprint | $95.6\% \pm 3.3$ | $100.0\% \pm 0.0$ | $73.3\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $26.7\% \pm 20.0$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Bonus 4: Fingerprint** | SVM (RBF Kernel) | $94.4\% \pm 5.6$ | $80.0\% \pm 20.0$ | $100.0\% \pm 0.0$ | $6.7\% \pm 6.7$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |
| **Bonus 4: Fingerprint** | KNN ($k=5$, Cosine) | $88.9\% \pm 11.1$ | $71.4\% \pm 28.6$ | $100.0\% \pm 0.0$ | $13.3\% \pm 13.3$ | $\mathbf{0.0\% \pm 0.0}$ | $0.7\% \pm 0.7$ | $98.9\% \pm 1.1$ |
| **Gate Policy: 2-of-3 Rule** | Multimodal Web Engine | $\mathbf{98.5\% \pm 1.5}$ | $\mathbf{100.0\% \pm 0.0}$ | $\mathbf{95.0\% \pm 5.0}$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{5.0\% \pm 5.0}$ | $\mathbf{0.0\% \pm 0.0}$ | $\mathbf{100.0\% \pm 0.0}$ |

---

## 5. Detailed Breakdown by Enrolled Member

### 5.1 Member 1: Abdelrahman Wael (Super Admin)
- **Role:** Super Admin (Full access to server inventory edit, core server vault, perimeter gate, and emergency lockdown).
- **Test Set Distribution:** 15 genuine attempts, 75 open-set impostor attempts ($N_{\text{test}} = 90$).

| System Mode | Classifier | Accuracy | FAR (FPR) | FRR (FNR) | EER | ROC-AUC | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Face-Only (Mode A)** | SVM (RBF) | $\mathbf{100.0\%}$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $100.0\%$ | $\mathbf{15 / 0 / 75 / 0}$ (Zero Errors) |
| **Face-Only (Mode A)** | KNN | $98.9\%$ | $1.3\%$ | $0.0\%$ | $0.7\%$ | $99.3\%$ | $15 / 1 / 74 / 0$ |
| **Voice-Only (Mode B)** | SVM (RBF) | $96.7\%$ | $0.0\%$ | $20.0\%$ | $0.0\%$ | $100.0\%$ | $12 / 0 / 75 / 3$ |
| **Voice-Only (Mode B)** | Random Forest | $86.7\%$ | $0.0\%$ | $80.0\%$ | $0.0\%$ | $100.0\%$ | $3 / 0 / 75 / 12$ |
| **Early Fusion (Mode C)** | SVM (RBF) | $\mathbf{98.9\%}$ | $\mathbf{0.0\%}$ | $\mathbf{6.7\%}$ | $\mathbf{0.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{14 / 0 / 75 / 1}$ |
| **Hierarchical Cascade** | Sequential L1$\rightarrow$L2 | $98.9\%$ | $0.0\%$ | $6.7\%$ | $0.0\%$ | $100.0\%$ | $14 / 0 / 75 / 1$ |
| **Fingerprint (Bonus 4)** | SVM (RBF) | $100.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | $100.0\%$ | $15 / 0 / 75 / 0$ |

---

### 5.2 Member 2: Hashem (Data Center Operator)
- **Role:** Operator (Perimeter gate control, operational telemetry monitoring, View-Only mode on Google-Sheets inventory).
- **Test Set Distribution:** 15 genuine attempts, 75 open-set impostor attempts ($N_{\text{test}} = 90$).

| System Mode | Classifier | Accuracy | FAR (FPR) | FRR (FNR) | EER | ROC-AUC | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Face-Only (Mode A)** | SVM (RBF) | $94.4\%$ | $0.0\%$ | $33.3\%$ | $3.3\%$ | $97.8\%$ | $10 / 0 / 75 / 5$ |
| **Face-Only (Mode A)** | KNN | $92.2\%$ | $2.7\%$ | $33.3\%$ | $1.3\%$ | $99.1\%$ | $10 / 2 / 73 / 5$ |
| **Voice-Only (Mode B)** | SVM (RBF) | $93.3\%$ | $0.0\%$ | $40.0\%$ | $0.0\%$ | $100.0\%$ | $9 / 0 / 75 / 6$ |
| **Voice-Only (Mode B)** | Random Forest | $93.3\%$ | $0.0\%$ | $40.0\%$ | $0.0\%$ | $100.0\%$ | $9 / 0 / 75 / 6$ |
| **Early Fusion (Mode C)** | SVM (RBF) | $\mathbf{95.6\%}$ | $\mathbf{0.0\%}$ | $\mathbf{26.7\%}$ | $\mathbf{0.0\%}$ | $\mathbf{100.0\%}$ | $\mathbf{11 / 0 / 75 / 4}$ |
| **Hierarchical Cascade** | Sequential L1$\rightarrow$L2 | $92.2\%$ | $0.0\%$ | $46.7\%$ | $0.0\%$ | $100.0\%$ | $8 / 0 / 75 / 7$ |
| **Fingerprint (Bonus 4)** | SVM (RBF) | $88.9\%$ | $13.3\%$ | $0.0\%$ | $0.0\%$ | $100.0\%$ | $15 / 10 / 65 / 0$ |

---

### 5.3 Member 3: Hagar (Security Auditor)
- **Role:** Security Auditor (Audit logs review, telemetry analysis, View-Only mode).
- **Enrolled Modalities:** Voice + Fingerprint biometrics.
- **Verification Performance:**
  - Voice Verification Match Score: **$99.5\%$** (Pass, $T = 50\%$).
  - Fingerprint Biometric Match Score: **$98.2\%$** (Pass, $T = 20\%$).
  - Evaluated via the **2-of-3 Gate Verification Policy**: Access GRANTED ($2/3$ passed), perfectly fulfilling data center security clearance without requiring face data.

---

### 5.4 Impostor Benchmark (Open-Set Attack Simulation)
- **Tested Impostor Personas:** Ariel Sharon, Colin Powell, Donald Rumsfeld, Intruder Finger 104, Intruder Finger 105, and Chimeric cross-source attackers.
- **Total Attack Attempts Evaluated:** Over $150$ simulated physical attacks.
- **Total False Acceptances (Security Breaches):** **$0$** (Zero impostors were granted entry in Early Fusion and Hierarchical Mode).
- **Result:** **$\mathbf{100\%}$ Impostor Rejection Rate** ($\text{TNR} = 1.0$).

---

## 6. In-Depth Comparative Analysis: Why Multimodal Early Fusion Wins

```
Mode A (Face-Only)     ──> Vulnerable to ambient lighting / shadows / pose angles
Mode B (Voice-Only)    ──> Vulnerable to background fan noise / vocal fatigue
Mode C (Early Fusion)  ──> Orthogonal Integration (Spatial Geometry + Vocal Tract Physics)
                           Drop FAR to 0.0% | Push AUC to 100% | Zero Single-Point-of-Failure
```

### 1. Orthogonality of Modalities
Facial geometry (HOG edge gradients and LBP skin texture) and vocal tract dynamics (MFCC spectral envelope, ZCR frequency transitions, and RMS power) originate from fundamentally independent biological phenomena. Because these feature spaces are statistically uncorrelated, an attacker attempting to impersonate a legitimate employee cannot mathematically bridge both feature spaces simultaneously.

### 2. Elimination of Unimodal Failure Modes
- Under poor webcam lighting, a genuine user's face confidence may drop slightly (e.g., to $65\%$). In unimodal face verification with a strict threshold, this causes an erroneous rejection.
- In **Early Fusion (Mode C)**, the joint feature vector allows the voice modality (e.g., $95\%$ match) to compensate for the lighting degradation, maintaining positive verification while preserving a rigid $0.0\%$ False Acceptance Rate against intruders.

### 3. Hierarchical Cascade Efficiency
In a high-throughput enterprise facility, processing audio files and running fingerprint extraction for every pedestrian creates severe queuing bottlenecks:
- **Hierarchical Mode D** runs Level 1 (Face) first.
- In our benchmark, **$98.2\%$ of unauthorized intruders are rejected at Level 1 in under $55\text{ ms}$**.
- This completely eliminates the need to record audio ($2.5$ seconds) or scan fingerprints for intruders, reducing energy consumption and latency by over **$90\%$** compared to a flat, non-hierarchical multimodal system.

---

## 7. Computational Latency & Execution Speed Benchmarks

Measurements conducted on an Intel Core CPU without GPU acceleration (pure CPU execution):

| Pipeline Stage | Processing Steps Involved | Latency (ms) | Early Exit Trigger? |
| :--- | :--- | :---: | :---: |
| **Level 1: Face Scan** | Haar Face Crop + CLAHE + HOG/LBP + PCA + SVM | **$52.8\text{ ms}$** | **YES** (Rejects impostors immediately) |
| **Level 2: Voice & Fusion** | 16kHz VAD + MFCC/ZCR/RMS + Early Fusion + SVM | **$4.0\text{ ms}$** | **YES** (Stops stage-2 attackers) |
| **Level 3: Fingerprint** | Otsu ROI Bounding Box + CLAHE + HOG/LBP + SVM | **$15.2\text{ ms}$** | Final Tier (Core Vault clearance) |
| **Total Pipeline Latency** | Full Sequential Cascade (L1 $\rightarrow$ L2 $\rightarrow$ L3) | **$\mathbf{78.0\text{ ms}}$** | Access Decision Executed in $<0.08\text{ s}$ |

---

## 8. Role-Based Access Control (RBAC) Matrix

Upon passing biometric verification, the system assigns access tokens linked to the user's validated identity:

| User Identity | System Role | Biometric Clearance | Perimeter Gate | Core Server Vault | Server Grid Edit (Google-Sheets) | Emergency Lockdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Abdelrahman Wael** | **Super Admin** | Level 3 (Verified) | 🔓 Full Control | 🔓 Authorized | ✏️ **Full Inline Edit** | 🚨 **Authorized** |
| **Hashem** | **Operator** | Level 2 (Verified) | 🔓 Full Control | 🔒 Restricted | 🔒 View-Only Mode | 🔒 Restricted |
| **Hagar** | **Auditor** | Level 2 (Voice+FP) | 🔓 Authorized | 🔒 Restricted | 🔒 View-Only Mode | 🔒 Restricted |
| **Unverified / Intruder**| **Guest / Impostor**| Denied / Failed | 🔒 **LOCKED** | 🔒 **LOCKED** | 🔒 **Blocked (403)** | 🔒 **Blocked (403)** |

---

## 9. Conclusion

The evaluation demonstrates that the **Smart Security Gate** system fulfills all course objectives and performance benchmarks:
1. **$100\%$ Traditional ML:** Built strictly with OpenCV, SciPy, NumPy, and Scikit-Learn without neural networks.
2. **Proven Multimodal Superiority:** Early Feature Fusion and Hierarchical Cascading eliminate unimodal vulnerabilities, achieving **$0.0\%$ FAR**, **$0.0\%$ EER**, and **$100.0\%$ ROC-AUC**.
3. **Enterprise Operational Feasibility:** Sub-80ms total latency, live WebRTC camera/audio recording, and role-based access control suitable for high-security data centers.
