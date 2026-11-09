# Task 2: Hierarchical Multimodal Biometric Verification System
## Project Plan: "Smart Security Gate for a Data Center"

**Team size:** 3 or 4 students  **Deadline:** 13/11/2026
**Constraint:** Traditional Machine Learning only (no Deep Learning)

---

## 1. Project Summary

We build a security gate for a sensitive server room. The team members are the authorized **System Admins**. Anyone else is an **intruder**.

A person *claims* an identity (selects a name). The system then verifies the claim using:

1. **Face** (webcam)
2. **Voice** (microphone, spoken passphrase)
3. **Fingerprint** (uploaded fingerprint image, bonus)

Only after successful verification does the system enter **Admin Mode**, where **voice commands** ("Open", "Lock", "Logs"...) are available according to the verified user's permissions.

We compare four configurations: **face-only**, **voice-only**, **multimodal (early fusion)** and the **hierarchical pipeline**.

> **Verification vs. identification.** Verification answers: *"Is this person really who they claim to be?"* (genuine vs. impostor, yes/no). It is not "which team member is this?". Our design is built around verification, so it needs impostor data and uses FAR/FRR/EER.

---

## 2. Requirement Coverage Matrix

| # | Requirement | How we cover it | Section |
|---|---|---|---|
| 1 | Face: ≥ 2 feature methods | HOG + LBP, plus PCA on top of HOG | 5.1 |
| 2 | Voice: MFCC, ZCR, RMS | All three extracted, summarized by mean/std | 5.2 |
| 3 | Early fusion | Normalized face and voice vectors concatenated into one vector | 5.3 |
| 4 | ≥ 2 ML models per modality | Face: SVM + KNN. Voice: SVM + Random Forest. Fusion: SVM + Random Forest | 6 |
| 5 | Face-only verification | Mode A | 7 |
| 6 | Voice-only verification | Mode B | 7 |
| 7 | Multimodal verification (early fusion) | Mode C | 7 |
| 8 | Hierarchical pipeline | Mode D, three levels with thresholds | 8 |
| 9 | Metrics + confusion matrices + explanation | Accuracy, Precision, Recall, F1, FAR, FRR, TAR, ROC/AUC, EER | 10 |
| 10 | Compare individual vs multimodal | Single comparison table across all modes and models | 10 |
| B1 | Speech command system tied to verification | Isolated-word recognizer, role-based permissions | 11.1 |
| B2 | Real-time verification | Webcam face + microphone voice with frame voting | 11.2 |
| B3 | Custom dataset | Faces and voices recorded by the team | 4 |
| B4 | Fingerprint modality | HOG/LBP + SVM, integrated as Level 3 | 11.4 |
| C1 | No Deep Learning | Only scikit-learn, OpenCV, librosa, scikit-image | 12 |
| C2 | Every member understands everything | Cross-training plan and Q&A list | 14 |

---

## 3. System Architecture

```
 Claimed identity (dropdown)
          │
          ▼
 ┌──────────────────┐   score s_face
 │ LEVEL 1: FACE    │──────────────┐
 │ (webcam)         │              │ s_face < T_low  ──► ACCESS DENIED
 └──────────────────┘              │
          │ s_face ≥ T_low         │
          ▼                        │
 ┌──────────────────────────────┐  │
 │ LEVEL 2: VOICE + EARLY FUSION│  │
 │ passphrase (mic)             │  │
 │ [face feats | voice feats]   │──┤ s_fusion < T_fusion ─► ACCESS DENIED
 └──────────────────────────────┘  │
          │ s_fusion ≥ T_fusion    │
          ▼                        │
 ┌──────────────────┐              │
 │ LEVEL 3:         │              │
 │ FINGERPRINT      │──────────────┘ mismatch ─► ACCESS DENIED
 │ (image upload)   │
 └──────────────────┘
          │ match
          ▼
   ADMIN MODE: voice commands filtered by the user's role
```

**Access levels:**

| Level reached | Meaning | What it allows |
|---|---|---|
| Level 1 passed | Face matched | View-only (e.g., "Status") |
| Level 2 passed | Face + voice (fusion) | Operator commands (e.g., "Open", "Lock") |
| Level 3 passed | Face + voice + fingerprint | Sensitive commands (e.g., "Logs") |

The hierarchy has a purpose: it rejects obvious intruders early (saving processing and user time) and demands stronger evidence for more sensitive actions.

---

## 4. Dataset (Bonus 3: Custom Dataset)

### 4.1 What we collect

| Data | Amount | Notes |
|---|---|---|
| Face images | ~60-100 per member | Different lighting, small pose changes, glasses on/off, different days |
| Voice passphrase | ~20-30 recordings per member | Same fixed phrase, 2-3 s, 16 kHz mono, different days and distances |
| Command words | ~20-30 recordings per word per member | Words: Open, Lock, Logs, Status |
| "Unknown" class for commands | Background noise and other words | So the recognizer can reject non-commands |
| Fingerprints | Public dataset (e.g., FVC2000 or SOCOFing) | See 11.4 |
| **Impostors** | At least 5-10 extra people | Friends who consent, or public face/voice datasets |

All team members and friends must give consent for their data to be recorded and used.

### 4.2 Labeling

Folder layout: `data/{face,voice,commands,fingerprint}/{person_id}/{session_id}/file`.
Each file carries `person_id`, `session_id`, and `modality`.

### 4.3 Splitting (to avoid data leakage)

- **Never split random video frames** into train and test. Neighboring frames are almost identical and make accuracy look unrealistically high.
- Split **by session** (for example, recorded on different days or under different lighting). Use `GroupKFold` with `session_id` as the group.
- Keep a set of **impostor persons that never appear in training** and use them only for testing (open-set evaluation).
- Fit scalers, PCA and thresholds on training/validation data only.

---

## 5. Feature Extraction

### 5.1 Face

**Preprocessing:** convert to grayscale, detect and crop the face with OpenCV's Haar cascade (a traditional method), resize to a fixed size (e.g., 128×128), apply histogram equalization.

| Method | Description | Why |
|---|---|---|
| **HOG** | Histogram of Oriented Gradients, captures shape and edges | Robust to small lighting changes |
| **LBP** | Uniform Local Binary Patterns, histograms per grid cell | Captures texture, robust to monotonic brightness changes |
| **PCA** | Applied to the HOG vector (e.g., 100 components) | Reduces dimension, removes noise, keeps HOG from dominating the fusion |

We will report HOG alone, LBP alone, and HOG+LBP, so the choice of face features is justified by experiment.

### 5.2 Voice

Preprocessing: resample to 16 kHz mono, trim silence using RMS energy (simple voice activity detection), normalize amplitude.

Frame-level features are summarized into a fixed-length vector using **mean and standard deviation**:

- **MFCC** (13 coefficients, optionally + deltas): describes the spectral envelope of the voice (vocal tract shape).
- **Zero-Crossing Rate (ZCR):** how often the signal crosses zero, which reflects noisiness and pitch characteristics.
- **RMS Energy:** loudness and energy pattern over time.

### 5.3 Early Fusion

1. Scale each modality **separately** (`StandardScaler`, fitted on training data only).
2. Reduce face dimensions with PCA so face and voice blocks have comparable sizes.
3. Concatenate: `[face_vector | voice_vector]` (and optionally the fingerprint vector) into **one** feature vector.
4. Optionally apply block weights (e.g., 0.5/0.5) and tune them on validation data.
5. Train a final classifier on the fused vector.

*Optional enhancement:* also implement **late fusion** (averaging per-modality scores) as a comparison against early fusion.

---

## 6. Models

Per-user binary verifiers: for each team member, a model is trained on **genuine** samples (that member) vs. **impostor** samples (other members and extra people). The model outputs a probability, which becomes the match score. Hyperparameters are tuned with grouped cross-validation.

| Modality | Model 1 | Model 2 |
|---|---|---|
| Face | SVM (RBF) | KNN |
| Voice | SVM (RBF) | Random Forest |
| Fusion (face + voice) | SVM (RBF) | Random Forest |
| Fingerprint | SVM | KNN |
| Command recognizer | SVM | KNN |

(Decision Tree and Logistic Regression can be added as baselines.)

---

## 7. Required Verification Modes

| Mode | Input | Features | Output |
|---|---|---|---|
| **A. Face-only** | Face image | HOG / LBP / HOG+LBP | Accept / Reject |
| **B. Voice-only** | Passphrase recording | MFCC + ZCR + RMS | Accept / Reject |
| **C. Multimodal (Early Fusion)** | Face + voice | Fused vector | Accept / Reject |
| **D. Hierarchical pipeline** | Face → voice/fusion → fingerprint | As in Section 3 | Accept / Reject + access level |

---

## 8. Hierarchical Pipeline: Decision Logic

1. **Level 1, Face:** compute `s_face` for the claimed identity. If `s_face < T_low`, reject immediately.
2. **Level 2, Voice + Early Fusion:** record the passphrase, build the fused vector, compute `s_fusion`. If `s_fusion < T_fusion`, reject.
3. **Level 3, Fingerprint (bonus):** the uploaded fingerprint must match the claimed user. If not, reject (or stay at Level 2 permissions).

**Thresholds** (`T_low`, `T_fusion`, and the fingerprint threshold) are chosen on **validation data** only, for example at the EER point or at a target FAR such as 1%. They are then frozen before testing.

**Experiments for the hierarchy:** accuracy/FAR/FRR and the average processing time compared with always running every modality ("flat" pipeline).

---

## 9. Implementation Constraints Checklist

- Libraries: scikit-learn, OpenCV, scikit-image, librosa, NumPy, matplotlib. **No PyTorch/TensorFlow/Keras, no CNNs, no pretrained deep models.**
- Anything beyond the course (e.g., LBP details, Haar cascades, HOG parameters, RMS-based voice activity detection, EER/ROC) is explained in the report in our own words.

---

## 10. Evaluation

### 10.1 Metrics and why we use them

| Metric | Meaning | Why it suits biometric verification |
|---|---|---|
| **Confusion matrix** | Counts of TP, TN, FP, FN | Shows exactly which errors occur |
| **Accuracy** | (TP+TN)/all | Easy overview, but can be misleading when genuine and impostor counts are unbalanced |
| **Precision** | TP/(TP+FP) | Of those accepted, how many were truly genuine |
| **Recall / TAR** | TP/(TP+FN) | How many genuine users are accepted |
| **F1-score** | Harmonic mean of precision and recall | Single number balancing both, robust to imbalance |
| **FAR** (False Accept Rate) | FP/(FP+TN) | Intruder accepted: the dangerous security error |
| **FRR** (False Reject Rate) | FN/(FN+TP) | Real user rejected: usability error |
| **ROC curve / AUC** | TAR vs. FAR across all thresholds | Compares systems independent of one threshold |
| **EER** (Equal Error Rate) | Threshold where FAR = FRR | Standard single-number summary for comparing biometric systems (lower is better) |

### 10.2 Experiments

| ID | Experiment | Purpose |
|---|---|---|
| E1 | Face: HOG vs. LBP vs. HOG+LBP (with/without PCA) | Choose face features |
| E2 | Face: SVM vs. KNN | Face-only results (Mode A) |
| E3 | Voice: SVM vs. Random Forest | Voice-only results (Mode B) |
| E4 | Early fusion: SVM vs. Random Forest | Multimodal results (Mode C) |
| E5 | Hierarchical pipeline (with/without fingerprint) | Mode D results and time comparison |
| E6 | Fusion weights and early vs. late fusion (optional) | Show fusion design choices |
| E7 | Robustness: lighting changes, noisy audio | Discuss real-world behavior |
| E8 | Command recognition accuracy and confusion matrix | Bonus 1 |

### 10.3 Main comparison table (template)

| System | Model | Accuracy | Precision | Recall | F1 | FAR | FRR | EER | AUC |
|---|---|---|---|---|---|---|---|---|---|
| Face-only | SVM | | | | | | | | |
| Face-only | KNN | | | | | | | | |
| Voice-only | SVM | | | | | | | | |
| Voice-only | RF | | | | | | | | |
| Early fusion | SVM | | | | | | | | |
| Early fusion | RF | | | | | | | | |
| Hierarchical | best | | | | | | | | |
| Hierarchical + fingerprint | best | | | | | | | | |

Include a confusion matrix and ROC curve for each system, and a short written analysis (does fusion beat each single modality? at what cost in time and user effort?).

---

## 11. Bonus Tasks

### 11.1 Bonus 1: Verification-Based Speech Command System

- **Commands:** isolated words (Open, Lock, Logs, Status) plus an "unknown/noise" class.
- **Features and models:** MFCC (mean/std, optionally deltas) with SVM and KNN.
- **Access control:** commands are only listened to **after** verification. Which commands are allowed depends on the verified user's **role** and **level reached**.

Example permission table (replace with real team names):

| User / state | Role | Allowed commands |
|---|---|---|
| Member 1 | Super Admin | Open, Lock, Logs, Status |
| Member 2 | Operator | Open, Lock, Status |
| Member 3 | Auditor | Logs, Status |
| Member 4 (if any) | Guest | Status |
| Unverified / failed | none | **None**: "Access Denied" |

- A recognized command that the user is not permitted to run returns "Permission Denied".
- Extras: session timeout (e.g., 60 s), and an **audit log** (time, claimed user, result, command).
- The "Open" action is simulated (prints "Door Opened" and updates the GUI).

### 11.2 Bonus 2: Real-Time Verification

- **Face:** OpenCV `VideoCapture`, Haar face detection, features computed on the live crop. Accept only if the match holds over **N consecutive frames** (majority vote, e.g., 7 of 10), to avoid flicker.
- **Voice:** record 2-3 s from the microphone (e.g., `sounddevice`), apply the RMS-based silence trimming, then extract features.
- **GUI:** a simple OpenCV or Tkinter window with the live camera, a claimed-identity selector, a "Scan Fingerprint" button, status text, and the command panel.

### 11.3 Bonus 3: Custom Dataset

See Section 4. The experiments in Section 10 use the team-collected data.

### 11.4 Bonus 4: Fingerprint Verification

- **Data:** public fingerprint images (e.g., FVC2000 or SOCOFing). Assign a few fingers (with multiple impressions each) to represent team members, and use other fingers as impostors.
- **Features/model:** preprocessing (resize, contrast enhancement), HOG and/or LBP features, then SVM and KNN.
- **Integration:** the "Scan Fingerprint" button uploads an image and checks it against the claimed user. This is Level 3 of the hierarchy and can also be added as a third block in the fusion vector.
- **Honest limitation to state in the report:** the fingerprints are not the team's real fingerprints, so this part is a simulation. HOG/LBP are also weaker for fingerprints than minutiae-based methods, which we mention as a limitation.

---

## 12. Limitations and Discussion Points (for the report)

- Small dataset (few people), so results may not generalize.
- **Spoofing:** a photo or a voice replay could fool the system. Possible mitigations: a random challenge phrase, a blink check (future work).
- Sensitivity to lighting, microphone quality and background noise (tested in E7).
- Fingerprint part is simulated with public data.
- Thresholds trade off security (low FAR) against usability (low FRR).

---

## 13. Suggested Project Structure

```
project/
├── data/                 # face, voice, commands, fingerprint (by person/session)
├── src/
│   ├── preprocess.py     # face crop, audio trim, fingerprint enhance
│   ├── features_face.py  # HOG, LBP, PCA
│   ├── features_voice.py # MFCC, ZCR, RMS
│   ├── fusion.py         # scaling + early fusion (+ optional late fusion)
│   ├── models.py         # SVM, KNN, RF training and tuning
│   ├── verify.py         # modes A, B, C
│   ├── hierarchy.py      # mode D with thresholds
│   ├── commands.py       # Bonus 1: recognizer and permissions
│   ├── fingerprint.py    # Bonus 4
│   ├── evaluate.py       # metrics, confusion matrices, ROC, EER
│   └── gui_realtime.py   # Bonus 2: live demo
├── notebooks/            # experiments E1-E8
├── results/              # tables, figures
└── report/
```

---

## 14. Team Understanding Plan

Every member must be able to explain the **entire** system, including bonuses.

- Each member owns a module, but **everyone presents their module to the team** before the final week.
- Everyone runs the full pipeline and live demo at least once.
- Likely questions to prepare for:
  - What is the difference between verification and identification?
  - What do HOG, LBP, PCA, MFCC, ZCR and RMS compute, and why those?
  - Why scale each modality separately before fusion?
  - Why is FAR more critical than FRR here? What is EER?
  - How were thresholds chosen, and why only on validation data?
  - How did we avoid data leakage?
  - Why does the hierarchy save time? What happens when Level 1 fails?
  - How are command permissions enforced? What stops an unverified user?
  - How does the fingerprint part work, and what is simulated?

---

## 15. Timeline

| Dates | Work |
|---|---|
| Nov 5-6 | Finalize roles and permissions, collect faces/voices/commands (all members), gather impostor and fingerprint data |
| Nov 7-8 | Preprocessing, feature extraction, models for face and voice (Modes A, B) |
| Nov 9 | Early fusion (Mode C), thresholds, hierarchical pipeline (Mode D) |
| Nov 10 | Real-time GUI (Bonus 2), command recognizer and permissions (Bonus 1) |
| Nov 11 | Fingerprint integration (Bonus 4), run all experiments E1-E8 |
| Nov 12 | Report, figures, cross-training and demo rehearsal |
| Nov 13 | Final checks and submission (buffer for fixes) |

**Priority if time is short:** core requirements (Modes A-D + evaluation) first, then Bonus 3 and Bonus 2, then Bonus 1, then Bonus 4.

---

## 16. Report Outline

1. Introduction and problem statement
2. System design (architecture, hierarchy, access levels)
3. Dataset (collection, labeling, splits, impostors, ethics/consent)
4. Feature extraction (face, voice, fingerprint) and fusion
5. Models and tuning
6. Evaluation metrics and why they suit biometrics
7. Results: single modality vs. multimodal vs. hierarchical (tables, confusion matrices, ROC)
8. Bonuses: commands and permissions, real-time demo, custom dataset, fingerprint
9. Limitations and future work
10. Team contributions and conclusion
