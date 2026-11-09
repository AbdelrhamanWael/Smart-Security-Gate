# Task 2: Hierarchical Multimodal Biometric Verification
## Team Work Division and Checklists (3 Members)

**Deadline:** 13/11/2026  **Constraint:** Traditional ML only, no Deep Learning
**Rule:** every member must be able to explain the **whole** project, including all bonuses.

---

## 1. Team Roles at a Glance

| | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| **Name** | _fill in_ | _fill in_ | _fill in_ |
| **Main area** | Face + Fingerprint | Voice + Speech Commands | Fusion + Hierarchy + Evaluation + Real-time GUI |
| **Required parts** | Face features, Face-only (Mode A) | Voice features, Voice-only (Mode B) | Early Fusion (Mode C), Hierarchical (Mode D), Evaluation |
| **Bonus parts** | Bonus 4 (Fingerprint) | Bonus 1 (Speech Commands) | Bonus 2 (Real-time) |
| **Bonus 3 (dataset)** | Collects and labels face data | Collects and labels voice + command data | Organizes dataset structure, impostor data, splits |
| **Role in the gate** (for permissions) | Super Admin | Operator | Auditor |

> Everyone records their own face, voice and commands, because the dataset is built from the three of us.

---

## 2. Requirements Master Checklist

Tick when done **and** verified by a teammate.

### Core requirements
- [x] Face: at least 2 feature methods (HOG + LBP, optionally PCA) , **M1**
- [x] Voice: MFCC + ZCR + RMS , **M2**
- [ ] Early Fusion of face + voice into one vector , **M3**
- [x] At least 2 ML models for face (SVM, KNN) , **M1**
- [x] At least 2 ML models for voice (SVM, Random Forest) , **M2**
- [x] Face-only verification (Mode A) , **M1**
- [x] Voice-only verification (Mode B) , **M2**
- [ ] Multimodal verification with early fusion (Mode C) , **M3**
- [ ] Hierarchical verification pipeline (Mode D) , **M3**
- [ ] Metrics + confusion matrices for every system , **M3** (all use it)
- [ ] Explanation of each metric and why it suits biometrics , **M3** (+ all review)
- [ ] Comparison: individual vs. multimodal vs. hierarchical , **M3**
- [x] No Deep Learning anywhere in the code , **all**
- [ ] Every member can explain the full system , **all**

### Bonuses
- [x] **Bonus 1:** Speech command recognition connected to verification, role-based permissions, unverified users blocked , **M2**
- [ ] **Bonus 2:** Real-time face (webcam) and voice (microphone) verification , **M3** (face live part with M1, voice live part with M2)
- [x] **Bonus 3:** Custom dataset collected and labeled by the team, used in experiments , **all**
- [x] **Bonus 4:** Fingerprint modality integrated into the pipeline , **M1**

### Deliverables
- [ ] Source code (clean, runs end-to-end)
- [ ] Report (see Section 8)
- [ ] Live demo prepared
- [ ] Dataset folder (or description if too large to submit)

---

## 3. Shared Tasks (everyone)

- [ ] Agree on the passphrase (one fixed sentence, e.g. 3-5 words) and command words: **Open, Lock, Logs, Status**
- [ ] Agree on folder structure and file naming: `data/{face,voice,commands,fingerprint}/{person_id}/{session_id}/file`
- [ ] Each member records **their own data** over at least **2-3 different sessions** (different days or conditions)
- [ ] Gather **impostor data** (at least 5-10 extra people or a public dataset), split work: M1 faces, M2 voices, M3 organizes and checks
- [ ] Follow the split rule: **split by session, not by random frames**, and keep impostor persons unseen for the final test
- [ ] Code reviews: each member reviews one other member's module before integration
- [ ] Cross-training sessions (Section 7)
- [ ] Write own sections of the report (Section 8)

---

## 4. Interfaces Between Members (agree on these early)

So the parts plug together, every module exposes the same simple functions:

```python
# M1 (face)
extract_face_features(image) -> np.ndarray          # HOG + LBP (+PCA)
face_score(image, user_id)   -> float               # match probability 0..1

# M2 (voice)
extract_voice_features(wav_path_or_array) -> np.ndarray   # MFCC + ZCR + RMS stats
voice_score(audio, user_id)  -> float

# M3 (fusion / hierarchy / evaluation)
fuse(face_vec, voice_vec, weights=None) -> np.ndarray     # scaled + concatenated
fusion_score(face_img, audio, user_id) -> float
hierarchical_verify(user_id, face_img, audio, fingerprint_img=None)
    -> {"decision": bool, "level_reached": int, "scores": {...}}
evaluate(y_true, y_score, threshold) -> dict               # acc, prec, rec, F1, FAR, FRR, EER, AUC + plots

# M1 (fingerprint, bonus)
fingerprint_score(image, user_id) -> float

# M2 (commands, bonus)
recognize_command(audio) -> str
allowed_commands(user_id, level_reached) -> list[str]
```

---

## 5. Member 1: Face + Fingerprint

### Responsibilities
Face preprocessing, face features, face models, Mode A, fingerprint modality, and face data collection.

### Checklist
**Data (Bonus 3)**
- [x] Record 60-100 face images of yourself across lighting, angles, glasses on/off, 2-3 sessions
- [x] Collect face images of impostors (friends with consent, or a public face dataset)
- [x] Label everything by `person_id` and `session_id`

**Preprocessing and features**
- [x] Face detection and crop with OpenCV Haar cascade
- [x] Grayscale, resize to fixed size, histogram equalization
- [x] HOG features
- [x] LBP features (uniform LBP, grid histograms)
- [x] PCA on HOG (fit on training data only)
- [x] Experiment E1: HOG vs. LBP vs. HOG+LBP (with and without PCA)

**Models and Mode A**
- [x] SVM (RBF) per-user genuine vs. impostor verifier
- [x] KNN per-user verifier
- [x] Hyperparameter tuning with grouped cross-validation (by session)
- [x] Implement `face_score(...)`
- [x] Face-only results: confusion matrix, ROC, EER (using M3's `evaluate`)

**Bonus 4: Fingerprint**
- [x] Download a public fingerprint dataset (e.g., FVC2000 or SOCOFing)
- [x] Assign 3 fingers (several impressions each) to represent the team, other fingers as impostors
- [x] Preprocess (resize, contrast enhancement), extract HOG/LBP
- [x] Train SVM and KNN, implement `fingerprint_score(...)`
- [x] "Scan Fingerprint" upload logic (with M3's GUI)
- [x] Write the honest limitation: fingerprints are simulated with public data

**Real-time support (Bonus 2)**
- [x] Provide a function that takes a live webcam frame and returns the face score
- [x] Implement frame voting (e.g., 7 of 10 frames) with M3

### Must be able to explain (besides own part)
Voice features (MFCC, ZCR, RMS), how fusion scaling works, how thresholds and FAR/FRR/EER are computed, how command permissions are enforced.

---

## 6. Member 2: Voice + Speech Commands

### Responsibilities
Audio preprocessing, voice features, voice models, Mode B, speech commands with permissions, and voice data collection.

### Checklist
**Data (Bonus 3)**
- [x] Record the passphrase 20-30 times across 2-3 sessions (different days/distances)
- [x] Record each command word (Open, Lock, Logs, Status) 20-30 times
- [x] Record an "unknown" class (background noise, other words)
- [x] Collect voice samples from impostors (friends with consent, or a public voice dataset)
- [x] Label everything by `person_id` and `session_id`

**Preprocessing and features**
- [x] Resample to 16 kHz mono, normalize amplitude
- [x] Silence trimming using RMS energy (simple voice activity detection)
- [x] MFCC (13 coefficients, optionally + deltas), mean and std
- [x] Zero-Crossing Rate, mean and std
- [x] RMS Energy, mean and std
- [x] Combine into one fixed-length vector in `extract_voice_features(...)`

**Models and Mode B**
- [x] SVM (RBF) per-user verifier
- [x] Random Forest per-user verifier
- [x] Hyperparameter tuning with grouped cross-validation (by session)
- [x] Implement `voice_score(...)`
- [x] Experiment E3: SVM vs. Random Forest
- [x] Voice-only results: confusion matrix, ROC, EER (using M3's `evaluate`)

**Bonus 1: Speech Command System**
- [x] MFCC-based isolated-word recognizer (SVM and KNN) with an "unknown" class
- [x] Command accuracy and confusion matrix (experiment E8)
- [x] Permission table implemented in code (role + level reached):

| User / state | Allowed commands |
|---|---|
| Member 1 (Super Admin) | Open, Lock, Logs, Status |
| Member 2 (Operator) | Open, Lock, Status |
| Member 3 (Auditor) | Logs, Status |
| Unverified / failed | **None**, "Access Denied" |

- [x] `allowed_commands(user_id, level_reached)` and `recognize_command(audio)`
- [x] Unverified users cannot run any protected action (test and show this in the demo)
- [x] "Permission Denied" message when a recognized command is not allowed for that user
- [x] Session timeout and audit log (time, user, result, command)

**Real-time support (Bonus 2)**
- [x] Microphone recording function (2-3 s) that feeds the voice pipeline (with M3's GUI)

### Must be able to explain (besides own part)
Face features (HOG, LBP, PCA), the fingerprint pipeline, how early fusion and the hierarchy decide, why session-based splitting avoids leakage.

---

## 7. Member 3: Fusion + Hierarchy + Evaluation + Real-time GUI

### Responsibilities
Evaluation framework, early fusion, hierarchical pipeline, results comparison, real-time GUI integration, dataset structure and splits, and final integration.

### Checklist
**Start first (others depend on it)**
- [ ] Write `evaluate.py` early: confusion matrix, accuracy, precision, recall, F1, FAR, FRR, TAR, ROC/AUC, EER, plots
- [ ] Set up the dataset folder structure, naming rules, and the **session-based split** (`GroupKFold`)
- [ ] Hold out unseen impostor persons for the final open-set test

**Early Fusion (Mode C)**
- [ ] Scale face and voice vectors separately (fit on training data only)
- [ ] Reduce face dimension (PCA) so blocks have comparable size
- [ ] Concatenate into one vector, with optional block weights tuned on validation data
- [ ] Train SVM and Random Forest on the fused vector
- [ ] Implement `fuse(...)` and `fusion_score(...)`
- [ ] Optional: late fusion (score averaging) as comparison (experiment E6)

**Hierarchical pipeline (Mode D)**
- [ ] Level 1: face score, reject if `s_face < T_low`
- [ ] Level 2: voice + fusion score, reject if `s_fusion < T_fusion`
- [ ] Level 3: fingerprint match (with M1's function)
- [ ] Choose thresholds on **validation data only**, then freeze them
- [ ] Implement `hierarchical_verify(...)` returning the decision and level reached
- [ ] Compare hierarchical vs. flat (always all modalities): FAR, FRR, EER, average processing time

**Evaluation and comparison**
- [ ] Run all experiments E1-E8 with the team and collect results
- [ ] Build the main comparison table (face-only, voice-only, fusion, hierarchical, with and without fingerprint)
- [ ] Prepare confusion matrices and ROC curves for every system
- [ ] Write the metrics explanation section (what each metric means and why it suits biometrics)
- [ ] Robustness test: lighting changes and noisy audio (E7)

**Bonus 2: Real-time GUI**
- [ ] GUI (OpenCV or Tkinter): live camera, claimed-identity selector, passphrase recording button, "Scan Fingerprint" button, status and result display
- [ ] Frame voting for stable face decisions
- [ ] Connect the command panel (M2) so commands appear only in Admin Mode
- [ ] Show access level reached (Level 1 / 2 / 3)

**Integration**
- [ ] Merge all modules, run end-to-end tests
- [ ] Final demo script and a backup recording of the demo in case live demo fails

### Must be able to explain (besides own part)
Every feature extractor (HOG, LBP, PCA, MFCC, ZCR, RMS), fingerprint pipeline, command recognizer and permissions.

---

## 8. Report Split

| Section | Owner |
|---|---|
| 1. Introduction and problem statement | M3 |
| 2. System design (architecture, hierarchy, access levels) | M3 |
| 3. Dataset (collection, labeling, splits, impostors, consent) | All (each writes own data part) |
| 4a. Face features and models | M1 |
| 4b. Fingerprint (Bonus 4) | M1 |
| 4c. Voice features and models | M2 |
| 4d. Speech commands and permissions (Bonus 1) | M2 |
| 4e. Early fusion | M3 |
| 5. Evaluation metrics and why they suit biometrics | M3 (reviewed by all) |
| 6. Results and comparison (tables, confusion matrices, ROC) | M3 (each adds own modality results) |
| 7. Real-time system (Bonus 2) | M3 |
| 8. Limitations and future work | All |
| 9. Team contributions and conclusion | All |

---

## 9. Cross-Training Plan (so everyone understands everything)

- **Session A (about Nov 9):** each member presents their module to the other two (features, models, results, code walkthrough, 15-20 min each).
- **Session B (about Nov 11-12):** swap roles: each member runs and explains **another member's** module live.
- **Mock viva (Nov 12):** teammates ask each other random questions. Everyone must answer on any part.

Questions every member must be able to answer:
- What is the difference between verification and identification?
- What do HOG, LBP, PCA, MFCC, ZCR and RMS compute, and why were they chosen?
- Why do we scale each modality separately before fusion?
- What are FAR, FRR, EER and ROC, and why is FAR critical for security?
- How were thresholds chosen, and why only on validation data?
- How did we avoid data leakage?
- Why does the hierarchy save time, and what happens when Level 1 fails?
- How are command permissions enforced, and what blocks an unverified user?
- How does the fingerprint part work, and what is simulated?
- Why is Deep Learning not used, and what traditional alternative did we use?

---

## 10. Timeline

| Dates | M1 (Face + Fingerprint) | M2 (Voice + Commands) | M3 (Fusion + Eval + GUI) |
|---|---|---|---|
| **Nov 5-6** | Record faces, collect impostor faces, set up face preprocessing | Record passphrase and commands, collect impostor voices, set up audio preprocessing | Folder structure, session splits, write `evaluate.py` |
| **Nov 7-8** | HOG, LBP, PCA, SVM/KNN, Mode A | MFCC, ZCR, RMS, SVM/RF, Mode B | Fusion code skeleton, help test M1/M2 outputs |
| **Nov 9** | Face results + cross-training Session A | Voice results + cross-training Session A | Early fusion (Mode C), thresholds, hierarchy (Mode D) |
| **Nov 10** | Fingerprint dataset and model, live face function | Command recognizer, permissions, mic function | Real-time GUI, connect all modules |
| **Nov 11** | Fingerprint integration, experiments | Command experiments (E8), audit log, timeout | Run all experiments, comparison table, robustness (E7) |
| **Nov 12** | Report sections, Session B, mock viva | Report sections, Session B, mock viva | Report integration, demo script, Session B, mock viva |
| **Nov 13** | Final checks and submission | Final checks and submission | Final checks and submission |

**If time gets short, priority order:** core (Modes A-D + evaluation) → Bonus 3 → Bonus 2 → Bonus 1 → Bonus 4.

---

## 11. Definition of Done

A part is done only when:
1. The code runs from a clean start without errors.
2. Its results are produced with the **session-based split** (no leakage).
3. Its metrics and confusion matrix are in the results folder.
4. A teammate has reviewed the code and can explain it.
5. Its report section is written.

---

## 12. Final Pre-Submission Checklist

- [ ] No Deep Learning libraries or models used (only scikit-learn, OpenCV, scikit-image, librosa, NumPy, etc.)
- [ ] All four modes work: Face-only, Voice-only, Early Fusion, Hierarchical
- [ ] All bonuses demonstrated: commands with permissions, real-time, custom dataset, fingerprint
- [ ] Unverified user is blocked from commands (demo case)
- [ ] An intruder is rejected (demo case)
- [ ] Comparison table, confusion matrices, ROC curves included
- [ ] Metrics explained in the report
- [ ] Limitations stated honestly (small dataset, simulated fingerprints, spoofing risk)
- [ ] Each member rehearsed explaining the full system
- [ ] Backup demo recording ready
