# Smart-Security-Gate

Hierarchical Multimodal Biometric Verification System for Data Center Security Gate.

## Overview
This system provides hierarchical biometric authentication using traditional machine learning algorithms (no Deep Learning) across multiple modalities:
1. **Face Verification (Mode A - Member 1)**: Haar cascade face detection, CLAHE contrast enhancement, HOG + Uniform LBP feature extraction, PCA dimensionality reduction, SVM (RBF) & KNN classifiers.
2. **Voice Verification (Mode B - Member 2)**: Polyphase resampling to 16 kHz, RMS energy Voice Activity Detection (VAD), 13 MFCCs + deltas, Zero-Crossing Rate (ZCR), RMS energy, SVM & Random Forest classifiers.
3. **Fingerprint Verification (Bonus 4 - Member 1)**: Ridge ROI segmentation, CLAHE enhancement, HOG + LBP descriptors, SVM & KNN classifiers.
4. **Speech Command System & RBAC (Bonus 1 - Member 2)**: Isolated-word speech recognizer (`open`, `lock`, `logs`, `status`, `unknown`), role-based access control matrix, session timeout, and security audit log.
5. **Real-time Live Verification (Bonus 2 - Members 1, 2, 3)**: Live webcam face verification with rolling frame voting and live microphone recording.
6. **Early Fusion (Mode C - Member 3)**: Normalized per-modality scaling, face PCA dimension balancing, tuned block weights, SVM and Random Forest on the fused vector.
7. **Hierarchical Verification Pipeline (Mode D - Member 3)**: 3-level cascade (Level 1: Face -> Level 2: Voice + Early Fusion -> Level 3: Fingerprint), frozen validation thresholds, and early rejection of impostors saving compute time.

## Project Structure
- `src/`: Modular feature extractors, dataset loaders, classifiers, fusion transformers, and verification engines:
  - `face_pipeline.py`, `face_models.py`: Face crop, HOG, LBP, PCA, SVM/KNN.
  - `voice_features.py`, `voice_verification.py`, `audio_recorder.py`: Audio preprocessing, MFCC, ZCR, RMS, SVM/RF.
  - `fingerprint.py`: Fingerprint ROI, CLAHE, HOG/LBP, SVM/KNN.
  - `command_system.py`: Speech commands, RBAC, session manager, audit logger.
  - `fusion.py`: Early fusion transformer, `fuse()`, `fusion_score()`.
  - `hierarchy.py`: Hierarchical cascade pipeline, `hierarchical_verify()`.
  - `multimodal_dataset.py`: Paired multimodal attempts with session-based split and open-set virtual impostors.
  - `evaluation.py`: Biometric metrics (Accuracy, Precision, Recall, F1, FAR, FRR, EER, AUC, ROC).
- `data/`: Multimodal dataset organized by person and session (`face`, `voice`, `commands`, `fingerprint`).
- `models/`: Trained production ML models:
  - `models/`: Face and fingerprint SVM & KNN models.
  - `models/voice/`: Voice verifiers (SVM & RF).
  - `models/commands/`: Command classifiers (SVM & KNN).
  - `models/fusion/`: Trained early fusion pipelines.
  - `models/hierarchy/`: Hierarchical bundles with frozen thresholds.
- `results/`:
  - `results/m3/`: Main multimodal comparison table (`m3_results_summary.md`), CSV, JSON, and ROC/CM figures.
  - Individual voice verification (Mode B) and speech command results.
- `logs/`: Security audit log (`audit_log.csv`).
- `reports/`: Experiment plots and ablation study results.
- `scripts/`: Data augmentation and synthetic data helpers.

## Running the System

```bash
# Member 1 Experiments (Face Mode A, E1 ablation, Bonus 4 Fingerprint)
python run_member1.py
python run_experiment_e1.py
python run_fingerprint_bonus4.py

# Member 2 Experiments (Voice Mode B, E3 SVM vs RF, E8 Speech Commands & RBAC)
python run_member2_experiments.py
python test_voice.py

# Member 3 Experiments (Early Fusion Mode C, Hierarchy Mode D, Timing, Robustness E7)
python run_member3_experiments.py
python demo_member3.py --user member1
```
