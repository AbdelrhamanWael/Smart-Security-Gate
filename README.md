# Smart-Security-Gate

Hierarchical Multimodal Biometric Verification System for Data Center Security Gate.

## Overview
This system provides hierarchical biometric authentication using traditional machine learning algorithms (no Deep Learning) across multiple modalities:
1. **Face Verification (Mode A)**: Haar cascade face detection, CLAHE contrast enhancement, HOG + Uniform LBP feature extraction, PCA dimensionality reduction, SVM (RBF) & KNN classifiers.
2. **Voice Verification (Mode B)**: Polyphase resampling to 16 kHz, RMS energy Voice Activity Detection (VAD), 13 MFCCs + deltas, Zero-Crossing Rate (ZCR), RMS energy, SVM & Random Forest classifiers.
3. **Fingerprint Verification (Bonus 4)**: Ridge ROI segmentation, CLAHE enhancement, HOG + LBP descriptors, SVM & KNN classifiers.
4. **Speech Command System & RBAC (Bonus 1)**: Isolated-word speech recognizer (`open`, `lock`, `logs`, `status`, `unknown`), role-based access control matrix, session timeout, and security audit log.
5. **Real-time Live Verification (Bonus 2)**: Live webcam face verification with rolling frame voting and live microphone recording.

## Project Structure
- `src/`: Modular feature extractors, dataset loaders, classifiers, and verification engines.
- `data/`: Multimodal dataset organized by person and session (`face`, `voice`, `commands`, `fingerprint`).
- `models/`: Trained production ML models.
- `results/`: Evaluation figures, ROC curves, confusion matrices, and experiment summaries.
- `logs/`: Security audit log (`audit_log.csv`).
- `reports/`: Experiment plots and ablation study results.
