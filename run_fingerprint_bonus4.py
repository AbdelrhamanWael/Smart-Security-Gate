import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.fingerprint import (
    load_fingerprint_dataset,
    build_fingerprint_pipeline,
    train_and_evaluate_fingerprint,
    save_fingerprint_model,
    fingerprint_score
)

def main():
    print("=" * 70)
    print("   BONUS 4: FINGERPRINT BIOMETRIC VERIFICATION TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load dataset for Member 1 evaluation
    X, y, groups, meta = load_fingerprint_dataset(target_user="member1")
    print(f"Dataset Loaded: {len(X)} fingerprint impressions")
    print(f" - Genuine (member1 - Finger 101): {int((y == 1).sum())}")
    print(f" - Impostors (Other Fingers):      {int((y == 0).sum())}")
    print(f" - Feature Dimension (HOG + LBP):  {X.shape[1]}")

    # 2. Evaluate SVM (RBF) via 3-Fold GroupKFold
    print("\n--- 3-Fold GroupKFold Cross-Validation: Fingerprint SVM (RBF) ---")
    svm_res = train_and_evaluate_fingerprint(X, y, groups, model_type="svm", n_splits=3)
    print(f"Accuracy:  {svm_res['accuracy'] * 100:.2f}%")
    print(f"Precision: {svm_res['precision'] * 100:.2f}%")
    print(f"Recall:    {svm_res['recall'] * 100:.2f}%")
    print(f"FAR:       {svm_res['FAR'] * 100:.2f}%")
    print(f"FRR:       {svm_res['FRR'] * 100:.2f}%")
    print(f"EER:       {svm_res['EER'] * 100:.2f}%  (Operating Threshold: {svm_res['EER_threshold']:.4f})")
    print(f"Confusion Matrix: {svm_res['confusion_matrix']}")

    # 3. Evaluate KNN (k=5) via 3-Fold GroupKFold
    print("\n--- 3-Fold GroupKFold Cross-Validation: Fingerprint KNN (k=5) ---")
    knn_res = train_and_evaluate_fingerprint(X, y, groups, model_type="knn", n_splits=3)
    print(f"Accuracy:  {knn_res['accuracy'] * 100:.2f}%")
    print(f"Precision: {knn_res['precision'] * 100:.2f}%")
    print(f"Recall:    {knn_res['recall'] * 100:.2f}%")
    print(f"FAR:       {knn_res['FAR'] * 100:.2f}%")
    print(f"FRR:       {knn_res['FRR'] * 100:.2f}%")
    print(f"EER:       {knn_res['EER'] * 100:.2f}%  (Operating Threshold: {knn_res['EER_threshold']:.4f})")
    print(f"Confusion Matrix: {knn_res['confusion_matrix']}")

    # 4. Train and save production fingerprint models for all 3 team members
    print("\n--- Training & Saving Production Fingerprint Models (All 3 Members) ---")
    for member in ["member1", "member2", "member3"]:
        X_m, y_m, _, _ = load_fingerprint_dataset(target_user=member)
        for m_type in ["svm", "knn"]:
            pipe = build_fingerprint_pipeline(model_type=m_type, n_components=min(30, len(X_m) - 1))
            pipe.fit(X_m, y_m)
            save_fingerprint_model(pipe, user_id=member, model_type=m_type)

    # 5. Test fingerprint_score() Interface (Simulating GUI Upload)
    print("\n--- Testing fingerprint_score() Interface (Level 3 Gate Simulation) ---")
    genuine_fp = [m["path"] for m in meta if m["is_target"] == 1][0]
    impostor_fp = [m["path"] for m in meta if m["is_target"] == 0][0]

    gen_score = fingerprint_score(genuine_fp, user_id="member1", model_type="svm")
    imp_score = fingerprint_score(impostor_fp, user_id="member1", model_type="svm")

    print(f"Uploaded Genuine Fingerprint ({genuine_fp}):  Score = {gen_score:.4f} -> {'ACCESS GRANTED (Level 3)' if gen_score >= 0.5 else 'DENIED'}")
    print(f"Uploaded Impostor Fingerprint ({impostor_fp}): Score = {imp_score:.4f} -> {'ACCESS GRANTED' if imp_score >= 0.5 else 'ACCESS DENIED (Mismatch)'}")
    print("\n[SUCCESS] Bonus 4 (Fingerprint Modality) is 100% complete and ready for Member 3!")

if __name__ == "__main__":
    main()

