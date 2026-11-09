import os
import cv2
import numpy as np
from src.dataset_loader import load_face_dataset
from src.face_models import (
    build_face_pipeline,
    train_and_evaluate_cv,
    save_model,
    face_score
)

def evaluate_and_train_user(user_id):
    print("====================================================")
    print(f"   {user_id.upper()}: FACE VERIFICATION TRAINING & EVALUATION ")
    print("====================================================")
    
    # 1. Load data
    X, y, groups, meta = load_face_dataset(target_user=user_id)
    
    # 2. Evaluate SVM
    print(f"\n--- [{user_id}] 3-Fold GroupKFold Cross-Validation: SVM (RBF) ---")
    svm_res = train_and_evaluate_cv(X, y, groups, model_type="svm", n_splits=3)
    print(f"Accuracy:  {svm_res['accuracy'] * 100:.2f}%")
    print(f"Precision: {svm_res['precision'] * 100:.2f}%")
    print(f"Recall:    {svm_res['recall'] * 100:.2f}%")
    print(f"FAR:       {svm_res['FAR'] * 100:.2f}%")
    print(f"FRR:       {svm_res['FRR'] * 100:.2f}%")
    print(f"EER:       {svm_res['EER'] * 100:.2f}%  (Operating Threshold: {svm_res['EER_threshold']:.4f})")
    print(f"Confusion Matrix: {svm_res['confusion_matrix']}")
    
    # 3. Evaluate KNN
    print(f"\n--- [{user_id}] 3-Fold GroupKFold Cross-Validation: KNN (k=5) ---")
    knn_res = train_and_evaluate_cv(X, y, groups, model_type="knn", n_splits=3)
    print(f"Accuracy:  {knn_res['accuracy'] * 100:.2f}%")
    print(f"FAR:       {knn_res['FAR'] * 100:.2f}%")
    print(f"FRR:       {knn_res['FRR'] * 100:.2f}%")
    print(f"EER:       {knn_res['EER'] * 100:.2f}%  (Operating Threshold: {knn_res['EER_threshold']:.4f})")
    print(f"Confusion Matrix: {knn_res['confusion_matrix']}")
    
    # 4. Fit final model on all data and save
    # Give higher sample weight (2.5x) to pure original camera images so the production
    # webcam model anchors strongly around the user's true undistorted face
    print(f"\n--- [{user_id}] Training & Saving Final Production Models ---")
    sample_weights = np.array([
        2.5 if (m["is_target"] == 1 and "_aug_" not in os.path.basename(m["path"])) else 1.0
        for m in meta
    ], dtype=np.float32)

    final_svm = build_face_pipeline("svm", n_components=min(35, len(X) - 1))
    final_svm.fit(X, y, clf__sample_weight=sample_weights)
    save_model(final_svm, user_id=user_id, model_type="svm")
    
    final_knn = build_face_pipeline("knn", n_components=min(35, len(X) - 1))
    final_knn.fit(X, y)
    save_model(final_knn, user_id=user_id, model_type="knn")
    
    # 5. Live test of face_score interface
    print(f"\n--- [{user_id}] Testing face_score() Interface ---")
    genuine_path = [m["path"] for m in meta if m["is_target"] == 1][0]
    impostor_path = [m["path"] for m in meta if m["is_target"] == 0][0]
    
    gen_score = face_score(cv2.imread(genuine_path), user_id=user_id, model_type="svm")
    imp_score = face_score(cv2.imread(impostor_path), user_id=user_id, model_type="svm")
    
    print(f"Genuine Sample ({os.path.basename(genuine_path)}): Score = {gen_score:.4f}")
    print(f"Impostor Sample ({os.path.basename(impostor_path)}): Score = {imp_score:.4f}\n")

def main():
    evaluate_and_train_user("member1")
    if os.path.exists("data/face/member2/session1") and len(os.listdir("data/face/member2/session1")) > 0:
        evaluate_and_train_user("member2")
    print("All Face Verification tasks completed successfully!")

if __name__ == "__main__":
    main()