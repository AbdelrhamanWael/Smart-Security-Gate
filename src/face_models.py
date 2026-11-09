import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_pipeline import crop_face, extract_face_features

# Global memory cache so webcam real-time frames don't lag from joblib.load
_LOADED_MODELS = {}

def build_face_pipeline(model_type="svm", n_components=30):
    """
    Why Pipeline:
      StandardScaler and PCA MUST be fit only on training folds.
      Fitting scaler/PCA on the entire dataset is data leakage.
    """
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42))
    ]
    
    if model_type == "svm":
        # RBF kernel with C=5.0 gives confident, well-separated probabilities (>92-99% for genuine webcam frames)
        steps.append(("clf", SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)))
    elif model_type == "knn":
        steps.append(("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine")))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return Pipeline(steps)

def calculate_biometric_metrics(y_true, y_scores):
    """
    Computes FAR, FRR, EER, and optimal threshold.
    
    Interview Definitions:
      - FAR (False Acceptance Rate / FPR): Intruders accepted / Total intruders.
      - FRR (False Rejection Rate / FNR): Authorized user rejected / Total genuine attempts.
      - EER (Equal Error Rate): The point where FAR == FRR. Lower EER = better system.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    
    # Point where FPR and FNR intersect
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    
    # Decisions using EER threshold
    y_pred = (y_scores >= eer_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "FRR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "EER": eer,
        "EER_threshold": eer_threshold,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

def train_and_evaluate_cv(X, y, groups, model_type="svm", n_splits=3):
    """Evaluates the model with GroupKFold cross-validation."""
    pipeline = build_face_pipeline(model_type=model_type, n_components=min(30, len(X) - 1))
    cv = GroupKFold(n_splits=n_splits)
    
    # cross_val_predict provides out-of-fold probability predictions
    y_scores = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return calculate_biometric_metrics(y, y_scores)

def save_model(pipeline, user_id="member1", model_type="svm", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"face_{user_id}_{model_type}.joblib")
    joblib.dump(pipeline, path)
    _LOADED_MODELS[f"{user_id}_{model_type}"] = pipeline
    print(f"Model saved -> {path}")

def face_score(image_bgr, user_id="member1", model_type="svm", models_dir="models"):
    """
    The function Member 3 calls.
    Returns: float (0.0 to 1.0 match probability). Returns 0.0 if no face detected.
    """
    face_gray = crop_face(image_bgr)
    if face_gray is None:
        return 0.0
        
    feats = extract_face_features(face_gray, mode="both")
    key = f"{user_id}_{model_type}"
    if key not in _LOADED_MODELS:
        path = os.path.join(models_dir, f"face_{user_id}_{model_type}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}. Train it first.")
        _LOADED_MODELS[key] = joblib.load(path)
        
    model = _LOADED_MODELS[key]
    return float(model.predict_proba([feats])[0, 1])