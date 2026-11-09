import os
import sys
import glob
import joblib
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_pipeline import hog_feat, lbp_feat
from src.face_models import calculate_biometric_metrics

# In-memory cache for fast fingerprint verification
_FP_MODELS = {}
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def preprocess_fingerprint(img_or_path, size=(128, 128)):
    """
    Preprocesses a fingerprint image:
      1. Reads image or converts BGR to Grayscale.
      2. Detects and crops the active fingerprint ridge ROI (removes white background margins).
      3. Resizes to standardized 128x128 resolution.
      4. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to sharpen ridges.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_or_path, np.ndarray):
        if len(img_or_path.shape) == 3:
            img = cv2.cvtColor(img_or_path, cv2.COLOR_BGR2GRAY)
        else:
            img = img_or_path.copy()
    else:
        return None

    if img is None:
        return None

    # Find fingerprint foreground ROI (ridges are darker than white background)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Keep ROI if it covers a reasonable area
        if w > 30 and h > 30:
            img = img[y:y + h, x:x + w]

    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    enhanced = _CLAHE.apply(resized)
    return enhanced

def extract_fingerprint_features(fp_gray, mode="both"):
    """
    Extracts HOG (ridge orientation flow) and Uniform LBP (minutiae/pore micro-texture).
    """
    if mode == "hog":
        return hog_feat(fp_gray)
    elif mode == "lbp":
        return lbp_feat(fp_gray)
    elif mode == "both":
        return np.concatenate([hog_feat(fp_gray), lbp_feat(fp_gray)])
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_fingerprint_dataset(data_dir="data/fingerprint", target_user="member1", feature_mode="both"):
    """
    Loads fingerprint dataset for a target user:
      - Genuine (y=1): target_user's impressions (grouped by session_id)
      - Impostor (y=0): impostor fingers (grouped by finger_id for open-set evaluation)
    """
    all_files = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    X_list, y_list, groups_list, meta_list = [], [], [], []

    for path in sorted(all_files):
        norm_path = path.replace("\\", "/")
        parts = norm_path.split("/")

        if "impostors" in parts:
            idx = parts.index("impostors")
            person_id = parts[idx + 1]
            session_id = parts[idx + 2]
        else:
            idx = parts.index("fingerprint")
            person_id = parts[idx + 1]
            session_id = parts[idx + 2]
            # For binary verification of target_user, skip augmented copies of OTHER team members
            # so impostor set remains pure and balanced (56 impostor impressions + 16 raw other members = 72)
            if person_id.lower() != target_user.lower() and "_aug_" in os.path.basename(path):
                continue

        fp_gray = preprocess_fingerprint(path)
        if fp_gray is None:
            continue

        feats = extract_fingerprint_features(fp_gray, mode=feature_mode)
        is_target = 1 if person_id.lower() == target_user.lower() else 0
        group_id = session_id if is_target else f"imp_{person_id}"

        X_list.append(feats)
        y_list.append(is_target)
        groups_list.append(group_id)
        meta_list.append({
            "path": path,
            "person_id": person_id,
            "session_id": session_id,
            "is_target": is_target
        })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    return X, y, groups, meta_list

def build_fingerprint_pipeline(model_type="svm", n_components=30):
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42))
    ]
    if model_type == "svm":
        steps.append(("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)))
    elif model_type == "knn":
        steps.append(("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(steps)

def train_and_evaluate_fingerprint(X, y, groups, model_type="svm", n_splits=3):
    pipeline = build_fingerprint_pipeline(model_type=model_type, n_components=min(30, len(X) - 1))
    cv = GroupKFold(n_splits=n_splits)
    y_scores = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return calculate_biometric_metrics(y, y_scores)

def save_fingerprint_model(pipeline, user_id="member1", model_type="svm", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"fingerprint_{user_id}_{model_type}.joblib")
    joblib.dump(pipeline, path)
    _FP_MODELS[f"{user_id}_{model_type}"] = pipeline
    print(f"Saved fingerprint model -> {path}")

def fingerprint_score(image_or_path, user_id="member1", model_type="svm", models_dir="models"):
    """
    Shared Interface Function for Member 3 (Level 3 Hierarchical Gate & GUI).
    Accepts either an image file path (str) or a BGR/Grayscale numpy array.
    Returns match probability in [0.0, 1.0].
    """
    fp_gray = preprocess_fingerprint(image_or_path)
    if fp_gray is None:
        return 0.0

    feats = extract_fingerprint_features(fp_gray, mode="both")
    key = f"{user_id}_{model_type}"
    if key not in _FP_MODELS:
        path = os.path.join(models_dir, f"fingerprint_{user_id}_{model_type}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fingerprint model not found: {path}. Run run_fingerprint_bonus4.py first.")
        _FP_MODELS[key] = joblib.load(path)

    model = _FP_MODELS[key]
    return float(model.predict_proba([feats])[0, 1])

