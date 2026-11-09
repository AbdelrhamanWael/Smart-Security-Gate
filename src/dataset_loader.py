import os
import sys
import glob
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.face_pipeline import crop_face, extract_face_features
def load_face_dataset(data_dir="data/face", target_user="member1", feature_mode="both", include_extreme_augs=False):
    """
    Loops through data/face/ and extracts features.
    
    Group assignment logic:
        - Genuine (you): group = session_id ("session1", "session2", "session3")
        - Impostors:     group = person_id ("imp_Tony_Blair", "imp_Colin_Powell", etc.)
      
    Why:
        Ensures an impostor person is NEVER split across train and test sets simultaneously.
        By default (`include_extreme_augs=False`), extreme face-occluding/warping compound files
        (`_aug_comp`) remain in the session folders for visual inspection, while the model trains
        on clean originals + realistic webcam augmentations (`_aug_cam`) for high live-camera accuracy.
    """

    # Get All Images

    all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in all_files if os.path.splitext(p)[1].lower() in valid_exts]
    
    X_list, y_list, groups_list, meta_list = [], [], [], []
    skipped = 0

    for path in sorted(image_paths):
        norm_path = path.replace("\\", "/")
        parts = norm_path.split("/")

        if "impostors" in parts:
            idx = parts.index("impostors")
            person_id = parts[idx + 1]
            session_id = parts[idx + 2]
        elif "imposter" in parts:
            idx = parts.index("imposter")
            person_id = parts[idx + 1]
            session_id = parts[idx + 2]
        else:
            idx = parts.index("face")
            person_id = parts[idx + 1]
            session_id = parts[idx + 2]

        # Skip augmented copies of OTHER team members when treating them as impostors
        basename = os.path.basename(path)
        if person_id.lower() != target_user.lower() and "_aug_" in basename:
            continue

        # Skip extreme distorted compound augmentations (_aug_comp*) when training the model
        # so face cutouts/barrel waves don't corrupt the genuine decision boundary on live webcam
        if not include_extreme_augs and "_aug_comp" in basename:
            continue

        # Load Images and Extract Features
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            skipped += 1
            continue

        face_gray = crop_face(img_bgr)
        if face_gray is None:
            skipped += 1
            continue
            
        feats = extract_face_features(face_gray, mode=feature_mode)
        is_target = 1 if person_id.lower() == target_user.lower() else 0
        
        # Group assignment rule
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
    
    print(f"Dataset Loaded: {len(X)} samples ({skipped} skipped without faces).")
    print(f" - Genuine ({target_user}): {np.sum(y == 1)}")
    print(f" - Impostors:             {np.sum(y == 0)}")
    print(f" - Feature Dimension:     {X.shape[1]}")
    
    return X, y, groups, meta_list


              