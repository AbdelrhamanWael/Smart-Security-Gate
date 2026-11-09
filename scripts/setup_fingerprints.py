import os
import io
import glob
import zipfile
import urllib.request
import cv2
import numpy as np

FVC_ZIP_URL = "https://raw.githubusercontent.com/k3mlol/fingercode/master/DB1_B.zip"
OUTPUT_DIR = "data/fingerprint"

def augment_fingerprint(img_gray):
    """
    Creates realistic sensor variations for a fingerprint impression:
      1. Slight rotation (+4 deg) - finger placed at a slight clockwise angle
      2. Slight rotation (-4 deg) - finger placed counter-clockwise
      3. Slight pressure shift (dilation/erosion or gamma darkening for heavy press)
      4. Slight light press (gamma brightening + tiny shift)
      5. Small spatial shift (dx=+4, dy=-4)
    """
    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    augs = {}

    # 1. Rotation +4 deg
    M_pos = cv2.getRotationMatrix2D(center, 4, 1.0)
    augs["rot_pos"] = cv2.warpAffine(img_gray, M_pos, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 2. Rotation -4 deg
    M_neg = cv2.getRotationMatrix2D(center, -4, 1.0)
    augs["rot_neg"] = cv2.warpAffine(img_gray, M_neg, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 3. Heavy finger pressure (darker ridges) + small shift
    M_s1 = np.float32([[1, 0, 4], [0, 1, -3]])
    shifted1 = cv2.warpAffine(img_gray, M_s1, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    augs["heavy_press"] = cv2.convertScaleAbs(shifted1, alpha=0.90, beta=-10)

    # 4. Light finger pressure (lighter ridges) + small shift
    M_s2 = np.float32([[1, 0, -4], [0, 1, 3]])
    shifted2 = cv2.warpAffine(img_gray, M_s2, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    augs["light_press"] = cv2.convertScaleAbs(shifted2, alpha=1.08, beta=10)

    # 5. Slight scale/zoom (1.05x)
    new_w, new_h = int(w * 1.05), int(h * 1.05)
    resized = cv2.resize(img_gray, (new_w, new_h))
    cx, cy = (new_w - w) // 2, (new_h - h) // 2
    augs["zoom"] = resized[cy:cy + h, cx:cx + w]

    return augs

def download_and_setup_fingerprints():
    print("=" * 65)
    print("  BONUS 4: DOWNLOADING & ORGANIZING FVC FINGERPRINT DATASET")
    print("=" * 65)
    
    req = urllib.request.Request(FVC_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading FVC DB1_B benchmark archive (~5.6 MB)...")
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
    print("Download complete! Extracting impressions...")

    # Clean old directory if exists
    if os.path.exists(OUTPUT_DIR):
        for old_f in glob.glob(os.path.join(OUTPUT_DIR, "**", "*.jpg"), recursive=True):
            os.remove(old_f)

    mapping = {
        "101": "member1",
        "102": "member2",
        "103": "member3"
    }

    originals_count = 0
    aug_count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = sorted([f for f in zf.namelist() if f.lower().endswith((".tif", ".bmp", ".png", ".jpg"))])

        for f in file_list:
            base_name = os.path.basename(f)
            if "_" not in base_name:
                continue

            finger_id, impression_id = os.path.splitext(base_name)[0].split("_")
            img_data = zf.read(f)
            img_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            imp_num = int(impression_id)
            # Distribute 8 impressions across 3 sessions for 3-Fold GroupKFold:
            # 1..3 -> session1, 4..6 -> session2, 7..8 -> session3
            if imp_num <= 3:
                session = "session1"
            elif imp_num <= 6:
                session = "session2"
            else:
                session = "session3"

            if finger_id in mapping:
                person = mapping[finger_id]
                dest_dir = os.path.join(OUTPUT_DIR, person, session)
                is_team_member = True
            else:
                person = f"finger_{finger_id}"
                dest_dir = os.path.join(OUTPUT_DIR, "impostors", person, session)
                is_team_member = False

            os.makedirs(dest_dir, exist_ok=True)
            out_file = os.path.join(dest_dir, f"img_{imp_num:03d}.jpg")
            cv2.imwrite(out_file, img)
            originals_count += 1

            # Augment team members' impressions within their respective session
            # so each member has 8 originals + 40 augmented = 48 genuine vs 56 impostors
            if is_team_member:
                augs = augment_fingerprint(img)
                for aug_name, aug_img in augs.items():
                    aug_file = os.path.join(dest_dir, f"img_{imp_num:03d}_aug_{aug_name}.jpg")
                    cv2.imwrite(aug_file, aug_img)
                    aug_count += 1

    print(f"\n[SUCCESS] Fingerprint dataset ready in '{OUTPUT_DIR}':")
    print(f" - Original FVC impressions: {originals_count} (10 fingers x 8 impressions)")
    print(f" - Team member impressions:  48 per member (8 originals + 40 sensor augmentations across 3 sessions)")
    print(f" - Impostor impressions:     56 (7 unseen fingers x 8 impressions)")
    print("=" * 65)

if __name__ == "__main__":
    download_and_setup_fingerprints()

