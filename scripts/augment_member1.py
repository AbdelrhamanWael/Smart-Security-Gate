import os
import glob
import cv2
import numpy as np

# Load Haar Cascade to ensure every generated image still has a detectable face
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

def has_face(img):
    """Checks if Haar Cascade detects a face in the image."""
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    return len(faces) > 0

def apply_zoom(img, scale=1.15, shift_x=0, shift_y=0):
    """Zooms in (>1.0) or out (<1.0) with optional off-center spatial shift."""
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if scale >= 1.0:
        crop_x = np.clip((new_w - w) // 2 + shift_x, 0, max(0, new_w - w))
        crop_y = np.clip((new_h - h) // 2 + shift_y, 0, max(0, new_h - h))
        return resized[crop_y:crop_y + h, crop_x:crop_x + w]
    else:
        pad_w = w - new_w
        pad_h = h - new_h
        left = np.clip(pad_w // 2 + shift_x, 0, pad_w)
        right = pad_w - left
        top = np.clip(pad_h // 2 + shift_y, 0, pad_h)
        bottom = pad_h - top
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT)

def apply_rotation(img, angle):
    """Rotates the image around its center."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_3d_perspective(img, mode="yaw_right", strength=1.0):
    """Applies a strong 3D camera perspective warp (Yaw or Pitch)."""
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    s = 0.08 * strength
    if mode == "yaw_right":
        pts2 = np.float32([
            [w * s, h * (s * 0.6)],
            [w * (1 - s * 0.3), 0],
            [w * s, h * (1 - s * 0.6)],
            [w * (1 - s * 0.3), h]
        ])
    else:  # pitch_top
        pts2 = np.float32([
            [w * s, h * (s * 0.5)],
            [w * (1 - s), h * (s * 0.5)],
            [0, h],
            [w, h]
        ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_shear(img, shear_x=0.10):
    """Applies horizontal affine shear."""
    h, w = img.shape[:2]
    M = np.float32([[1, shear_x, -shear_x * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_lens_wave(img, amp=6.0):
    """Applies wide-angle barrel/wave geometric distortion."""
    h, w = img.shape[:2]
    x_c, y_c = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_c + amp * np.sin(2 * np.pi * y_c / (h / 2.5))).astype(np.float32)
    map_y = (y_c + (amp * 0.6) * np.cos(2 * np.pi * x_c / (w / 2.5))).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_face_cutout(img, region="chin"):
    """Applies partial occlusion cutout directly on the detected face region."""
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        if region == "chin":
            # Occlude bottom 22% of the face (chin / mouth corner)
            cy1 = y + int(fh * 0.76)
            cy2 = y + int(fh * 0.98)
            cx1 = x + int(fw * 0.20)
            cx2 = x + int(fw * 0.80)
            out[cy1:cy2, cx1:cx2] = (45, 45, 45)
        elif region == "cheek":
            # Occlude side cheek patch
            cy1 = y + int(fh * 0.48)
            cy2 = y + int(fh * 0.72)
            cx1 = x + int(fw * 0.62)
            cx2 = x + int(fw * 0.90)
            out[cy1:cy2, cx1:cx2] = (200, 200, 200)
    return out

def apply_split_shadow(img):
    """Applies a strong directional shadow across half of the image."""
    h, w = img.shape[:2]
    # Sharp transition gradient across the face
    x_norm = np.linspace(-1, 1, w)
    mask = 0.35 + 0.65 * (1 / (1 + np.exp(-6 * x_norm)))
    mask = mask.astype(np.float32).reshape(1, w, 1)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

def generate_compound_variations(img):
    """
    Generates 6 radical compound transformations per original image.
    Each combines multiple geometric & structural operations so the change is visually unmistakable.
    """
    variations = {}

    # 1. Compound 1: Horizontal Flip + Strong Positive Tilt (+12 deg) + 16% Zoom In
    for angle in [12, 9, 6]:
        c1 = cv2.flip(img, 1)
        c1 = apply_rotation(c1, angle)
        c1 = apply_zoom(c1, scale=1.16, shift_x=15, shift_y=-10)
        if has_face(c1):
            variations["comp1_flip_rot_zoom"] = c1
            break

    # 2. Compound 2: 3D Side Perspective (Yaw) + Off-Center Spatial Shift + Zoom
    for strength in [1.0, 0.7, 0.5]:
        c2 = apply_3d_perspective(img, mode="yaw_right", strength=strength)
        c2 = apply_zoom(c2, scale=1.12, shift_x=-20, shift_y=12)
        if has_face(c2):
            variations["comp2_3d_yaw_shift"] = c2
            break

    # 3. Compound 3: Flip + Negative Tilt (-12 deg) + Strong Split Half-Shadow
    for angle in [-12, -9, -6]:
        c3 = cv2.flip(img, 1)
        c3 = apply_rotation(c3, angle)
        c3 = apply_split_shadow(c3)
        if has_face(c3):
            variations["comp3_flip_tilt_shadow"] = c3
            break

    # 4. Compound 4: Zoom Out (0.86x) + Affine Shear + Partial Chin/Mask Occlusion
    for shear in [0.11, 0.07, 0.04]:
        c4 = apply_zoom(img, scale=0.86, shift_x=18, shift_y=10)
        c4 = apply_shear(c4, shear_x=shear)
        c4 = apply_face_cutout(c4, region="chin")
        if has_face(c4):
            variations["comp4_zoomout_shear_cutout"] = c4
            break

    # 5. Compound 5: 3D Top-Down Pitch + Lens Wave Distortion + Rotation (+8 deg)
    for amp in [6.0, 4.0, 2.5]:
        c5 = apply_3d_perspective(img, mode="pitch_top", strength=0.85)
        c5 = apply_lens_wave(c5, amp=amp)
        c5 = apply_rotation(c5, 8)
        if has_face(c5):
            variations["comp5_3d_pitch_lenswave"] = c5
            break

    # 6. Compound 6: Flip + Close-Up Zoom (1.22x) + Cheek Glare/Cutout + Negative Shear
    for scale in [1.22, 1.15, 1.10]:
        c6 = cv2.flip(img, 1)
        c6 = apply_zoom(c6, scale=scale, shift_x=-15, shift_y=-12)
        c6 = apply_shear(c6, shear_x=-0.08)
        c6 = apply_face_cutout(c6, region="cheek")
        if has_face(c6):
            variations["comp6_flip_closeup_cutout"] = c6
            break

    return variations

def generate_webcam_variations(img):
    """
    Generates realistic live-webcam variations (mirroring, head tilt, camera distance,
    warm/dim/bright lighting, slight sensor blur) so the trained model recognizes the user
    instantly on a real webcam with >92-99% confidence while keeping radical variations in the folder.
    """
    cam_vars = {}

    # 1. Mirror Flip (Webcams often mirror or shift horizontal pose)
    m1 = cv2.flip(img, 1)
    if has_face(m1):
        cam_vars["cam1_mirror"] = m1

    # 2. Slight Head Tilt (+5 deg) + Slight Closer Distance (1.06x)
    m2 = apply_rotation(img, 5)
    m2 = apply_zoom(m2, scale=1.06, shift_x=5, shift_y=-4)
    if has_face(m2):
        cam_vars["cam2_tilt_pos"] = m2

    # 3. Slight Head Tilt (-5 deg) + Slight Further Distance (0.94x)
    m3 = apply_rotation(img, -5)
    m3 = apply_zoom(m3, scale=0.94, shift_x=-5, shift_y=4)
    if has_face(m3):
        cam_vars["cam3_tilt_neg"] = m3

    # 4. Brighter Room Lighting (Gamma 0.8 + Contrast 1.12)
    lut_bright = np.array([np.clip(((i / 255.0) ** 0.80) * 255.0 * 1.08, 0, 255) for i in range(256)], dtype=np.uint8)
    m4 = cv2.LUT(img, lut_bright)
    if has_face(m4):
        cam_vars["cam4_bright_room"] = m4

    # 5. Dimmer Room Lighting + Mirror + Slight Motion/Sensor Softness
    lut_dim = np.array([np.clip(((i / 255.0) ** 1.25) * 255.0 * 0.92, 0, 255) for i in range(256)], dtype=np.uint8)
    m5 = cv2.LUT(cv2.flip(img, 1), lut_dim)
    m5 = cv2.GaussianBlur(m5, (3, 3), 0.6)
    if has_face(m5):
        cam_vars["cam5_dim_mirror"] = m5

    return cam_vars

def organize_and_augment(base_dir="data/face/member1", member_name="member1"):
    print("=" * 70)
    print(f"  [{member_name.upper()}] STEP 1: ORGANIZING SESSIONS & REMOVING OLD AUGS")
    print("=" * 70)

    s1_dir = os.path.join(base_dir, "session1")
    s2_dir = os.path.join(base_dir, "session2")
    s3_dir = os.path.join(base_dir, "session3")

    for s_dir in [s1_dir, s2_dir, s3_dir]:
        os.makedirs(s_dir, exist_ok=True)
        for old_aug in glob.glob(os.path.join(s_dir, "*_aug_*.*")):
            os.remove(old_aug)

    # 2. Check if WIN_ files exist in session1 and session2 and split them 4 + 4
    s1_win = sorted(glob.glob(os.path.join(s1_dir, "WIN_*.jpg")))
    s2_win = sorted(glob.glob(os.path.join(s2_dir, "WIN_*.jpg")))

    if len(s1_win) == 8 and len(s2_win) == 8:
        print(" -> Detected 8 identical WIN_*.jpg images in session1 and session2.")
        print(" -> Keeping first 4 in session1 and second 4 in session2 to prevent data leakage...")
        for f in s1_win[4:]:
            os.remove(f)
        for f in s2_win[:4]:
            os.remove(f)

    # Helper to find original images (jpg, jpeg, png, bmp) excluding _aug_
    def get_originals(folder):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        for f in glob.glob(os.path.join(folder, "*.*")):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts and "_aug_" not in os.path.basename(f):
                files.append(f)
        return sorted(files)

    # 3. If all images are placed in session1 while session2 and session3 are empty, distribute evenly!
    s1_origs = get_originals(s1_dir)
    s2_origs = get_originals(s2_dir)
    s3_origs = get_originals(s3_dir)

    if len(s1_origs) >= 3 and len(s2_origs) == 0 and len(s3_origs) == 0:
        total_n = len(s1_origs)
        n1 = (total_n + 2) // 3
        n2 = (total_n - n1 + 1) // 2
        n3 = total_n - n1 - n2
        print(f" -> Distributing {total_n} images from session1 across 3 sessions: ({n1}, {n2}, {n3})...")
        loaded_all = [(f, cv2.imread(f)) for f in s1_origs]
        for f, _ in loaded_all:
            os.remove(f)
        splits = [
            (s1_dir, loaded_all[:n1]),
            (s2_dir, loaded_all[n1:n1 + n2]),
            (s3_dir, loaded_all[n1 + n2:])
        ]
        for target_dir, items in splits:
            for idx, (_, img) in enumerate(items):
                if img is not None:
                    cv2.imwrite(os.path.join(target_dir, f"img_{idx:03d}.jpg"), img)

    # 4. Standardize original filenames in each session to img_000.jpg, img_001.jpg, ...
    for s_name, s_dir in [("session1", s1_dir), ("session2", s2_dir), ("session3", s3_dir)]:
        orig_files = get_originals(s_dir)
        loaded = [(f, cv2.imread(f)) for f in orig_files]
        for f, _ in loaded:
            os.remove(f)
        for idx, (_, img) in enumerate(loaded):
            if img is not None:
                cv2.imwrite(os.path.join(s_dir, f"img_{idx:03d}.jpg"), img)
        print(f" -> {s_name}: Standardized {len(loaded)} unique original images.")

    print("\n" + "=" * 70)
    print(f"  [{member_name.upper()}] STEP 2: GENERATING RADICAL COMPOUND AUGMENTATIONS")
    print("=" * 70)

    total_originals = 0
    total_generated = 0

    for s_name, s_dir in [("session1", s1_dir), ("session2", s2_dir), ("session3", s3_dir)]:
        orig_files = get_originals(s_dir)
        # Balance session size so each session has ~21-28 total images (total ~76 genuine)
        if len(orig_files) >= 8:
            max_vars = 2
        elif len(orig_files) == 5:
            max_vars = 5
        else:
            max_vars = 6

        session_gen = 0
        session_cam = 0
        for idx_img, f_path in enumerate(orig_files):
            total_originals += 1
            img = cv2.imread(f_path)
            base_name = os.path.splitext(os.path.basename(f_path))[0]
            
            # A) Keep Radical Compound Variations in the folder
            variations = generate_compound_variations(img)
            all_keys = list(variations.keys())
            selected_keys = [all_keys[(idx_img * max_vars + k) % len(all_keys)] for k in range(min(max_vars, len(all_keys)))]
            for aug_key in selected_keys:
                aug_img = variations[aug_key]
                out_path = os.path.join(s_dir, f"{base_name}_aug_{aug_key}.jpg")
                cv2.imwrite(out_path, aug_img)
                session_gen += 1
                total_generated += 1

            # B) Also generate realistic webcam variations so live camera is super accurate
            cam_variations = generate_webcam_variations(img)
            for cam_key, cam_img in cam_variations.items():
                out_path = os.path.join(s_dir, f"{base_name}_aug_{cam_key}.jpg")
                cv2.imwrite(out_path, cam_img)
                session_cam += 1
                total_generated += 1

        print(f" -> {s_name}: {len(orig_files)} originals + {session_gen} radical + {session_cam} webcam augs = {len(orig_files) + session_gen + session_cam} total")

    print("\n" + "=" * 70)
    print(f"FINAL DATASET SUMMARY ({member_name.upper()}):")
    print(f" - Unique Original Images:       {total_originals}")
    print(f" - Radical Compound Augmented:   {total_generated}")
    print(f" - Total Genuine ({member_name}):      {total_originals + total_generated}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    organize_and_augment("data/face/member1", "member1")
    if os.path.exists("data/face/member2"):
        organize_and_augment("data/face/member2", "member2")
