"""
train_isl.py  -  ISL Image Folder Trainer
==========================================
Reads images from dataset/a, dataset/b ... dataset/z
Extracts MediaPipe landmarks, augments, trains Random Forest.

Usage:
    python train_isl.py
"""

import os
import json
import time
import numpy as np
import joblib
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================================================
# CONFIG  -  change these if needed
# =============================================================================
DATASET_DIR   = "dataset"          # folder with a/, b/, c/ ... z/ inside
MODELS_DIR    = "models"           # where to save trained model
AUGMENT_TIMES = 3                  # augmentation multiplier (3 is enough for 1200 imgs)
IMG_EXTS      = (".jpg", ".jpeg", ".png", ".bmp")
# =============================================================================

os.makedirs(MODELS_DIR, exist_ok=True)
mp_hands = mp.solutions.hands

# -----------------------------------------------------------------------------
# 1. LANDMARK EXTRACTOR
# -----------------------------------------------------------------------------
def extract_landmarks(img_path):
    """
    Returns a 126-d numpy array (left + right hand landmarks)
    or None if no hand is detected.
    Tries original image first, then brightness-adjusted versions.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    variants = [
        img,
        cv2.convertScaleAbs(img, alpha=1.4, beta=30),   # brighter
        cv2.convertScaleAbs(img, alpha=0.6, beta=-10),  # darker
    ]

    for v in variants:
        rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.2,
        ) as hands:
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                continue

            left_lm  = np.zeros(63, dtype=np.float32)
            right_lm = np.zeros(63, dtype=np.float32)

            for hl, hd in zip(res.multi_hand_landmarks,
                               res.multi_handedness):
                side  = hd.classification[0].label   # "Left" or "Right"
                wrist = hl.landmark[0]
                coords = []
                for lm in hl.landmark:
                    coords.extend([
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z,
                    ])
                arr = np.array(coords, dtype=np.float32)
                if side == "Left":
                    left_lm = arr
                else:
                    right_lm = arr

            return np.concatenate([left_lm, right_lm])   # shape (126,)

    return None   # no hand found in any variant


# -----------------------------------------------------------------------------
# 2. AUGMENTATION
# -----------------------------------------------------------------------------
def augment(lm, n=3):
    """
    Generate n augmented copies of a 126-d landmark vector.
    Applies random rotation, scale, jitter, and mirror.
    """
    pts = lm.reshape(2, 21, 3)
    results = []
    for _ in range(n):
        aug = pts.copy()

        # random in-plane rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        R = np.array([
            [ np.cos(angle), -np.sin(angle), 0],
            [ np.sin(angle),  np.cos(angle), 0],
            [ 0,              0,             1],
        ])
        aug = aug @ R.T

        # random scale (90% to 110%)
        aug *= np.random.uniform(0.90, 1.10)

        # small random jitter
        aug += np.random.randn(*aug.shape) * 0.008

        # random mirror (50% chance)
        if np.random.random() > 0.5:
            aug[:, :, 0] *= -1

        results.append(aug.flatten())
    return results


# -----------------------------------------------------------------------------
# 3. SCAN DATASET & EXTRACT
# -----------------------------------------------------------------------------
print("=" * 60)
print("  ISL Trainer  -  Image Folder Dataset")
print("=" * 60)
print(f"\nDataset folder : {os.path.abspath(DATASET_DIR)}")

# Find all single-letter class folders (a-z)
all_dirs = os.listdir(DATASET_DIR)
classes  = sorted([
    d for d in all_dirs
    if os.path.isdir(os.path.join(DATASET_DIR, d))
    and len(d) == 1
    and d.lower() in "abcdefghijklmnopqrstuvwxyz"
])

if not classes:
    print("\nERROR: No letter folders (a-z) found inside:", DATASET_DIR)
    print("Make sure your structure is:")
    print("  ISL Project/")
    print("    dataset/")
    print("      a/  img1.jpg img2.jpg ...")
    print("      b/  img1.jpg ...")
    exit(1)

print(f"Classes found  : {[c.upper() for c in classes]}  ({len(classes)} total)")
print(f"Augmentation   : {AUGMENT_TIMES}x per image")
print()

X_list, y_list = [], []
total_skipped  = 0
start_time     = time.time()

for i, cls in enumerate(classes):
    cls_dir  = os.path.join(DATASET_DIR, cls)
    files    = [f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
    total    = len(files)
    saved    = 0
    skipped  = 0

    for j, fname in enumerate(files):
        # progress indicator every 100 images
        if (j + 1) % 100 == 0 or (j + 1) == total:
            elapsed = time.time() - start_time
            print(f"  [{cls.upper()}] {j+1}/{total} images processed"
                  f"  |  {saved} landmarks extracted"
                  f"  |  elapsed {elapsed:.0f}s",
                  end="\r")

        lm = extract_landmarks(os.path.join(cls_dir, fname))
        if lm is not None:
            X_list.append(lm)
            y_list.append(cls.upper())
            saved += 1
            for aug_lm in augment(lm, AUGMENT_TIMES):
                X_list.append(aug_lm)
                y_list.append(cls.upper())
        else:
            skipped += 1

    total_skipped += skipped
    print(f"  [{cls.upper()}] {total} images  ->  "
          f"{saved} detected  ->  "
          f"{saved * (1 + AUGMENT_TIMES)} samples after augment"
          f"  ({skipped} skipped)")

elapsed_total = time.time() - start_time
print(f"\nExtraction done in {elapsed_total/60:.1f} minutes")
print(f"Total samples  : {len(X_list)}")
print(f"Total skipped  : {total_skipped} (no hand detected)")

if len(X_list) == 0:
    print("\nERROR: No landmarks extracted at all!")
    print("Check that images actually contain hands.")
    exit(1)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)


# -----------------------------------------------------------------------------
# 4. ENCODE + SCALE
# -----------------------------------------------------------------------------
le      = LabelEncoder()
y_enc   = le.fit_transform(y)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
feature_cols = [f"f{i}" for i in range(X.shape[1])]

print(f"\nClasses : {list(le.classes_)}")
print(f"Shape   : {X_sc.shape}")


# -----------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc,
)
print(f"Train   : {len(X_train)} samples")
print(f"Test    : {len(X_test)} samples\n")


# -----------------------------------------------------------------------------
# 6. TRAIN RANDOM FOREST
# -----------------------------------------------------------------------------
print("Training Random Forest ...")
print("(this may take 2-5 minutes for large datasets)\n")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features="sqrt",
    n_jobs=-1,           # use all CPU cores
    random_state=42,
    verbose=1,
)

t0 = time.time()
rf.fit(X_train, y_train)
print(f"\nTraining finished in {(time.time()-t0)/60:.1f} minutes")


# -----------------------------------------------------------------------------
# 7. EVALUATE
# -----------------------------------------------------------------------------
y_pred = rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  ACCURACY : {acc*100:.2f}%")
print(f"{'='*60}\n")
print("Per-class report:")
print(classification_report(
    y_test, y_pred,
    target_names=[str(c) for c in le.classes_],
))


# -----------------------------------------------------------------------------
# 8. CONFUSION MATRIX
# -----------------------------------------------------------------------------
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
n       = len(le.classes_)

fig, ax = plt.subplots(figsize=(max(12, n), max(10, n - 2)))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_,
    ax=ax, annot_kws={"size": 7},
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"ISL Confusion Matrix  (Accuracy {acc*100:.1f}%)")
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved -> {cm_path}")


# -----------------------------------------------------------------------------
# 9. SAVE MODEL + ARTEFACTS
# -----------------------------------------------------------------------------
joblib.dump(rf,     os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le,     os.path.join(MODELS_DIR, "label_encoder.pkl"))
with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

print(f"\nSaved to models/:")
print(f"  rf_model.pkl")
print(f"  scaler.pkl")
print(f"  label_encoder.pkl")
print(f"  feature_cols.json")
print(f"  confusion_matrix.png")
print(f"\nDone! Accuracy = {acc*100:.2f}%")
print(f"Total time = {(time.time()-start_time)/60:.1f} minutes")
print(f"\nNext step -> run: python detect_realtime.py")
