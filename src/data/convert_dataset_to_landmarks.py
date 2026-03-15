"""
convert_dataset_to_landmarks.py
--------------------------------
Part of the AI Sign Language Communication System.

Loads the Sign Language Digits dataset stored as NumPy arrays (X.npy / Y.npy),
runs MediaPipe Hands on every image, and writes the resulting 63-value landmark
feature vectors — together with their integer labels — to a CSV file ready for
ML training.

Dataset expected layout
-----------------------
dataset/digits-dataset/
    X.npy   shape: (2062, 64, 64)   — grayscale images
    Y.npy   shape: (2062, 10)       — one-hot encoded labels (digits 0-9)

Output
------
dataset/digits_landmarks_dataset.csv
    Columns: label, x1, y1, z1, ..., x21, y21, z21   (64 columns total)

Location: src/data/convert_dataset_to_landmarks.py
"""

import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR  = "dataset"
X_NPY_PATH   = os.path.join(DATASET_DIR, "digits-dataset", "X.npy")
Y_NPY_PATH   = os.path.join(DATASET_DIR, "digits-dataset", "Y.npy")
OUTPUT_CSV   = os.path.join(DATASET_DIR, "digits_landmarks_dataset.csv")

# Images are upscaled before detection — larger images improve MediaPipe accuracy
RESIZE_DIM = 256   # target width and height in pixels

# MediaPipe confidence thresholds (relaxed slightly for low-res/stylised images)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------

NUM_LANDMARKS       = 21
COORDS_PER_LANDMARK = 3  # x, y, z

COORD_NAMES = [
    f"{axis}{i}"
    for i in range(1, NUM_LANDMARKS + 1)
    for axis in ("x", "y", "z")
]

CSV_COLUMNS = ["label"] + COORD_NAMES   # 64 columns in total


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def grayscale_to_bgr(image_gray: np.ndarray, target_size: int) -> np.ndarray:
    """
    Convert a single-channel grayscale image to a 3-channel BGR image and
    resize it so MediaPipe has enough detail for reliable detection.

    Args:
        image_gray:  2-D NumPy array of shape (H, W) with dtype uint8 or float.
        target_size: Output width and height in pixels (square resize).

    Returns:
        3-channel BGR uint8 array of shape (target_size, target_size, 3).
    """
    # Normalise to uint8 if the array contains float values in [0, 1]
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).clip(0, 255).astype(np.uint8)

    # Upscale to give MediaPipe more pixel detail
    image_resized = cv2.resize(
        image_gray,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # MediaPipe expects a 3-channel image; replicate the single channel
    image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)

    return image_bgr


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_landmark_vector(
    image_bgr: np.ndarray,
    hands_model,
) -> np.ndarray | None:
    """
    Run MediaPipe Hands on a BGR image and return a flat 63-value feature vector.

    Args:
        image_bgr:   3-channel BGR uint8 image.
        hands_model: Initialised ``mediapipe.solutions.hands.Hands`` instance.

    Returns:
        NumPy float32 array of shape (63,) — [x1,y1,z1, ..., x21,y21,z21] —
        or ``None`` when no hand is detected.
    """
    # Convert BGR → RGB as required by MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Skip internal writeable check for a small speed gain
    image_rgb.flags.writeable = False
    results = hands_model.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None  # No hand found — caller will skip this sample

    # Take only the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    vector = np.zeros(NUM_LANDMARKS * COORDS_PER_LANDMARK, dtype=np.float32)
    for idx, lm in enumerate(hand_landmarks.landmark):
        offset = idx * COORDS_PER_LANDMARK
        vector[offset]     = lm.x
        vector[offset + 1] = lm.y
        vector[offset + 2] = lm.z

    return vector


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert_dataset() -> None:
    """
    Load X.npy / Y.npy, extract MediaPipe landmarks for every image, and
    save the labelled feature vectors to a CSV file.

    Images where MediaPipe fails to detect a hand are skipped silently; the
    final summary reports exactly how many were kept vs. dropped.
    """

    # ------------------------------------------------------------------
    # 1. Load raw arrays
    # ------------------------------------------------------------------
    print(f"[INFO] Loading X from : {X_NPY_PATH}")
    print(f"[INFO] Loading Y from : {Y_NPY_PATH}\n")

    X = np.load(X_NPY_PATH)   # (2062, 64, 64)
    Y = np.load(Y_NPY_PATH)   # (2062, 10)

    total_images = X.shape[0]
    print(f"[INFO] Dataset loaded  — {total_images:,} images, "
          f"image shape: {X.shape[1]}x{X.shape[2]} px\n")

    # ------------------------------------------------------------------
    # 2. Decode one-hot labels to integer class indices
    # ------------------------------------------------------------------
    labels_int = np.argmax(Y, axis=1)   # shape: (2062,)

    # ------------------------------------------------------------------
    # 3. Initialise MediaPipe Hands in static-image mode
    # ------------------------------------------------------------------
    mp_hands    = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=True,           # each image is independent
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # ------------------------------------------------------------------
    # 4. Iterate — extract landmarks and accumulate rows
    # ------------------------------------------------------------------
    rows          = []   # list of dicts — converted to DataFrame at the end
    total_saved   = 0
    total_skipped = 0

    for i in tqdm(range(total_images), desc="Extracting landmarks", unit="img"):
        label = int(labels_int[i])

        # Pre-process: grayscale (64x64) -> BGR (256x256)
        image_bgr = grayscale_to_bgr(X[i], target_size=RESIZE_DIM)

        # Run MediaPipe hand detection
        vector = extract_landmark_vector(image_bgr, hands_model)

        if vector is None:
            total_skipped += 1
            continue

        # Build a column-aligned dict for this sample
        row = {"label": label}
        row.update(dict(zip(COORD_NAMES, vector.tolist())))
        rows.append(row)
        total_saved += 1

    # Release MediaPipe resources
    hands_model.close()

    # ------------------------------------------------------------------
    # 5. Persist to CSV
    # ------------------------------------------------------------------
    if not rows:
        print(
            "\n[ERROR] No hands were detected across the entire dataset.\n"
            "The CSV will not be created.\n"
            "Try lowering MIN_DETECTION_CONFIDENCE or verify the image paths."
        )
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Conversion complete")
    print("=" * 55)
    print(f"  Total samples processed : {total_images:,}")
    print(f"  Total samples saved     : {total_saved:,}")
    print(f"  Total samples skipped   : {total_skipped:,}  (no hand detected)")
    print(f"  Detection rate          : "
          f"{total_saved / total_images * 100:.1f}%")
    print(f"  Output CSV              : {OUTPUT_CSV}")
    print(f"  CSV shape               : "
          f"{df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset()