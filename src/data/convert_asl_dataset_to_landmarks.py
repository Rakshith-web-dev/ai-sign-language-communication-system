"""
convert_asl_dataset_to_landmarks.py
------------------------------------
Part of the AI Sign Language Communication System.

Walks the ASL alphabet training dataset (one folder per letter A-Z),
runs MediaPipe Hands on every image via the existing HandDetector /
LandmarkExtractor modules, and writes all labelled feature vectors to a
single CSV file for ML training.

Dataset layout expected
-----------------------
dataset/asl_alphabet_train/
    A/   ←  images of the letter A
    B/   ←  images of the letter B
    …
    Z/   ←  images of the letter Z
    del/        ← ignored
    space/      ← ignored
    nothing/    ← ignored

Output
------
dataset/asl_landmarks_dataset.csv
    Columns: label, x1, y1, z1, …, x21, y21, z21   (64 columns total)

Location: src/data/convert_asl_dataset_to_landmarks.py
"""

import os
import sys
import string

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allows direct execution from the project root
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.vision.hand_detector import HandDetector
from src.vision.landmark_extractor import LandmarkExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR  = os.path.join("dataset", "asl_alphabet_train")
OUTPUT_CSV   = os.path.join("dataset", "asl_landmarks_dataset.csv")

# Folders present in the dataset that do not represent alphabet letters
IGNORED_FOLDERS = {"del", "space", "nothing"}

# Only process the 26 uppercase ASCII letters
VALID_LABELS = set(string.ascii_uppercase)

# Supported image file extensions (lowercase)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

NUM_LANDMARKS       = 21
COORDS_PER_LANDMARK = 3   # x, y, z

COORD_NAMES = [
    f"{axis}{i}"
    for i in range(1, NUM_LANDMARKS + 1)
    for axis in ("x", "y", "z")
]

CSV_COLUMNS = ["label"] + COORD_NAMES   # 64 columns total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sorted_letter_folders(dataset_dir: str) -> list[str]:
    """
    Return a sorted list of valid letter folder names (A-Z) found inside
    ``dataset_dir``, excluding any entries listed in ``IGNORED_FOLDERS``.

    Args:
        dataset_dir: Path to the root of the ASL alphabet training dataset.

    Returns:
        Sorted list of folder name strings, e.g. ['A', 'B', …, 'Z'].
    """
    entries = [
        e.name
        for e in os.scandir(dataset_dir)
        if e.is_dir()
        and e.name.upper() in VALID_LABELS
        and e.name.lower() not in IGNORED_FOLDERS
    ]
    return sorted(entries)


def collect_image_paths(folder_path: str) -> list[str]:
    """
    Return a sorted list of full paths to supported image files inside
    ``folder_path``.

    Args:
        folder_path: Absolute or relative path to a single letter folder.

    Returns:
        Sorted list of image file path strings.
    """
    return sorted([
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ])


def build_csv_row(label: str, vector: np.ndarray) -> dict:
    """
    Combine a letter label and a 63-element landmark vector into a dict
    aligned with ``CSV_COLUMNS``.

    Args:
        label:  Single uppercase letter string (e.g. "A").
        vector: Float32 NumPy array of shape (63,).

    Returns:
        dict suitable for appending to the rows list.
    """
    row = {"label": label}
    row.update(dict(zip(COORD_NAMES, vector.tolist())))
    return row


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert_dataset() -> None:
    """
    Full conversion pipeline:
      1. Discover valid letter folders.
      2. For each image: load → detect → extract → store.
      3. Write accumulated rows to CSV.
      4. Print a final statistics summary.
    """

    # ------------------------------------------------------------------
    # Validate dataset directory
    # ------------------------------------------------------------------
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: '{DATASET_DIR}'\n"
              "        Please check the path and try again.")
        sys.exit(1)

    letter_folders = get_sorted_letter_folders(DATASET_DIR)

    if not letter_folders:
        print(f"[ERROR] No valid letter folders (A–Z) found in '{DATASET_DIR}'.")
        sys.exit(1)

    print(f"[INFO] Dataset directory : {DATASET_DIR}")
    print(f"[INFO] Letters found     : {', '.join(letter_folders)}")
    print(f"[INFO] Output CSV        : {OUTPUT_CSV}\n")

    # ------------------------------------------------------------------
    # Initialise vision modules
    # ------------------------------------------------------------------
    detector  = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    extractor = LandmarkExtractor()

    # ------------------------------------------------------------------
    # Iterate over letter folders
    # ------------------------------------------------------------------
    all_rows       = []
    total_processed = 0   # images attempted
    total_saved     = 0   # landmarks successfully extracted
    total_skipped   = 0   # images where no hand was detected

    for letter in letter_folders:
        folder_path  = os.path.join(DATASET_DIR, letter)
        image_paths  = collect_image_paths(folder_path)

        if not image_paths:
            print(f"  [WARNING] No images found in folder '{letter}'. Skipping.")
            continue

        folder_saved   = 0
        folder_skipped = 0

        # tqdm progress bar scoped to this letter folder
        for image_path in tqdm(
            image_paths,
            desc=f"  Processing '{letter}'",
            unit="img",
            leave=True,
        ):
            total_processed += 1

            # --- Load image ---
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"\n  [WARNING] Could not read '{image_path}'. Skipping.")
                folder_skipped += 1
                total_skipped  += 1
                continue

            # --- Detect hand ---
            results = detector.detect_hands(frame)

            # --- Extract 63-value landmark vector ---
            vector = extractor.extract(results)

            if vector is None:
                # No hand detected — skip this image
                folder_skipped += 1
                total_skipped  += 1
                continue

            # --- Store labelled row ---
            all_rows.append(build_csv_row(letter, vector))
            folder_saved += 1
            total_saved  += 1

        print(f"    → Saved: {folder_saved:,}  |  Skipped: {folder_skipped:,}")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    if not all_rows:
        print("\n[ERROR] No landmarks were extracted. CSV will not be created.")
        return

    print(f"\n[INFO] Writing {total_saved:,} rows to '{OUTPUT_CSV}' …")
    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Conversion complete")
    print("=" * 55)
    print(f"  Total images processed  : {total_processed:,}")
    print(f"  Landmarks extracted     : {total_saved:,}")
    print(f"  Images skipped          : {total_skipped:,}  (no hand detected)")
    print(f"  Detection rate          : "
          f"{total_saved / max(total_processed, 1) * 100:.1f}%")
    print(f"  Output CSV              : {OUTPUT_CSV}")
    print(f"  CSV shape               : "
          f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset()