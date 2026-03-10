"""
dataset_collector.py
--------------------
Part of the AI Sign Language Communication System.

Captures real-time hand landmark vectors via webcam and saves them as labelled
rows in a CSV dataset for training gesture-recognition models.

Location : src/data/dataset_collector.py

Keyboard controls
-----------------
  H  →  save sample labelled "hello"
  T  →  save sample labelled "thanks"
  Y  →  save sample labelled "yes"
  N  →  save sample labelled "no"
  Q  →  quit

Dataset schema
--------------
  label, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21   (64 columns total)
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — supports running directly or as part of the package
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.vision.hand_detector import HandDetector
from src.vision.landmark_extractor import LandmarkExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR  = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "sign_language_dataset.csv")

# Mapping: keyboard key (lowercase) → gesture label
KEY_LABEL_MAP: dict[str, str] = {
    "h": "hello",
    "t": "thanks",
    "y": "yes",
    "n": "no",
}

# Column names for the CSV  (label + 63 landmark coordinates)
NUM_LANDMARKS = 21
COORD_NAMES   = [
    f"{axis}{i}"
    for i in range(1, NUM_LANDMARKS + 1)
    for axis in ("x", "y", "z")
]
CSV_COLUMNS = ["label"] + COORD_NAMES


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def ensure_dataset_dir() -> None:
    """Create the dataset directory if it does not already exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)


def load_or_create_dataframe() -> pd.DataFrame:
    """
    Load the existing dataset CSV into a DataFrame, or create an empty one
    with the correct columns if the file does not yet exist.

    Returns:
        pd.DataFrame with columns [label, x1, y1, z1, ..., z21].
    """
    if os.path.isfile(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"[INFO] Loaded existing dataset — {len(df)} samples found.")
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)
        print("[INFO] No existing dataset found. Starting a fresh one.")
    return df


def save_dataframe(df: pd.DataFrame) -> None:
    """Persist the full DataFrame to the CSV file (overwrites each time)."""
    df.to_csv(DATASET_PATH, index=False)


def build_row(label: str, vector: np.ndarray) -> dict:
    """
    Combine a label string and a 63-element feature vector into a dict
    that maps directly onto the CSV column names.

    Args:
        label:  Gesture label string (e.g. "hello").
        vector: NumPy array of shape (63,) — [x1,y1,z1, ..., x21,y21,z21].

    Returns:
        dict suitable for ``pd.DataFrame.loc[len(df)] = row``.
    """
    row = {"label": label}
    row.update(dict(zip(COORD_NAMES, vector.tolist())))
    return row


# ---------------------------------------------------------------------------
# On-screen HUD rendering
# ---------------------------------------------------------------------------

def draw_hud(
    frame,
    last_label: str | None,
    total_samples: int,
    feedback_text: str | None,
    feedback_color: tuple,
) -> None:
    """
    Overlay collection statistics and user feedback onto the video frame.

    Args:
        frame:          BGR image to annotate (modified in-place).
        last_label:     Most recently saved gesture label, or None.
        total_samples:  Running count of saved samples.
        feedback_text:  Short status message to flash on screen.
        feedback_color: BGR colour tuple for the feedback message.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Total sample count
    cv2.putText(
        frame,
        f"Samples collected: {total_samples}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    # Last saved label
    label_display = last_label if last_label else "—"
    cv2.putText(
        frame,
        f"Last label: {label_display}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (100, 230, 100),
        2,
        cv2.LINE_AA,
    )

    # Transient feedback message (e.g. "Saved: hello")
    if feedback_text:
        cv2.putText(
            frame,
            feedback_text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            feedback_color,
            2,
            cv2.LINE_AA,
        )

    # Key legend in the bottom-right corner
    legend_lines = ["H=hello  T=thanks", "Y=yes  N=no  Q=quit"]
    for i, line in enumerate(reversed(legend_lines)):
        cv2.putText(
            frame,
            line,
            (w - 230, h - 15 - i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collector() -> None:
    """
    Open the webcam and enter the data-collection loop.

    Each iteration:
      1. Captures a frame.
      2. Detects the hand with HandDetector.
      3. Extracts the 63-value feature vector via LandmarkExtractor.
      4. Waits for a key press; if it matches a label key, appends a row
         to the in-memory DataFrame and flushes to disk.
      5. Overlays live HUD information and displays the frame.
    """
    ensure_dataset_dir()
    df = load_or_create_dataframe()

    detector  = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    extractor = LandmarkExtractor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        sys.exit(1)

    print("[INFO] Dataset collector running.")
    print(f"[INFO] Saving data to: {DATASET_PATH}\n")
    for key, label in KEY_LABEL_MAP.items():
        print(f"  Press '{key.upper()}' to save a '{label}' sample")
    print("  Press 'Q' to quit\n")

    last_label     = None
    feedback_text  = None
    feedback_color = (255, 255, 255)
    feedback_timer = 0          # frames remaining to display feedback

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Dropped frame — retrying.")
            continue

        # Mirror for a natural view
        frame = cv2.flip(frame, 1)

        # --- Detection & extraction ---
        results = detector.detect_hands(frame)
        vector  = extractor.extract(results)

        # Draw skeleton when a hand is present
        frame = detector.draw_landmarks(frame, results)

        # --- Keyboard input (1 ms poll) ---
        key_code = cv2.waitKey(1) & 0xFF
        pressed  = chr(key_code).lower() if key_code != 255 else None

        if pressed == "q":
            print("\n[INFO] Quit signal received — saving dataset and exiting.")
            save_dataframe(df)
            break

        if pressed in KEY_LABEL_MAP:
            gesture_label = KEY_LABEL_MAP[pressed]

            if vector is not None:
                # Append new labelled sample
                new_row = build_row(gesture_label, vector)
                df.loc[len(df)] = new_row
                save_dataframe(df)

                last_label     = gesture_label
                feedback_text  = f"✓ Saved: '{gesture_label}'  (total: {len(df)})"
                feedback_color = (0, 220, 0)
                feedback_timer = 45     # display for ~1.5 s at 30 fps

                print(f"[DATA]  Saved sample — label='{gesture_label}'  "
                      f"total={len(df)}")
            else:
                # No hand in frame — do not save
                feedback_text  = "⚠ No hand detected — sample not saved"
                feedback_color = (0, 80, 220)
                feedback_timer = 45

        # Decrement feedback display timer
        if feedback_timer > 0:
            feedback_timer -= 1
        else:
            feedback_text = None

        # --- HUD overlay ---
        draw_hud(
            frame,
            last_label=last_label,
            total_samples=len(df),
            feedback_text=feedback_text if feedback_timer > 0 else None,
            feedback_color=feedback_color,
        )

        cv2.imshow("Dataset Collector — Sign Language System", frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Dataset saved to '{DATASET_PATH}' with {len(df)} total samples.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_collector()