"""
realtime_asl_predictor.py
--------------------------
Part of the AI Sign Language Communication System.

Loads the trained ASL alphabet Random Forest model and label encoder,
then runs real-time letter recognition from the webcam. A rolling prediction
buffer smooths out frame-to-frame noise before displaying the result.

Dependencies
------------
  models/asl_alphabet_model.pkl     — trained RandomForest classifier
  models/asl_label_encoder.pkl      — fitted LabelEncoder (int ↔ letter)
  src.vision.hand_detector          — MediaPipe hand-detection wrapper
  src.vision.landmark_extractor     — 63-value feature vector builder

Location: src/inference/realtime_asl_predictor.py
"""

import os
import sys
from collections import deque

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — supports: python src/inference/realtime_asl_predictor.py
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.vision.hand_detector import HandDetector
from src.vision.landmark_extractor import LandmarkExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH   = os.path.join("models", "asl_alphabet_model.pkl")
ENCODER_PATH = os.path.join("models", "asl_label_encoder.pkl")

CAMERA_INDEX    = 0
BUFFER_SIZE     = 10   # number of recent predictions used for smoothing


# ---------------------------------------------------------------------------
# Prediction smoother
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """
    Maintains a rolling buffer of recent predictions and returns the majority vote.
    This helps to stabilize the output against momentary misclassifications.

    """

    def __init__(self, buffer_size: int = BUFFER_SIZE) -> None:
        self._buffer: deque[str] = deque(maxlen=buffer_size)

    def update(self, label: str) -> str:
        """
        Add a new prediction to the buffer and return the current majority vote.
        Args:
            label: Latest predicted letter (e.g., 'A', 'B', ..., 'Z').
        """
        self._buffer.append(label)
        return max(set(self._buffer), key=self._buffer.count)

    def reset(self) -> None:
        """Clear the buffer (call when the hand leaves the frame)."""
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        """True once the buffer holds at least one prediction."""
        return len(self._buffer) > 0


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------

def draw_hud(
    frame,
    smoothed_label: str | None,
    hand_detected: bool,
    buffer_fill: int,
) -> None:
    """
    Overlay the prediction result, confidence fill-bar, and status info
    onto the video frame.

    Args:
        frame:          BGR image to annotate (modified in-place).
        smoothed_label: Smoothed prediction string, or None.
        hand_detected:  Whether a hand is currently detected.
        buffer_fill:    Number of predictions currently in the buffer (0-10).
    """
    h, w = frame.shape[:2]

    # --- Semi-transparent top banner ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if hand_detected and smoothed_label:
        # Large letter prediction
        cv2.putText(
            frame,
            f"Prediction: {smoothed_label}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 230, 0),
            3,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Show a hand to the camera ...",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (100, 100, 220),
            2,
            cv2.LINE_AA,
        )

    # --- Buffer fill bar (shows smoothing confidence) ---
    bar_x, bar_y, bar_w, bar_h = 15, h - 40, 200, 14
    filled_w = int(bar_w * (buffer_fill / BUFFER_SIZE))

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    if filled_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 200, 120), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    cv2.putText(
        frame,
        f"Buffer: {buffer_fill}/{BUFFER_SIZE}",
        (bar_x + bar_w + 10, bar_y + bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # --- Bottom-right quit hint ---
    cv2.putText(
        frame,
        "Press 'Q' to quit",
        (w - 210, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_encoder():
    """
    Load the trained RandomForest model and LabelEncoder from disk.

    Returns:
        model:   Fitted sklearn RandomForestClassifier.
        encoder: Fitted sklearn LabelEncoder.

    Raises:
        SystemExit if either file is missing.
    """
    for path, name in [(MODEL_PATH, "model"), (ENCODER_PATH, "label encoder")]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name.capitalize()} file not found: '{path}'\n"
                  "        Run 'python src/ml/train_asl_model.py' first.")
            sys.exit(1)

    print(f"[INFO] Loading model        : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"[INFO] Loading label encoder: {ENCODER_PATH}")
    encoder = joblib.load(ENCODER_PATH)

    print(f"[INFO] Classes recognised   : {list(encoder.classes_)}\n")
    return model, encoder


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_predictor() -> None:
    """
    Main loop: capture webcam frames, run ASL prediction, and display results.
    Loads the trained model and label encoder, then continuously processes
    video frames until the user quits.
     - Detects hand landmarks using MediaPipe.
     - Extracts a 63-value feature vector from the landmarks.
     - Predicts the ASL letter using the Random Forest model.
     - Smooths predictions with a rolling majority vote buffer.
     - Overlays the predicted letter and status info onto the video feed.
    """

    model, encoder = load_model_and_encoder()

    detector  = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    extractor = LandmarkExtractor()
    smoother  = PredictionSmoother(buffer_size=BUFFER_SIZE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}). Exiting.")
        sys.exit(1)

    print("[INFO] ASL real-time predictor running — press 'Q' to quit.\n")

    smoothed_label = None

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Dropped frame — retrying.")
            continue

        # Mirror for a natural user experience
        frame = cv2.flip(frame, 1)

        # --- Step 1: Detect hand ---
        results = detector.detect_hands(frame)

        # --- Step 2: Extract feature vector ---
        vector = extractor.extract(results)

        hand_detected = vector is not None

        if hand_detected:
            # --- Step 3: Classify ---
            raw_int   = model.predict([vector])[0]
            raw_label = encoder.inverse_transform([raw_int])[0]

            # --- Step 4: Smooth via rolling majority vote ---
            smoothed_label = smoother.update(raw_label)
        else:
            # Hand left the frame — reset buffer so stale votes don't linger
            smoother.reset()
            smoothed_label = None

        # --- Step 5: Draw skeleton overlay ---
        frame = detector.draw_landmarks(frame, results)

        # --- Step 6: Render HUD ---
        draw_hud(
            frame,
            smoothed_label=smoothed_label,
            hand_detected=hand_detected,
            buffer_fill=len(smoother._buffer),
        )

        cv2.imshow("ASL Alphabet Recogniser — Sign Language System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received — shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_predictor()