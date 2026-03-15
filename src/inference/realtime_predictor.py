"""
realtime_predictor.py
---------------------
Part of the AI Sign Language Communication System.

Loads the trained Random Forest gesture model and runs real-time digit
recognition from the webcam, overlaying the predicted digit and hand
landmark skeleton on every frame.

Dependencies
------------
  models/digit_model.pkl          — trained classifier (joblib)
  src.vision.hand_detector        — MediaPipe wrapper
  src.vision.landmark_extractor   — 63-value feature vector builder

Location: src/inference/realtime_predictor.py
"""

import os
import sys

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows running directly: python src/inference/realtime_predictor.py
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.vision.hand_detector import HandDetector
from src.vision.landmark_extractor import LandmarkExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "digit_model.pkl")

# Webcam index (0 = default built-in camera)
CAMERA_INDEX = 0


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------

def draw_prediction_hud(
    frame,
    prediction: int | None,
    hand_detected: bool,
) -> None:
    """
    Overlay the prediction result and status badge onto the video frame.

    Args:
        frame:         BGR image to annotate (modified in-place).
        prediction:    Predicted digit class (0-9), or None when no hand present.
        hand_detected: Whether MediaPipe found a hand in this frame.
    """
    h, w = frame.shape[:2]

    # --- Top banner (semi-transparent dark strip) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if hand_detected and prediction is not None:
        # Large, prominent digit label
        digit_text = f"Predicted Digit: {prediction}"
        cv2.putText(
            frame,
            digit_text,
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 230, 0),        # bright green
            3,
            cv2.LINE_AA,
        )
    else:
        # Soft prompt when no hand is in view
        cv2.putText(
            frame,
            "Show a hand to the camera …",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 100, 220),    # muted blue
            2,
            cv2.LINE_AA,
        )

    # --- Bottom-right: quit hint ---
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
# Main inference loop
# ---------------------------------------------------------------------------

def run_predictor() -> None:
    """
    Open the webcam and continuously:
      1. Capture a frame.
      2. Detect the hand with HandDetector.
      3. Extract the 63-feature landmark vector via LandmarkExtractor.
      4. Classify the gesture with the loaded Random Forest model.
      5. Render the prediction and landmark skeleton on-screen.
    """

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model file not found at '{MODEL_PATH}'.\n"
              "        Run 'python src/ml/train_model.py' first.")
        sys.exit(1)

    print(f"[INFO] Loading model from : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[INFO] Model loaded successfully.\n")

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
    # Open webcam
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}). Exiting.")
        sys.exit(1)

    print("[INFO] Real-time predictor running — press 'Q' to quit.\n")

    prediction = None   # holds the latest predicted digit

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Dropped frame — retrying.")
            continue

        # Mirror for natural user experience
        frame = cv2.flip(frame, 1)

        # ------------------------------------------------------------------
        # Step 1: Detect hand landmarks
        # ------------------------------------------------------------------
        results = detector.detect_hands(frame)

        # ------------------------------------------------------------------
        # Step 2: Extract 63-value feature vector
        # ------------------------------------------------------------------
        vector = extractor.extract(results)

        # ------------------------------------------------------------------
        # Step 3: Classify gesture when a hand is present
        # ------------------------------------------------------------------
        if vector is not None:
            prediction = int(model.predict([vector])[0])

        # ------------------------------------------------------------------
        # Step 4: Draw landmark skeleton overlay
        # ------------------------------------------------------------------
        frame = detector.draw_landmarks(frame, results)

        # ------------------------------------------------------------------
        # Step 5: Render prediction HUD
        # ------------------------------------------------------------------
        hand_detected = vector is not None
        draw_prediction_hud(frame, prediction if hand_detected else None, hand_detected)

        cv2.imshow("Sign Language Digit Recogniser", frame)

        # Exit on 'q' key press
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