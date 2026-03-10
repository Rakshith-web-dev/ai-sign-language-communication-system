"""
hand_detector.py
----------------
Part of the AI Sign Language Communication System.
Detects and visualizes hand landmarks in real-time using MediaPipe and OpenCV.
"""

import cv2
import mediapipe as mp



class HandDetector:
    """
    Detects a single hand in a video frame using MediaPipe Hands,
    extracts 21 landmarks, and draws the hand skeleton overlay.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        """
        Initialize MediaPipe Hands model and drawing utilities.

        Args:
            max_num_hands:            Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence threshold for initial detection.
            min_tracking_confidence:  Minimum confidence threshold for landmark tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Configure the MediaPipe Hands model
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_hands(self, frame):
        """
        Convert a BGR frame to RGB and run MediaPipe hand detection.

        Args:
            frame: BGR image captured from OpenCV.

        Returns:
            results: MediaPipe Hands detection results containing hand landmarks.
        """
        # MediaPipe requires RGB input; OpenCV captures in BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark the frame as non-writeable to improve performance
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        return results

    def draw_landmarks(self, frame, results):
        """
        Draw the 21 hand landmarks and skeletal connections onto the frame.

        Args:
            frame:   BGR image to annotate.
            results: MediaPipe Hands detection results.

        Returns:
            frame: Annotated BGR image.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections (bones) using the default hand style
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style(),
                )
        return frame

    def run(self) -> None:
        """
        Start the webcam capture loop.
        Detects and draws hand landmarks on each frame until 'q' is pressed.
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Unable to access the webcam. Please check your device.")
            return

        print("[INFO] Hand detector running. Press 'q' to quit.")

        while True:
            success, frame = cap.read()

            if not success:
                print("[WARNING] Failed to read frame from webcam. Retrying...")
                continue

            # Flip the frame horizontally for a natural mirror-view
            frame = cv2.flip(frame, 1)

            # Detect hands in the current frame
            results = self.detect_hands(frame)

            # Annotate frame with landmarks and skeleton
            frame = self.draw_landmarks(frame, results)

            # Overlay a status indicator
            hand_detected = results.multi_hand_landmarks is not None
            status_text = "Hand: Detected" if hand_detected else "Hand: Not Detected"
            status_color = (0, 200, 0) if hand_detected else (0, 0, 220)
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Detector — Sign Language System", frame)

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit signal received. Stopping.")
                break

        # Release resources cleanly
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector.run()