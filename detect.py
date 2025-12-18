import threading
import time
from typing import Any, Optional, Tuple

import cv2
import torch
from ultralytics.models import YOLO


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, model: YOLO):
        self.device = get_device()
        self.model = model

        self._frame = None
        self._results = None
        self._result_frame = None
        self._fps = 0.0
        self._new_frame_available = threading.Event()

        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._detect, daemon=True)
        self._thread.start()

    def _detect(self):
        while self._running:
            if not self._new_frame_available.wait(timeout=0.1):
                continue

            with self._lock:
                frame = self._frame
                self._new_frame_available.clear()

            if frame is None:
                continue

            start = time.perf_counter()
            results = self.model(frame, verbose=False, device=self.device)
            elapsed = time.perf_counter() - start

            with self._lock:
                self._results = results
                self._result_frame = frame
                self._fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def update_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()
        self._new_frame_available.set()

    def get_results(self) -> Tuple[Optional[Any], Optional[Any], float]:
        """Returns (results, frame_used_for_results, fps)"""
        with self._lock:
            return self._results, self._result_frame, self._fps

    def stop(self):
        self._running = False
        self._new_frame_available.set()
        self._thread.join(timeout=2.0)


def main():
    print("Loading model...")
    model = YOLO("yolov8x-oiv7.pt")
    detector = Detector(model)

    print(f"Using device: {detector.device}")

    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Opening window...")
    window_name = "Object Detection - Press 'q' to quit"

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Send frame to detector thread
        detector.update_frame(frame)

        # Get latest results
        results, result_frame, fps = detector.get_results()
        if results is None or result_frame is None:
            continue

        # Draw results on the frame that was used for detection
        annotated_frame = results[0].plot(img=result_frame)

        # Show detection FPS
        cv2.putText(
            annotated_frame,
            f"Detection FPS: {fps:.1f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main()
