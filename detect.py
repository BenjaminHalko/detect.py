import threading
import time

import cv2
import torch
from ultralytics.models import YOLO


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, model):
        self.model = model
        self.frame = None
        self.results = None
        self.fps = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._detect)
        self.thread.start()

    def _detect(self):
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is not None:
                start = time.time()
                results = self.model(frame, verbose=False, device=get_device())
                elapsed = time.time() - start
                with self.lock:
                    self.results = results
                    self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_results(self):
        with self.lock:
            return self.results, self.fps

    def stop(self):
        self.running = False
        self.thread.join()


def main():
    print("Loading model...")
    model = YOLO("yolov8x-oiv7.pt")
    print(f"Using device: {get_device()}")

    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Opening window...")
    window_name = "Object Detection - Press 'q' to quit"

    detector = Detector(model)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Send frame to detector thread
        detector.update_frame(frame)

        # Get latest results
        results, fps = detector.get_results()
        if results is None:
            continue

        # Draw results
        annotated_frame = results[0].plot(img=frame)

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
