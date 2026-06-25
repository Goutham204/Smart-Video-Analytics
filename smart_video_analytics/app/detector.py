import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")

class ObjectDetector:
    _instance = None

    @classmethod
    def get_instance(cls, confidence=0.4):
        if cls._instance is None:
            cls._instance = cls(confidence)
        return cls._instance

    def __init__(self, confidence=0.4):
        if not os.path.exists(MODEL_PATH):
            from ultralytics import YOLO as _YOLO
            _YOLO("yolov8n.pt")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model = YOLO(MODEL_PATH)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
            device="cpu"
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2)
            })
        return detections

    def draw_boxes(self, frame, detections, features):
        h, w = frame.shape[:2]
        person_count = 0

        for d in detections:
            if d["label"] != "person":
                continue

            person_count += 1
            x1, y1, x2, y2 = d["bbox"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            label_text = f"person {d['confidence']}"
            cv2.putText(frame, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

        cv2.putText(frame, f"People: {person_count}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if features.get("crowd_alert"):
            threshold = features.get("crowd_threshold", 10)
            if person_count >= threshold:
                cv2.putText(frame, "CROWD ALERT", (12, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, person_count