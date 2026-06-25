import os
import io
import json
import base64
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "")

app = FastAPI(title="Smart Video Analytics API")

if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

_detector = None
_s3_client = None
_dynamo_resource = None
_trackers = {}


def get_detector(confidence=0.4):
    global _detector
    if _detector is None:
        logger.info("Cold start: loading ObjectDetector for the first time...")
        from detector import ObjectDetector
        _detector = ObjectDetector.get_instance(confidence=confidence)
        logger.info("ObjectDetector loaded successfully.")
    return _detector


def get_tracker(session_id, orientation="vertical", position_ratio=0.5):
    if session_id not in _trackers:
        from tracker import CentroidTracker, LineCrossingCounter
        _trackers[session_id] = {
            "tracker": CentroidTracker(max_distance=80, max_missing_frames=10),
            "counter": LineCrossingCounter(orientation=orientation, position_ratio=position_ratio)
        }
    return _trackers[session_id]


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3")
    return _s3_client


def get_dynamo_resource():
    global _dynamo_resource
    if _dynamo_resource is None:
        import boto3
        _dynamo_resource = boto3.resource("dynamodb")
    return _dynamo_resource


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    use_case: str = Form(default="mall"),
    confidence: float = Form(default=0.4),
    session_id: str = Form(default="default"),
    line_orientation: str = Form(default="vertical"),
    line_position: float = Form(default=0.5),
    enable_detection: str = Form(default="true"),
    enable_crowd_alert: str = Form(default=""),
    enable_line_crossing: str = Form(default="true"),
    crowd_threshold: str = Form(default=""),
):
    if use_case not in CONFIG["features"]:
        raise HTTPException(status_code=400, detail=f"Unknown use case: {use_case}")

    features = dict(CONFIG["features"][use_case])

    enable_detection_bool = enable_detection.lower() == "true"
    enable_line_crossing_bool = enable_line_crossing.lower() == "true"

    if enable_crowd_alert.lower() == "true":
        features["crowd_alert"] = True
    elif enable_crowd_alert.lower() == "false":
        features["crowd_alert"] = False

    if crowd_threshold.strip() != "":
        try:
            features["crowd_threshold"] = int(crowd_threshold)
        except ValueError:
            pass  

    if not enable_detection_bool:
        return {
            "person_count": 0,
            "tracked_count": 0,
            "line_in": 0,
            "line_out": 0,
            "detections": [],
            "annotated_image": None,
            "use_case": use_case,
            "crowd_alert": False,
            "s3_key": None,
            "timestamp": datetime.utcnow().isoformat(),
            "detection_disabled": True
        }

    import cv2
    import numpy as np

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    frame_height, frame_width = frame.shape[:2]

    detector = get_detector(confidence=confidence)
    detections = detector.detect(frame)

    person_detections = [d for d in detections if d["label"] == "person"]
    centroids = []
    for d in person_detections:
        x1, y1, x2, y2 = d["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centroids.append((cx, cy))

    if enable_line_crossing_bool:
        session = get_tracker(session_id, orientation=line_orientation, position_ratio=line_position)
        tracked_objects = session["tracker"].update(centroids)
        line_counts = session["counter"].update(tracked_objects, frame_width, frame_height)
    else:
        session = None
        tracked_objects = {}
        line_counts = {"in": 0, "out": 0}

    annotated, person_count = detector.draw_boxes(frame, detections, features)

    if enable_line_crossing_bool and session is not None:
        line_coord = session["counter"].get_line_coordinate(frame_width, frame_height)
        if line_orientation == "vertical":
            cv2.line(annotated, (line_coord, 0), (line_coord, frame_height), (255, 0, 0), 2)
        else:
            cv2.line(annotated, (0, line_coord), (frame_width, line_coord), (255, 0, 0), 2)

        cv2.putText(annotated, f"IN: {line_counts['in']}  OUT: {line_counts['out']}",
                    (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    s3_key = f"results/{use_case}/{timestamp}.jpg"

    if S3_BUCKET:
        try:
            s3 = get_s3_client()
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=buffer.tobytes(),
                ContentType="image/jpeg"
            )
        except Exception as e:
            logger.warning(f"S3 save failed: {e}")

    crowd_alert = (
        person_count >= features.get("crowd_threshold", 10)
        if features.get("crowd_alert") else False
    )

    if DYNAMO_TABLE:
        try:
            dynamo = get_dynamo_resource()
            table = dynamo.Table(DYNAMO_TABLE)
            table.put_item(Item={
                "id": timestamp,
                "use_case": use_case,
                "person_count": person_count,
                "timestamp": datetime.utcnow().isoformat(),
                "crowd_alert": crowd_alert,
                "crowd_threshold_used": features.get("crowd_threshold", 10),
                "line_in": line_counts["in"],
                "line_out": line_counts["out"]
            })
        except Exception as e:
            logger.warning(f"DynamoDB save failed: {e}")

    return {
        "person_count": person_count,
        "tracked_count": len(tracked_objects),
        "line_in": line_counts["in"],
        "line_out": line_counts["out"],
        "detections": [
            {k: list(v) if isinstance(v, tuple) else v
             for k, v in d.items()}
            for d in person_detections
        ],
        "annotated_image": img_base64,
        "use_case": use_case,
        "crowd_alert": crowd_alert,
        "crowd_threshold_used": features.get("crowd_threshold", 10),
        "s3_key": s3_key,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/reset_session")
async def reset_session(session_id: str = Form(default="default")):
    if session_id in _trackers:
        del _trackers[session_id]
    return {"status": "reset", "session_id": session_id}


handler = Mangum(app)