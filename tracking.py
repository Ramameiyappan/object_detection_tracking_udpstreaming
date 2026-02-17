import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

VIDEO_PATH = r"D:\assignment\computer vision\checking_video.mp4"
OUTPUT_PATH = r"D:\assignment\computer vision\output_detection_video.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
MAX_TRAJECTORY = 30
SAVE_OUTPUT = True

model = YOLO(MODEL_PATH)
track_history = defaultdict(lambda: deque(maxlen=MAX_TRAJECTORY))

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video file")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if SAVE_OUTPUT:
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=TARGET_CLASSES,
        tracker="bytetrack.yaml"
    )

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, class_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls = int(cls)

            label = model.names[cls]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)

            text = f"ID {track_id} | {label} {conf:.2f}"
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track_history[track_id].append((cx, cy))

            points = track_history[track_id]
            for i in range(1, len(points)):
                cv2.line(
                    annotated_frame,
                    points[i-1],
                    points[i],
                    (0,0,255),
                    2
                )
    if SAVE_OUTPUT:
        out.write(annotated_frame)

cap.release()
if SAVE_OUTPUT:
    out.release()