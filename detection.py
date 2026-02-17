import cv2
import time
from ultralytics import YOLO

VIDEO_PATH = r"D:\assignment\computer vision\checking_video.mp4"
OUTPUT_PATH = r"D:\assignment\computer vision\output_detection_video.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
SHOW_FPS = True
SAVE_OUTPUT = True

model = YOLO(MODEL_PATH)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

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
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD
    )

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    if SHOW_FPS:
        current_time = time.time()
        fps = 1 / (current_time - start_time)
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    if SAVE_OUTPUT:
        out.write(frame)

cap.release()
if SAVE_OUTPUT:
    out.release()