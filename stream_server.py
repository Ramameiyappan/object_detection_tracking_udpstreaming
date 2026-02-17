import cv2
import socket
import struct
import time
from ultralytics import YOLO

VIDEO_PATH = r"D:\assignment\computer vision\checking_video.mp4"
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MAX_PACKET_SIZE = 65507
JPEG_QUALITY = 70
MODEL_PATH = "yolov8n.pt"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    object_count = 0

    for result in results:
        boxes = result.boxes
        object_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]
            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    data = buffer.tobytes()
    data_size = len(data)

    if data_size > MAX_PACKET_SIZE - 16:
        print("Frame too large, reduce resolution or quality")
        continue

    timestamp = time.time()

    header = struct.pack("!IdI", frame_id, timestamp, data_size)
    packet = header + data

    sock.sendto(packet, (UDP_IP, UDP_PORT))

    frame_id += 1
