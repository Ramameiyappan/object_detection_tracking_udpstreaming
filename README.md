# Real-Time Object Detection, Tracking & UDP Video Streaming

This project performs real-time object detection and tracking using YOLOv8 and ByteTrack, and streams the processed video to a separate client application using UDP.

The system includes:

- Object detection for selected target classes: Person, Bicycle, Car, Motorcycle, Bus, and Truck
- Unique ID assignment for each tracked object
- Trajectory visualization (last 30 frames per object)
- UDP-based real-time video streaming
- FPS and latency monitoring
- Custom packet structure with metadata

---

# 1️⃣ Setup Instructions

## 🔧 Dependencies Installation

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy
```

---

## 🤖 Model Download

The YOLOv8 model automatically downloads during the first execution.

Default model used:

```
yolov8n.pt
```

This lightweight model is selected for better real-time performance on standard hardware.

---

# ▶ Running the Detection Program

The detection program takes a video path from the configuration section, performs object detection, and saves the processed video as output.

Run:

```bash
python detection.py
```

Make sure the input and output paths are correctly set in the configuration section inside the script.

---

# ▶ Running the UDP Streaming Program

In the UDP streaming system:

- The server captures video
- Performs detection and tracking
- Streams the processed frames over UDP
- The client receives and displays the video in real-time

Open two terminals in the project folder.

### Step 1 – Start Client First

```bash
python stream_client.py
```

### Step 2 – Start Server

```bash
python stream_server.py
```

Press `ESC` to stop the client display window.

---

# 2️⃣ Configuration

## Detection & Tracking Configuration

```python
VIDEO_PATH = input_video_path
OUTPUT_PATH = output_video_path
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
MAX_TRAJECTORY = 30
SHOW_FPS = True
SAVE_OUTPUT = True
```

Target Class IDs:
- 0 → Person
- 1 → Bicycle
- 2 → Car
- 3 → Motorcycle
- 5 → Bus
- 7 → Truck

---

## UDP Streaming Settings

```python
VIDEO_PATH = r"D:\assignment\computer vision\checking_video.mp4"
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MAX_PACKET_SIZE = 65507
JPEG_QUALITY = 70
MODEL_PATH = "yolov8n.pt"
```

Packet Structure Used:

```
[ frame_id | timestamp | data_size | image_data ]
```

- frame_id → Used to track frame order and detect packet loss
- timestamp → Used to calculate latency
- data_size → Size of compressed image
- image_data → JPEG compressed frame bytes

---

# 3️⃣ System Design Decisions

YOLOv8 Nano (yolov8n) was selected because it provides a strong balance between detection accuracy and real-time performance. It supports multiple object classes and runs efficiently on CPU-based systems. Additionally, YOLOv8 has built-in tracking support which simplifies integration.

ByteTrack was chosen as the tracking algorithm because it maintains object IDs more reliably during short occlusions compared to traditional SORT. Unlike DeepSORT, it does not rely on heavy appearance feature extraction, making it computationally efficient while preserving tracking consistency.

UDP was selected instead of TCP because real-time streaming prioritizes low latency over guaranteed delivery. TCP retransmission can introduce delay, while UDP allows faster frame transmission. A custom packet structure was implemented to maintain frame order, measure latency, and handle packet size constraints within the 65507-byte UDP limit.
