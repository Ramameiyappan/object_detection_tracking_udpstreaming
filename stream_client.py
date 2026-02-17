import socket
import struct
import cv2
import numpy as np
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
MAX_PACKET_SIZE = 65507

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    packet, addr = sock.recvfrom(MAX_PACKET_SIZE)

    header_size = struct.calcsize("!IdI")
    header = packet[:header_size]
    frame_id, timestamp, data_size = struct.unpack("!IdI", header)

    image_data = packet[header_size:]

    np_data = np.frombuffer(image_data, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    current_time = time.time()

    latency = current_time - timestamp

    cv2.putText(frame, f"Latency: {latency:.3f}s", (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("UDP Client Stream", frame)

    if cv2.waitKey(1) == 27:
        break

sock.close()
cv2.destroyAllWindows()
