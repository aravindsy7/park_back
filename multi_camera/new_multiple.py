import cv2
import numpy as np
from ultralytics import YOLO
import threading

# ================= CONFIG =================
CONFIDENCE_THRESHOLD = 0.50

# Load YOLO only once (shared among threads)
model = YOLO("yolov8n.pt")

OBSTACLE_CLASSES = {
    "tree", "potted plant", "plant",
    "parking meter", "stop sign", "fire hydrant",
    "bench", "traffic light", "street light", "pole"
}

# ==================== PARKING PROCESSOR ====================

def process_camera(cam_id, video_path):
    print(f"[INFO] Starting Camera {cam_id} : {video_path}")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # independent stabilisation variables for each camera
    slot_y_stable = None
    avg_slot_w = None
    avg_slot_h = None
    SMOOTHING = 0.25

    while True:
        # discard buffered frames
        for _ in range(4):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Camera {cam_id} disconnected!")
            break

        H, W = frame.shape[:2]
        small = cv2.resize(frame, (640, 360))
        sw, sh = W/640, H/360

        # YOLO inference
        results = model(small, stream=False)

        # ------------------ CLASSIFY BOXES ------------------
        car_boxes, bike_boxes, obstacle_boxes = [], [], []
        car_ws, car_hs, car_bottoms = [], [], []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1 = int(x1 * sw), int(y1 * sh)
                x2, y2 = int(x2 * sw), int(y2 * sh)
                w, h = x2 - x1, y2 - y1

                if label == "car":
                    car_boxes.append([x1, y1, w, h])
                    car_ws.append(w)
                    car_hs.append(h)
                    car_bottoms.append(y2)

                elif label in ("motorcycle", "motorbike", "bicycle"):
                    bike_boxes.append([x1, y1, w, h])

                elif label in OBSTACLE_CLASSES:
                    obstacle_boxes.append([x1, y1, w, h])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ------------------ SLOT AUTO-SIZE ------------------
        if car_boxes:
            avg_slot_w = int(np.mean(car_ws))
            avg_slot_h = int(np.mean(car_hs))
            target_y = int(np.mean(car_bottoms)) - int(avg_slot_h * 0.7)

            if slot_y_stable is None:
                slot_y_stable = target_y
            else:
                slot_y_stable = int(slot_y_stable * (1 - SMOOTHING) +
                                    target_y * SMOOTHING)
        else:
            if avg_slot_w is None:
                avg_slot_w = W // 5
                avg_slot_h = H // 3
            if slot_y_stable is None:
                slot_y_stable = int(H * 0.55)

        slot_y = slot_y_stable

        # ------------------ DRAW SLOTS ------------------
        def iou(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = w1*h1 + w2*h2 - inter
            return inter / union if union > 0 else 0

        occupied_slots = []
        available_slots = []

        x = 0
        slot_num = 1
        total = 0
        occupied = 0

        while x + avg_slot_w <= W:
            slot = [x, slot_y, avg_slot_w, avg_slot_h]

            full_car = any(iou(slot, c) > 0.20 for c in car_boxes)
            full_bike = any(iou(slot, b) > 0.01 for b in bike_boxes)
            obs = any(iou(slot, o) > 0.01 for o in obstacle_boxes)

            full = full_car or full_bike or obs
            total += 1

            if full:
                color = (0, 0, 255)
                text = "Full"
                occupied += 1
                occupied_slots.append(slot_num)
            else:
                color = (0, 255, 0)
                text = "Available"
                available_slots.append(slot_num)

            cv2.rectangle(frame, (x, slot_y),
                          (x + avg_slot_w, slot_y + avg_slot_h), color, 2)
            cv2.putText(frame, f"Slot {slot_num}: {text}",
                        (x + 5, slot_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            x += avg_slot_w + 10
            slot_num += 1

        free = total - occupied

        # ------------------ DISPLAY INFO ------------------
        cv2.putText(frame, f"Camera {cam_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Occupied Slots: {occupied_slots}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Available Slots: {available_slots}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Total: {total}  Occupied: {occupied}  Free: {free}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"Camera {cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {cam_id}")

# ==================== MAIN (4 CAMERAS) ====================

camera_streams = [
    ("Cam1", "http://192.0.0.4:8080/video"),
    ("Cam2", "http://192.0.0.4:8080/video")
    ("Cam3", "http://192.0.0.4:8080/video")
    ("Cam4", "http://192.0.0.4:8080/video")
  
]

threads = []

for cam_id, url in camera_streams:
    t = threading.Thread(target=process_camera, args=(cam_id, url))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
