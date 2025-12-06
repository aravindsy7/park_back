import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import logging

logger = logging.getLogger(__name__)

class ParkingDetector:
    def __init__(self, video_path="https://10.123.150.217:8080/video"):
        self.video_path = video_path
        self.CONFIDENCE_THRESHOLD = 0.40
        self.model = YOLO("yolov8n.pt")
        
        self.OBSTACLE_CLASSES = {
            "tree", "potted plant", "plant",
            "parking meter", "stop sign", "fire hydrant",
            "bench", "traffic light", "street light", "pole"
        }
        
        self.cap = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.latest_data = {
            "total_slots": 0,
            "occupied_slots": 0,
            "free_slots": 0,
            "slots": [],
            "detections": [],
            "timestamp": time.time()
        }
        
        self.SMOOTHING = 0.25
        self.slot_y_stable = None
        self.avg_slot_w = None
        self.avg_slot_h = None
        
        logger.info(f"🎥 ParkingDetector initialized with video source: {video_path}")
        
    def iou(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0
    
    def start(self):
        """Start video capture and processing"""
        logger.info(f"🎬 Starting video capture from: {self.video_path}")
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"❌ Failed to read from video source: {self.video_path}")
                return False
            
            logger.info(f"✅ Successfully connected to video source")
            logger.info(f"📹 Video resolution: {frame.shape[1]}x{frame.shape[0]}")
            
            self.running = True
            thread = threading.Thread(target=self._process_video, daemon=True)
            thread.start()
            logger.info("✅ Video processing thread started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting video capture: {e}")
            return False
        
    def stop(self):
        """Stop video processing"""
        logger.info("🛑 Stopping video capture")
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("✅ Video capture stopped")
    
    def get_current_frame(self):
        """Get the latest frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _process_video(self):
        """Process video frames in background"""
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while self.running:
            try:
                # Skip frames for performance
                for _ in range(4):
                    if not self.running:
                        return
                    self.cap.grab()
                
                ret, frame = self.cap.read()
                if not ret:
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error(f"❌ Too many frame read errors ({error_count}). Stopping.")
                        self.running = False
                        return
                    logger.warning(f"⚠️ Failed to read frame ({error_count}/{max_errors})")
                    time.sleep(1)
                    continue
                
                error_count = 0
                frame_count += 1
                
                # Store current frame with thread safety
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                H, W = frame.shape[:2]
                small = cv2.resize(frame, (640, 360))
                sw, sh = W/640, H/360
                
                results = self.model(small, stream=False, verbose=False)
                
                car_boxes, bike_boxes, obstacle_boxes = [], [], []
                car_ws, car_hs, car_bottoms = [], [], []
                detections = []
                
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        conf = float(box.conf[0])
                        
                        if conf < self.CONFIDENCE_THRESHOLD:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1 = int(x1*sw), int(y1*sh)
                        x2, y2 = int(x2*sw), int(y2*sh)
                        w, h = x2-x1, y2-y1
                        
                        detections.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "box": [x1, y1, w, h]
                        })
                        
                        if label == "car":
                            car_boxes.append([x1,y1,w,h])
                            car_ws.append(w)
                            car_hs.append(h)
                            car_bottoms.append(y2)
                        elif label in ("motorcycle", "motorbike", "bicycle"):
                            bike_boxes.append([x1,y1,w,h])
                        elif label in self.OBSTACLE_CLASSES:
                            obstacle_boxes.append([x1,y1,w,h])
                
                # SLOT AUTO SIZE
                if car_boxes:
                    self.avg_slot_w = int(np.mean(car_ws))
                    self.avg_slot_h = int(np.mean(car_hs))
                    target_y = int(np.mean(car_bottoms)) - int(self.avg_slot_h * 0.7)
                    if self.slot_y_stable is None:
                        self.slot_y_stable = target_y
                    else:
                        self.slot_y_stable = int(self.slot_y_stable*(1-self.SMOOTHING) + target_y*self.SMOOTHING)
                else:
                    if self.avg_slot_w is None:
                        self.avg_slot_w = W//5
                        self.avg_slot_h = H//3
                    if self.slot_y_stable is None:
                        self.slot_y_stable = int(H*0.55)
                
                slot_y = self.slot_y_stable
                
                # CALCULATE SLOTS
                x = 0
                slot_num = 1
                total = 0
                occupied = 0
                slots = []
                
                while x + self.avg_slot_w <= W:
                    slot = [x, slot_y, self.avg_slot_w, self.avg_slot_h]
                    full_car = any(self.iou(slot, c) > 0.20 for c in car_boxes)
                    full_bike = any(self.iou(slot, b) > 0.01 for b in bike_boxes)
                    obs = any(self.iou(slot, o) > 0.01 for o in obstacle_boxes)
                    full = full_car or full_bike or obs
                    
                    total += 1
                    if full:
                        occupied += 1
                        status = "occupied"
                    else:
                        status = "available"
                    
                    slots.append({
                        "number": slot_num,
                        "status": status,
                        "x": x,
                        "y": slot_y,
                        "width": self.avg_slot_w,
                        "height": self.avg_slot_h
                    })
                    
                    x += self.avg_slot_w + 10
                    slot_num += 1
                
                free = total - occupied
                
                # Update latest data
                self.latest_data = {
                    "total_slots": total,
                    "occupied_slots": occupied,
                    "free_slots": free,
                    "slots": slots,
                    "detections": detections,
                    "timestamp": time.time()
                }
                
                if frame_count % 30 == 0:
                    logger.info(f"📊 Frame {frame_count}: Total={total}, Occupied={occupied}, Free={free}")
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"❌ Error processing frame: {e}")
                error_count += 1
                if error_count >= max_errors:
                    logger.error(f"Too many errors. Stopping processing.")
                    self.running = False
                    return
                time.sleep(1)
    
    def get_latest_data(self):
        """Get latest parking detection data"""
        return self.latest_data