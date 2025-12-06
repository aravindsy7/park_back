import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultiCameraDetector:
    """
    Manages multiple camera feeds with synchronized parking slot detection.
    Supports 2, 3, or 4 cameras covering ~10-12 total parking slots.
    """
    
    def __init__(self, camera_urls, confidence_threshold=0.40, frame_resize=(640, 360)):
        """
        Args:
            camera_urls: list of camera URLs
            confidence_threshold: YOLO detection confidence
            frame_resize: (width, height) for inference
        """
        self.camera_urls = camera_urls
        self.confidence_threshold = confidence_threshold
        self.frame_resize = frame_resize
        
        # Load YOLO model once (shared across all cameras)
        logger.info("🤖 Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        logger.info("✅ YOLOv8 model loaded")
        
        # Camera state
        self.cameras = {}  # idx -> {cap, url, frame, data}
        self.running = False
        self.lock = threading.Lock()
        
        # Obstacle classes
        self.OBSTACLE_CLASSES = {
            "tree", "potted plant", "plant",
            "parking meter", "stop sign", "fire hydrant",
            "bench", "traffic light", "street light", "pole"
        }
        
        # Parking slot tracking
        self.slot_configs = {}  # idx -> {y_stable, avg_w, avg_h}
        self.SMOOTHING = 0.25
        
        # Combined data
        self.all_slots_data = {
            "total_slots": 0,
            "occupied_slots": 0,
            "free_slots": 0,
            "slots": [],
            "detections": [],
            "cameras_active": 0,
            "timestamp": time.time()
        }
        
        logger.info(f"🎥 Multi-Camera Detector initialized with {len(camera_urls)} cameras")
        
    def _open_cameras(self):
        """Open all camera connections"""
        for idx, url in enumerate(self.camera_urls):
            try:
                cap = cv2.VideoCapture(url)
                time.sleep(0.5)  # Wait for connection
                
                if cap.isOpened():
                    logger.info(f"✅ Camera {idx+1} opened: {url}")
                    self.cameras[idx] = {
                        "cap": cap,
                        "url": url,
                        "frame": None,
                        "data": None,
                        "connected": True,
                        "error_count": 0
                    }
                else:
                    logger.error(f"❌ Camera {idx+1} failed to open: {url}")
                    cap.release()
            except Exception as e:
                logger.error(f"❌ Error opening camera {idx+1}: {e}")
        
        if len(self.cameras) == 0:
            logger.error("❌ No cameras could be opened!")
            return False
        
        logger.info(f"✅ Opened {len(self.cameras)}/{len(self.camera_urls)} cameras")
        return True
    
    def _close_cameras(self):
        """Close all camera connections"""
        for idx, cam_info in self.cameras.items():
            try:
                cam_info["cap"].release()
                logger.info(f"✅ Camera {idx+1} released")
            except Exception as e:
                logger.warning(f"⚠️ Error releasing camera {idx+1}: {e}")
    
    def start(self):
        """Start multi-camera processing"""
        logger.info("🚀 Starting Multi-Camera Detector...")
        
        if not self._open_cameras():
            return False
        
        self.running = True
        
        # Start processing thread for each camera
        for idx in self.cameras.keys():
            thread = threading.Thread(
                target=self._process_camera,
                args=(idx,),
                daemon=True,
                name=f"CameraProcessor-{idx+1}"
            )
            thread.start()
        
        # Start aggregation thread
        agg_thread = threading.Thread(
            target=self._aggregate_data,
            daemon=True,
            name="DataAggregator"
        )
        agg_thread.start()
        
        logger.info("✅ Multi-Camera Detector started")
        return True
    
    def stop(self):
        """Stop all processing"""
        logger.info("🛑 Stopping Multi-Camera Detector...")
        self.running = False
        time.sleep(1)
        self._close_cameras()
        logger.info("✅ Multi-Camera Detector stopped")
    
    def _process_camera(self, camera_idx):
        """Process frames from a single camera (runs in separate thread)"""
        cam_info = self.cameras[camera_idx]
        cap = cam_info["cap"]
        frame_count = 0
        max_errors = 10
        
        logger.info(f"📹 Camera {camera_idx+1} processing thread started")
        
        while self.running:
            try:
                # Skip frames for performance
                for _ in range(3):
                    if not self.running:
                        return
                    cap.grab()
                
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    cam_info["error_count"] += 1
                    if cam_info["error_count"] >= max_errors:
                        logger.error(f"❌ Camera {camera_idx+1} lost connection")
                        cam_info["connected"] = False
                        return
                    logger.warning(f"⚠️ Camera {camera_idx+1} frame read failed ({cam_info['error_count']}/{max_errors})")
                    time.sleep(0.5)
                    continue
                
                cam_info["error_count"] = 0
                frame_count += 1
                
                # Process frame
                H, W = frame.shape[:2]
                small = cv2.resize(frame, self.frame_resize)
                sw, sh = W / self.frame_resize[0], H / self.frame_resize[1]
                
                # Run YOLO
                results = self.model(small, stream=False, verbose=False)
                
                # Extract detections
                car_boxes, bike_boxes, obstacle_boxes = [], [], []
                car_ws, car_hs, car_bottoms = [], [], []
                detections = []
                
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = self.model.names[cls]
                        conf = float(box.conf[0])
                        
                        if conf < self.confidence_threshold:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1 = int(x1 * sw), int(y1 * sh)
                        x2, y2 = int(x2 * sw), int(y2 * sh)
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "box": [x1, y1, w, h],
                            "camera": camera_idx + 1
                        })
                        
                        if label == "car":
                            car_boxes.append([x1, y1, w, h])
                            car_ws.append(w)
                            car_hs.append(h)
                            car_bottoms.append(y2)
                        elif label in ("motorcycle", "motorbike", "bicycle"):
                            bike_boxes.append([x1, y1, w, h])
                        elif label in self.OBSTACLE_CLASSES:
                            obstacle_boxes.append([x1, y1, w, h])
                
                # Auto-size parking slots
                if camera_idx not in self.slot_configs:
                    self.slot_configs[camera_idx] = {
                        "y_stable": None,
                        "avg_w": None,
                        "avg_h": None
                    }
                
                config = self.slot_configs[camera_idx]
                
                if car_boxes:
                    config["avg_w"] = int(np.mean(car_ws))
                    config["avg_h"] = int(np.mean(car_hs))
                    target_y = int(np.mean(car_bottoms)) - int(config["avg_h"] * 0.7)
                    if config["y_stable"] is None:
                        config["y_stable"] = target_y
                    else:
                        config["y_stable"] = int(
                            config["y_stable"] * (1 - self.SMOOTHING) + target_y * self.SMOOTHING
                        )
                else:
                    if config["avg_w"] is None:
                        config["avg_w"] = W // 5
                        config["avg_h"] = H // 3
                    if config["y_stable"] is None:
                        config["y_stable"] = int(H * 0.55)
                
                # Calculate parking slots for this camera
                slot_y = config["y_stable"]
                x = 0
                slot_num = 1
                total = 0
                occupied = 0
                slots = []
                
                while x + config["avg_w"] <= W:
                    slot = [x, slot_y, config["avg_w"], config["avg_h"]]
                    full_car = any(self._iou(slot, c) > 0.20 for c in car_boxes)
                    full_bike = any(self._iou(slot, b) > 0.01 for b in bike_boxes)
                    obs = any(self._iou(slot, o) > 0.01 for o in obstacle_boxes)
                    full = full_car or full_bike or obs
                    
                    total += 1
                    if full:
                        occupied += 1
                        status = "occupied"
                    else:
                        status = "available"
                    
                    slots.append({
                        "number": f"C{camera_idx+1}-S{slot_num}",
                        "camera": camera_idx + 1,
                        "status": status,
                        "x": x,
                        "y": slot_y,
                        "width": config["avg_w"],
                        "height": config["avg_h"]
                    })
                    
                    x += config["avg_w"] + 10
                    slot_num += 1
                
                # Store camera data
                with self.lock:
                    cam_info["frame"] = frame.copy()
                    cam_info["data"] = {
                        "camera": camera_idx + 1,
                        "total_slots": total,
                        "occupied_slots": occupied,
                        "free_slots": total - occupied,
                        "slots": slots,
                        "detections": detections,
                        "timestamp": time.time()
                    }
                
                if frame_count % 30 == 0:
                    logger.info(f"📊 Camera {camera_idx+1}: Total={total}, Occupied={occupied}, Free={total-occupied}")
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.exception(f"❌ Error processing camera {camera_idx+1}: {e}")
                cam_info["error_count"] += 1
                time.sleep(1)
    
    def _aggregate_data(self):
        """Aggregate data from all cameras every 1 second"""
        logger.info("📊 Data aggregation thread started")
        
        while self.running:
            try:
                with self.lock:
                    all_slots = []
                    all_detections = []
                    total_slots = 0
                    occupied_slots = 0
                    cameras_active = 0
                    
                    for idx, cam_info in self.cameras.items():
                        if cam_info["data"] is None:
                            continue
                        
                        cameras_active += 1
                        data = cam_info["data"]
                        
                        total_slots += data["total_slots"]
                        occupied_slots += data["occupied_slots"]
                        all_slots.extend(data["slots"])
                        all_detections.extend(data["detections"])
                    
                    free_slots = total_slots - occupied_slots
                    
                    self.all_slots_data = {
                        "total_slots": total_slots,
                        "occupied_slots": occupied_slots,
                        "free_slots": free_slots,
                        "slots": all_slots,
                        "detections": all_detections,
                        "cameras_active": cameras_active,
                        "timestamp": time.time()
                    }
                
                logger.info(f"📊 Aggregated: Total={total_slots}, Occupied={occupied_slots}, Free={free_slots}, Active Cameras={cameras_active}")
                time.sleep(1)
                
            except Exception as e:
                logger.exception(f"❌ Error in data aggregation: {e}")
                time.sleep(1)
    
    def _iou(self, b1, b2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0
    
    def get_current_frame(self, camera_idx):
        """Get current frame from specific camera"""
        with self.lock:
            if camera_idx in self.cameras:
                return self.cameras[camera_idx]["frame"].copy() if self.cameras[camera_idx]["frame"] is not None else None
        return None
    
    def get_all_frames(self):
        """Get frames from all cameras as grid"""
        with self.lock:
            frames = []
            for idx in range(len(self.camera_urls)):
                if idx in self.cameras and self.cameras[idx]["frame"] is not None:
                    frame = self.cameras[idx]["frame"].copy()
                    frames.append((idx, frame))
                else:
                    h, w = self.frame_resize[1], self.frame_resize[0]
                    blank = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Cam {idx+1} offline", (20, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    frames.append((idx, blank))
            return frames
    
    def get_latest_data(self):
        """Get aggregated parking data from all cameras"""
        with self.lock:
            return self.all_slots_data.copy()
    
    def grid_display(self, annotate=True):
        """Create grid display of all cameras"""
        frames = self.get_all_frames()
        
        if annotate:
            # Annotate each frame with detections and slots
            annotated = []
            for cam_idx, frame in frames:
                if cam_idx in self.cameras and self.cameras[cam_idx]["data"]:
                    data = self.cameras[cam_idx]["data"]
                    
                    # Draw detections
                    for det in data["detections"]:
                        box = det.get("box")
                        if box and len(box) == 4:
                            x, y, w, h = box
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_text = f"{det.get('label')} {int(det.get('confidence', 0)*100)}%"
                            cv2.putText(frame, label_text, (x1, y1-6),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw slots
                    for slot in data["slots"]:
                        x, y, w, h = slot["x"], slot["y"], slot["width"], slot["height"]
                        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                        color = (0, 255, 0) if slot["status"] == "available" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Camera label and stats
                    cv2.putText(frame, f"Cam {cam_idx+1}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Total:{data['total_slots']} Occ:{data['occupied_slots']} Free:{data['free_slots']}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                annotated.append((cam_idx, frame))
        else:
            annotated = frames
        
        # Create grid
        n = len(annotated)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        tile_h, tile_w = self.frame_resize[1], self.frame_resize[0]
        canvas_h = rows * tile_h
        canvas_w = cols * tile_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        i = 0
        for r in range(rows):
            for c in range(cols):
                if i >= n:
                    break
                _, frame = annotated[i]
                tile = cv2.resize(frame, (tile_w, tile_h))
                y0 = r * tile_h
                x0 = c * tile_w
                canvas[y0:y0+tile_h, x0:x0+tile_w] = tile
                i += 1
        
        return canvas