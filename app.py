from flask import Flask, jsonify, Response, request, make_response
from flask_cors import CORS
from parking_detector import ParkingDetector
from firebase_config import firebase_manager
import os
import time
import logging
import threading
import base64
import cv2
from dotenv import load_dotenv

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Flask app setup ---
app = Flask(__name__)

# --- PROPER CORS CONFIGURATION ---
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "max_age": 3600
    },
    r"/video_feed": {
        "origins": ["*"],
        "methods": ["GET", "OPTIONS"],
        "max_age": 3600
    }
})

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 200

# --- Initialize detector ---
logger.info("=" * 60)
logger.info("Initializing Parking Detector...")
logger.info("=" * 60)

    

VIDEO_PATH = os.getenv("VIDEO_PATH", "http://192.0.0.4:8080/video")
logger.info(f"📷 Camera URL: {VIDEO_PATH}")

detector = None

try:
    detector = ParkingDetector(video_path=VIDEO_PATH)
    detector.start()
    time.sleep(1.0)

    if detector is None or not getattr(detector, "running", False):
        logger.error("❌ Detector failed to start or is not running.")
    else:
        frame = detector.get_current_frame()
        if frame is None:
            logger.warning("⚠️ Detector started but no frames received yet.")
        else:
            logger.info("✅ Initial frame received from detector.")
except Exception as e:
    logger.exception(f"❌ Failed to initialize ParkingDetector: {e}")
    detector = None

# --- Shared state for latest frame ---
latest_frame_base64 = None
latest_frame_timestamp = None
latest_frame_lock = threading.Lock()

# --- Background thread to update Firestore every 2 seconds ---
def update_firestore_background():
    """Continuously update parking data every 2 seconds."""
    global latest_frame_base64, latest_frame_timestamp
    logger.info("🔄 Starting parking data update thread (every 2 seconds)...")

    while True:
        try:
            if detector is None or not getattr(detector, "running", False):
                time.sleep(2)
                continue

            data = detector.get_latest_data()

            logger.debug("📊 Current data from detector:")
            logger.debug(f"   Total Slots: {data.get('total_slots')}")
            logger.debug(f"   Occupied: {data.get('occupied_slots')}")
            logger.debug(f"   Free: {data.get('free_slots')}")

            firebase_manager.update_parking_data(data)

            # Capture current frame
            frame = detector.get_current_frame()
            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        encoded = base64.b64encode(buffer.tobytes()).decode('utf-8')
                        timestamp = int(time.time())

                        with latest_frame_lock:
                            latest_frame_base64 = encoded
                            latest_frame_timestamp = timestamp

                        firebase_manager.upload_video_frame(encoded, timestamp)
                except Exception as e:
                    logger.exception(f"Exception encoding frame: {e}")

            time.sleep(2)
        except Exception as e:
            logger.exception(f"❌ Error in update thread: {e}")
            time.sleep(2)

# Start the Firestore thread as daemon
firestore_thread = threading.Thread(target=update_firestore_background, daemon=True)
firestore_thread.start()

# --- API endpoints ---

def detector_available():
    return detector is not None and getattr(detector, "running", False)

@app.route('/api/parking-status', methods=['GET', 'OPTIONS'])
def get_parking_status():
    """Return the latest parking status JSON."""
    if not detector_available():
        logger.error("❌ Detector not running")
        return jsonify({"error": "Detector not running"}), 503

    data = detector.get_latest_data()
    logger.info(f"📊 API Status Request - Total: {data.get('total_slots')}, Occupied: {data.get('occupied_slots')}, Free: {data.get('free_slots')}")
    return jsonify(data)

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint."""
    firestore_status = "✅ Connected" if firebase_manager.db is not None else "❌ Not Connected"
    det_status = "inactive"
    if detector is None:
        det_status = "missing"
    elif getattr(detector, "running", False):
        det_status = "active"
    else:
        det_status = "stopped"

    return jsonify({
        "status": "running",
        "timestamp": time.time(),
        "detector": det_status,
        "firestore": firestore_status,
        "update_interval": "2 seconds",
        "cors": "✅ Enabled"
    })

@app.route('/api/latest-frame', methods=['GET', 'OPTIONS'])
def get_latest_frame():
    """Return the latest frame (base64) used for Firebase mode."""
    with latest_frame_lock:
        if latest_frame_base64:
            return jsonify({
                "frame": latest_frame_base64,
                "timestamp": latest_frame_timestamp,
                "type": "jpeg"
            })
    return jsonify({"error": "No frame available"}), 404

def generate_frames():
    """MJPEG frame generator for video_feed endpoint."""
    if not detector_available():
        logger.warning("/video_feed requested but detector not available.")
        return

    while True:
        try:
            frame = detector.get_current_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            data = detector.get_latest_data()

            # Draw detections
            for detection in data.get('detections', []):
                box = detection.get('box')
                if box and len(box) == 4:
                    x, y, w, h = box
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    label = detection.get('label', 'obj')
                    conf = detection.get('confidence', 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label} {int(conf*100)}%"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1-25), (x1+text_width+10, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label_text, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw slots
            for slot in data.get('slots', []):
                x = slot.get('x'); y = slot.get('y'); w = slot.get('width'); h = slot.get('height')
                if x is None or y is None or w is None or h is None:
                    continue
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                status = slot.get('status', 'unknown')
                color = (0, 255, 0) if status == 'available' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Overlay summary
            total_slots = data.get('total_slots', 0)
            occupied_slots = data.get('occupied_slots', 0)
            free_slots = data.get('free_slots', 0)
            summary = f"Total: {total_slots} | Occupied: {occupied_slots} | Free: {free_slots}"
            cv2.rectangle(frame, (10, 10), (700, 50), (0, 0, 0), -1)
            cv2.putText(frame, summary, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.01)
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
        except GeneratorExit:
            logger.info("Client closed video stream.")
            break
        except Exception as e:
            logger.exception(f"Error in generate_frames: {e}")
            time.sleep(0.1)

@app.route('/video_feed', methods=['GET', 'OPTIONS'])
def video_feed():
    """Return multipart MJPEG stream of annotated frames."""
    if not detector_available():
        return jsonify({"error": "Detector not running"}), 503

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



# At the end of your app.py file, change:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    