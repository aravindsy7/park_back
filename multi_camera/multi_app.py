import os
import time
import logging
import threading
import base64
import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, Response, request, make_response
from flask_cors import CORS

from multi_camera_detector import MultiCameraDetector
from firebase_config import firebase_manager

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Flask app setup ---
app = Flask(__name__)

# --- CORS Configuration ---
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

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 200

# --- Initialize Multi-Camera Detector ---
logger.info("=" * 60)
logger.info("🚀 Initializing Multi-Camera Parking Detector")
logger.info("=" * 60)

# Get camera URLs from environment or use defaults
CAMERA_URLS = os.getenv("CAMERA_URLS", "").split(",")
CAMERA_URLS = [u.strip() for u in CAMERA_URLS if u.strip()]

if not CAMERA_URLS:
    # Fallback to default cameras
    CAMERA_URLS = [
        "http://192.0.0.4:8080/video",
      "http://192.0.0.4:8080/video",
        "http://192.0.0.4:8080/video",
       "http://192.0.0.4:8080/video"
    ]

logger.info(f"📷 Configured cameras: {len(CAMERA_URLS)}")
for i, url in enumerate(CAMERA_URLS):
    logger.info(f"   {i+1}. {url}")

detector = MultiCameraDetector(camera_urls=CAMERA_URLS)
success = detector.start()

if not success:
    logger.error("❌ Failed to start Multi-Camera Detector")
else:
    logger.info("✅ Multi-Camera Detector started successfully")

# --- Background thread for Firestore updates ---
def update_firestore_background():
    """Update Firestore with aggregated parking data every 2 seconds"""
    logger.info("🔄 Starting Firestore update thread (every 2 seconds)...")
    
    while True:
        try:
            data = detector.get_latest_data()
            
            if data.get("total_slots", 0) > 0:
                logger.info(f"📊 Current data: Total={data.get('total_slots')}, Occupied={data.get('occupied_slots')}, Free={data.get('free_slots')}, Active Cameras={data.get('cameras_active')}")
                
                if firebase_manager.db is not None:
                    try:
                        success = firebase_manager.update_parking_data(data)
                        if success:
                            logger.info(f"✅ Firestore UPDATED")
                        else:
                            logger.warning("⚠️ Failed to update Firestore")
                    except Exception as e:
                        logger.exception(f"Exception updating Firestore: {e}")
                else:
                    logger.warning("⚠️ Firestore not initialized")
            else:
                logger.warning("⚠️ No parking data available yet")
            
            time.sleep(2)
            
        except Exception as e:
            logger.exception(f"❌ Error in Firestore update thread: {e}")
            time.sleep(2)

# Start Firestore thread
firestore_thread = threading.Thread(target=update_firestore_background, daemon=True)
firestore_thread.start()

# --- API Endpoints ---

def detector_available():
    return detector is not None and len(detector.cameras) > 0

@app.route('/api/parking-status', methods=['GET', 'OPTIONS'])
def get_parking_status():
    """Get aggregated parking status from all cameras"""
    if not detector_available():
        return jsonify({"error": "Detector not running"}), 503
    
    data = detector.get_latest_data()
    logger.info(f"📊 API Request - Total: {data.get('total_slots')}, Occupied: {data.get('occupied_slots')}, Free: {data.get('free_slots')}")
    return jsonify(data)

@app.route('/api/camera/<int:camera_id>/status', methods=['GET', 'OPTIONS'])
def get_camera_status(camera_id):
    """Get status from specific camera"""
    if not detector_available():
        return jsonify({"error": "Detector not running"}), 503
    
    camera_idx = camera_id - 1
    if camera_idx not in detector.cameras:
        return jsonify({"error": f"Camera {camera_id} not available"}), 404
    
    cam_info = detector.cameras[camera_idx]
    if cam_info["data"] is None:
        return jsonify({"error": f"Camera {camera_id} has no data yet"}), 202
    
    return jsonify(cam_info["data"])

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    firestore_status = "✅ Connected" if firebase_manager.db is not None else "❌ Not Connected"
    cameras_active = len([c for c in detector.cameras.values() if c["connected"]])
    
    return jsonify({
        "status": "running",
        "timestamp": time.time(),
        "detector": "✅ Active" if detector_available() else "❌ Inactive",
        "cameras_total": len(detector.camera_urls),
        "cameras_active": cameras_active,
        "firestore": firestore_status,
        "update_interval": "2 seconds",
        "cors": "✅ Enabled"
    })

def generate_grid_frames():
    """Generate grid view of all cameras"""
    if not detector_available():
        logger.warning("Detector not available for grid frames")
        return
    
    while True:
        try:
            # Get grid display
            grid = detector.grid_display(annotate=True)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.05)
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05)  # ~20 FPS
            
        except GeneratorExit:
            logger.info("Client closed grid stream")
            break
        except Exception as e:
            logger.exception(f"Error in generate_grid_frames: {e}")
            time.sleep(0.1)

def generate_camera_frame(camera_idx):
    """Generate single camera stream"""
    if camera_idx not in detector.cameras:
        return
    
    while True:
        try:
            frame = detector.get_current_frame(camera_idx)
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Get camera data for overlay
            cam_info = detector.cameras[camera_idx]
            if cam_info["data"]:
                data = cam_info["data"]
                
                # Draw stats
                cv2.putText(frame, f"Camera {camera_idx+1}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Total:{data['total_slots']} Occ:{data['occupied_slots']} Free:{data['free_slots']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.05)
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05)
            
        except GeneratorExit:
            logger.info(f"Client closed camera {camera_idx+1} stream")
            break
        except Exception as e:
            logger.exception(f"Error in camera {camera_idx+1} stream: {e}")
            time.sleep(0.1)

@app.route('/video_feed', methods=['GET', 'OPTIONS'])
def video_feed():
    """Grid view of all cameras"""
    if not detector_available():
        return jsonify({"error": "Detector not running"}), 503
    
    return Response(
        generate_grid_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video_feed/camera/<int:camera_id>', methods=['GET', 'OPTIONS'])
def camera_feed(camera_id):
    """Single camera feed"""
    camera_idx = camera_id - 1
    if camera_idx not in detector.cameras:
        return jsonify({"error": f"Camera {camera_id} not available"}), 404
    
    return Response(
        generate_camera_frame(camera_idx),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# --- Application start ---
if __name__ == '__main__':
    try:
        logger.info("=" * 60)
        logger.info("🌐 Starting Multi-Camera Parking Server")
        logger.info("=" * 60)
        logger.info("🎯 Endpoints:")
        logger.info("   📊 Status: http://0.0.0.0:5000/api/parking-status")
        logger.info("   📹 Grid Feed: http://0.0.0.0:5000/video_feed")
        logger.info("   📹 Cam 1: http://0.0.0.0:5000/video_feed/camera/1")
        logger.info("   📹 Cam 2: http://0.0.0.0:5000/video_feed/camera/2")
        logger.info("   ❤️  Health: http://0.0.0.0:5000/api/health")
        logger.info("=" * 60)
        
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        detector.stop()
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        detector.stop()