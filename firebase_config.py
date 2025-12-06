import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
import logging
import json
import time

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Firebase
firebase_initialized = False

try:
    # Try to load from serviceAccountKey.json
    if os.path.exists('serviceAccountKey.json'):
        logger.info("Loading Firebase credentials from serviceAccountKey.json")
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
        firebase_initialized = True
        logger.info("✅ Firebase initialized successfully with service account")
    else:
        logger.warning("⚠️ serviceAccountKey.json not found - Firebase disabled")
        firebase_initialized = False
except Exception as e:
    logger.warning(f"⚠️ Firebase initialization skipped: {e}")
    firebase_initialized = False

# Get Firestore database client
if firebase_initialized:
    try:
        db = firestore.client()
        logger.info("✅ Firestore client created successfully")
    except Exception as e:
        logger.warning(f"⚠️ Firestore client error (will continue without Firebase): {e}")
        db = None
else:
    db = None

class FirebaseManager:
    def __init__(self):
        self.db = db
        self.parking_collection = 'parking'
        self.parking_doc = 'current'
        self.video_collection = 'video_frames'
        
        if self.db is None:
            logger.warning("⚠️ Firestore database is not initialized - running in local mode only")
    
    def update_parking_data(self, data):
        """Update parking data in Firestore"""
        if self.db is None:
            logger.debug("Firestore disabled - skipping parking data update")
            return True
        
        try:
            parking_doc = {
                'total_slots': data.get('total_slots', 0),
                'occupied_slots': data.get('occupied_slots', 0),
                'free_slots': data.get('free_slots', 0),
                'slots': data.get('slots', []),
                'detections': data.get('detections', []),
                'timestamp': int(time.time()),
                'updatedAt': time.time()
            }
            
            self.db.collection(self.parking_collection).document(self.parking_doc).set(parking_doc)
            logger.debug(f"✅ Updated Firestore - Total: {parking_doc['total_slots']}, Occupied: {parking_doc['occupied_slots']}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error updating Firestore: {e}")
            return False

    def upload_video_frame(self, frame_base64, timestamp):
        """Upload video frame metadata to Firestore"""
        if self.db is None:
            logger.debug("Firestore disabled - skipping frame upload")
            return True
        
        try:
            frame_data = {
                'timestamp': timestamp,
                'frameSize': len(frame_base64),
                'type': 'video_frame',
                'createdAt': time.time()
            }
            
            self.db.collection(self.video_collection).add(frame_data)
            logger.debug(f"✅ Uploaded frame metadata: {timestamp}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error uploading frame: {e}")
            return False
    
    def get_parking_data(self):
        """Get current parking data from Firestore"""
        if self.db is None:
            logger.debug("Firestore disabled - returning None")
            return None
        
        try:
            doc = self.db.collection(self.parking_collection).document(self.parking_doc).get()
            if doc.exists:
                logger.debug("✅ Retrieved parking data from Firestore")
                return doc.to_dict()
            logger.debug("⚠️ Parking document does not exist in Firestore")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error getting parking data: {e}")
            return None
    
    @staticmethod
    def _convert_data(data):
        """Convert numpy types to Python types for Firestore compatibility"""
        if isinstance(data, dict):
            return {k: FirebaseManager._convert_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [FirebaseManager._convert_data(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            try:
                import numpy as np
                if isinstance(data, np.integer):
                    return int(data)
                elif isinstance(data, np.floating):
                    return float(data)
                elif isinstance(data, np.ndarray):
                    return data.tolist()
            except:
                pass
            return str(data)

# Create global instance
firebase_manager = FirebaseManager()