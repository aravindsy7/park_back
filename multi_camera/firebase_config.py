import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

class FirebaseManager:
    def __init__(self):
        self.db = None
        self.initialized = False
        
        service_account_path = os.path.join(
            os.path.dirname(__file__), 
            'serviceAccountKey.json'
        )
        
        if os.path.exists(service_account_path):
            try:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.initialized = True
                logger.info("✅ Firebase Firestore connected")
            except Exception as e:
                logger.debug(f"Firebase unavailable: {str(e)[:50]}")
                self.db = None
        else:
            logger.info("ℹ️ System working in LOCAL MODE (no cloud sync)")
            self.db = None
    
    def update_parking_data(self, data):
        if self.db is None:
            return False
        try:
            self.db.collection('parking').document('current').set(data)
            return True
        except:
            return False
    
    def upload_video_frame(self, frame_base64, timestamp):
        if self.db is None:
            return False
        try:
            self.db.collection('frames').document(str(timestamp)).set({
                "image": frame_base64,
                "timestamp": timestamp
            })
            return True
        except:
            return False

firebase_manager = FirebaseManager()
