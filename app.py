from flask import Flask, render_template, Response, jsonify, request
import cv2 as cv
import numpy as np
import dlib
import random
from datetime import datetime
import base64
import threading
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config.update(
    SHAPE_PREDICTOR_PATH=os.getenv('SHAPE_PREDICTOR_PATH', 'shape_predictor_68_face_landmarks.dat'),
    DEBUG=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
    PORT=int(os.getenv('PORT', 5001))
)

class BlinkDetector:
    def __init__(self):

        self.target_blink_count = 0
        self.total_blink_count = 0
        self.capture_frame = None
        self.is_capturing = False
        self.previous_eye_states = {}
        
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.2
        self.MIN_BLINK_COUNT = 1
        self.MAX_BLINK_COUNT = 5
        
        self.SHOW_LANDMARKS = False
        self.SHOW_EAR = False
        self.SHOW_BLINK_COUNT = False
        
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(app.config['SHAPE_PREDICTOR_PATH'])
            self.camera = None
            self.lock = threading.Lock()
        except Exception as e:
            app.logger.error(f"Error initializing BlinkDetector: {e}")
            raise

    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio given eye landmark points"""
        try:
            y1 = np.sqrt((eye_points[1].x - eye_points[5].x)**2 + 
                        (eye_points[1].y - eye_points[5].y)**2)
            y2 = np.sqrt((eye_points[2].x - eye_points[4].x)**2 + 
                        (eye_points[2].y - eye_points[4].y)**2)
            
            x = np.sqrt((eye_points[0].x - eye_points[3].x)**2 + 
                       (eye_points[0].y - eye_points[3].y)**2)
            
            ear = (y1 + y2) / (2.0 * x) if x > 0 else 0
            return ear
        except Exception as e:
            app.logger.error(f"Error calculating EAR: {e}")
            return 0

    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on the frame"""
        for point in landmarks:
            cv.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

    def draw_ear_values(self, frame, left_ear, right_ear):
        """Draw eye aspect ratio values on the frame"""
        left_color = (60, 49, 213) if left_ear < self.EYE_ASPECT_RATIO_THRESHOLD else (112, 182, 83)
        right_color = (60, 49, 213) if right_ear < self.EYE_ASPECT_RATIO_THRESHOLD else (112, 182, 83)
        
        left_text = f"Left EAR: {left_ear:.2f}"
        right_text = f"Right EAR: {right_ear:.2f}"
        
        cv.putText(frame, left_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, left_color, 2)
        cv.putText(frame, right_text, (200, 30), cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, right_color, 2)

    def draw_blink_count(self, frame):
        """Draw current blink count on the frame"""
        blink_text = f"Blinks: {self.total_blink_count}"
        cv.putText(frame, blink_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, (0, 255, 0), 2)

    def start_capture(self):
        """Start a new capture session"""
        try:
            self.target_blink_count = random.randint(self.MIN_BLINK_COUNT, self.MAX_BLINK_COUNT)
            self.total_blink_count = 0
            self.capture_frame = None
            self.previous_eye_states = {}
            
            if self.camera is not None:
                self.camera.release()
            self.camera = cv.VideoCapture(0)
            
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
                
            self.is_capturing = True
            return self.target_blink_count
            
        except Exception as e:
            app.logger.error(f"Error in start_capture: {e}")
            raise

    def stop_capture(self):
        try:
            self.is_capturing = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        except Exception as e:
            app.logger.error(f"Error in stop_capture: {e}")

    def process_frame(self, frame):
        """Process a single frame for blink detection"""
        try:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = self.detector(rgb_frame)
            blinks_detected = 0
            
            for face_idx, face in enumerate(faces):
                landmarks = self.predictor(rgb_frame, face).parts()
                
                left_eye = list(landmarks[36:42])
                right_eye = list(landmarks[42:48])
                
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                
                if self.SHOW_LANDMARKS:
                    self.draw_landmarks(frame, landmarks)
                
                if self.SHOW_EAR:
                    self.draw_ear_values(frame, left_ear, right_ear)
                
                current_left_open = left_ear >= self.EYE_ASPECT_RATIO_THRESHOLD
                current_right_open = right_ear >= self.EYE_ASPECT_RATIO_THRESHOLD
                
                previous_state = self.previous_eye_states.get(face_idx, (True, True))
                prev_left_open, prev_right_open = previous_state
                
                if (prev_left_open and prev_right_open and 
                    not current_left_open and not current_right_open):
                    blinks_detected += 1
                
                self.previous_eye_states[face_idx] = (current_left_open, current_right_open)
            
            return blinks_detected
            
        except Exception as e:
            app.logger.error(f"Error in process_frame: {e}")
            return 0

    def get_frame(self):
        """Get and process the current frame"""
        if not self.is_capturing or self.camera is None:
            return None
        
        try:
            with self.lock:
                ret, frame = self.camera.read()
                if not ret:
                    return None
                
                frame = cv.flip(frame, 1)
                
                blinks = self.process_frame(frame)
                self.total_blink_count += blinks
                
                if self.SHOW_BLINK_COUNT:
                    self.draw_blink_count(frame)
                
                if self.total_blink_count >= self.target_blink_count:
                    self.capture_frame = frame.copy()
                    self.stop_capture()
                    return self.capture_frame
                
                return frame
                
        except Exception as e:
            app.logger.error(f"Error in get_frame: {e}")
            return None

detector = BlinkDetector()

def generate_frames():
    """Generator function for video streaming"""
    while True:
        frame = detector.get_frame()
        if frame is not None:
            try:
                ret, buffer = cv.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          buffer.tobytes() + 
                          b'\r\n')
            except Exception as e:
                app.logger.error(f"Error in generate_frames: {e}")
                break

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/start')
def start_session():
    """Start new blink detection session"""
    try:
        target = detector.start_capture()
        return jsonify({'target': target})
    except Exception as e:
        app.logger.error(f"Error starting session: {e}")
        return jsonify({'error': 'Failed to start camera'}), 500

@app.route('/stop')
def stop_session():
    """Stop current session"""
    try:
        detector.stop_capture()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        app.logger.error(f"Error stopping session: {e}")
        return jsonify({'error': 'Failed to stop session'}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Get current session status"""
    try:
        return jsonify({
            'current': detector.total_blink_count,
            'target': detector.target_blink_count,
            'completed': detector.total_blink_count >= detector.target_blink_count
        })
    except Exception as e:
        app.logger.error(f"Error getting status: {e}")
        return jsonify({'error': 'Failed to get status'}), 500

@app.route('/capture')
def get_capture():
    """Get captured frame when target is reached"""
    try:
        if detector.capture_frame is not None:
            ret, buffer = cv.imencode('.jpg', detector.capture_frame)
            if ret:
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({'image': image_base64})
        return jsonify({'image': None})
    except Exception as e:
        app.logger.error(f"Error getting capture: {e}")
        return jsonify({'error': 'Failed to get capture'}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update display settings and detection parameters"""
    try:
        settings = request.json
        
        detector.SHOW_LANDMARKS = settings.get('show_landmarks', False)
        detector.SHOW_EAR = settings.get('show_ear', False)
        detector.SHOW_BLINK_COUNT = settings.get('show_blink_count', True)
        detector.EYE_ASPECT_RATIO_THRESHOLD = float(settings.get('ear_threshold', 0.2))
        detector.MIN_BLINK_COUNT = int(settings.get('min_blinks', 1))
        detector.MAX_BLINK_COUNT = int(settings.get('max_blinks', 5))
        
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error updating settings: {e}")
        return jsonify({'error': 'Failed to update settings'}), 500

if __name__ == '__main__':
    port = app.config['PORT']
    debug = app.config['DEBUG']
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )