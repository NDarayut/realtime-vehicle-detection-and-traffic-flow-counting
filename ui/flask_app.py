from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import openpyxl
import os
import base64
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global State
video_path = None
polygons_data = [] # List of lists of points
lines_data = []    # List of lists of points
processing_fps = 30
processing_progress = 0
processing_status = "idle"
is_cancelled = False 
counter_instance = None
current_frame = 0
total_frames = 0
output_video_path = None
show_detection_area = True

class VehicleLineCounter:
    def __init__(self, polygons, lines, show_area=True, model_path="../best.pt"):
        self.show_area = show_area
        # Convert all polygon point lists to numpy arrays
        self.polygons = [np.array(p, np.int32) for p in polygons]
        # Lines are stored as lists of point tuples
        self.lines = lines 
        
        self.model = YOLO(model_path) 
        self.tracked_vehicles = {}
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.class_names = ['Bus', 'Car', 'Motocycle', 'Rickshaw', 'Truck']
        self.colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (255, 0, 255)}

    def is_inside_any_polygon(self, point):
        for poly in self.polygons:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return True
        return False

    def intersect(self, A, B, C, D):
        """Standard line segment intersection check"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def crossed_any_line(self, old_pos, new_pos):
        """Checks if movement from old_pos to new_pos crosses any segment of any line"""
        for line in self.lines:
            # Each line can have multiple points (polyline)
            for i in range(len(line) - 1):
                p1 = line[i]
                p2 = line[i+1]
                if self.intersect(old_pos, new_pos, p1, p2):
                    return True
        return False

    def update_tracking(self, track_id, center, class_name):
        if track_id not in self.tracked_vehicles:
            self.tracked_vehicles[track_id] = {'last_position': center, 'crossed': False}
            return False
        
        vehicle = self.tracked_vehicles[track_id]
        if not vehicle['crossed']:
            if self.crossed_any_line(vehicle['last_position'], center):
                vehicle['crossed'] = True
                self.total_count += 1
                self.class_counts[class_name] += 1
                return True
        
        vehicle['last_position'] = center
        return False

    def draw_ui(self, frame):
        # Draw Polygons
        if self.show_area:
            overlay = frame.copy()
            for poly in self.polygons:
                cv2.fillPoly(overlay, [poly], (0, 255, 0))
                cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw Lines (Yellow)
        for line in self.lines:
            pts = np.array(line, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 3)
        
        # Dashboard
        y_offset = 40
        cv2.rectangle(frame, (10, 10), (250, 40 + len(self.class_names) * 30), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL: {self.total_count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, name in enumerate(self.class_names):
            y_offset += 30
            cv2.putText(frame, f"{name}: {self.class_counts[name]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.get(i, (255,255,255)), 2)
        return frame

    def process_video(self, video_path, output_path, target_fps):
        global processing_progress, processing_status, current_frame, total_frames, is_cancelled
        try:
            cap = cv2.VideoCapture(video_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))
            processing_status = "processing"
            
            while cap.isOpened():
                if is_cancelled: break
                ret, frame = cap.read()
                if not ret: break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # YOLO Tracking
                results = self.model.track(source=frame, persist=True, verbose=False, conf=0.25)
                
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, tid, cls in zip(boxes, track_ids, classes):
                        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        
                        # Only track/count if center is in at least ONE of the detection polygons
                        if self.is_inside_any_polygon(center):
                            c_name = self.class_names[cls] if cls < len(self.class_names) else "Vehicle"
                            just_crossed = self.update_tracking(tid, center, c_name)
                            
                            color = (0, 255, 0) if just_crossed else self.colors.get(cls, (255,255,255))
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

                frame = self.draw_ui(frame)
                out.write(frame)
                processing_progress = int((current_frame / total_frames) * 100)
            
            cap.release()
            out.release()
            processing_status = "complete" if not is_cancelled else "idle"
            
        except Exception as e:
            print(f"Error: {e}")
            processing_status = "error"

# --- Flask Routes ---

@app.route('/set_regions', methods=['POST'])
def set_regions():
    global polygons_data, lines_data, processing_fps, show_detection_area
    data = request.json
    # Now receiving lists of lists from your new frontend
    polygons_data = data.get('polygons', [])
    lines_data = data.get('lines', [])
    processing_fps = int(data.get('fps', 30))
    show_detection_area = data.get('show_detection_area', True)
    return jsonify({'success': True})

@app.route('/process', methods=['POST'])
def process():
    global counter_instance, output_video_path, is_cancelled
    is_cancelled = False
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    
    counter_instance = VehicleLineCounter(polygons_data, lines_data, show_area=show_detection_area)
    
    t = threading.Thread(target=counter_instance.process_video, args=(video_path, output_video_path, processing_fps))
    t.start()
    return jsonify({'success': True})

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path, is_cancelled
    is_cancelled = False 
    file = request.files['video']
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    return jsonify({'success': True})

@app.route('/first_frame')
def get_first_frame():
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    global video_path, detection_area, counting_line, processing_fps, show_detection_area
    global counter_instance, processing_status, processing_progress, output_video_path, is_cancelled
    
    is_cancelled = False
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f'out_{datetime.now().strftime("%H%M%S")}.mp4')
    
    # Initialize instance with the show_area flag
    counter_instance = VehicleLineCounter(
        detection_area, 
        counting_line, 
        show_area=show_detection_area, 
        model_path="../best.pt"
    )
    
    thread = threading.Thread(target=counter_instance.process_video, args=(video_path, output_video_path, processing_fps))
    thread.daemon = True
    thread.start()
    return jsonify({'success': True})

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    global is_cancelled
    is_cancelled = True
    return jsonify({'success': True})

@app.route('/progress')
def get_progress():
    global processing_progress, processing_status, current_frame, total_frames
    return jsonify({
        'progress': processing_progress, 
        'status': processing_status,
        'current_frame': current_frame,
        'total_frames': total_frames
    })

@app.route('/download_video')
def download_video():
    return send_file(output_video_path, as_attachment=True)

@app.route('/download_excel')
def download_excel():
    excel_path = os.path.join(app.config['OUTPUT_FOLDER'], 'report.xlsx')
    counter_instance.save_to_excel(excel_path)
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)