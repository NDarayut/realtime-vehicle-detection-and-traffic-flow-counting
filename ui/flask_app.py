from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import os
import base64
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Use absolute paths relative to the flask app file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"Output folder: {app.config['OUTPUT_FOLDER']}")

video_path = None
detection_area = []
counting_line = []
processing_progress = 0
processing_status = "idle"
counter_instance = None
current_frame = 0
total_frames = 0
output_video_path = None

class VehicleLineCounter:
    def __init__(self, detection_area, counting_line, model_path="best.pt"):
        self.detection_area = np.array(detection_area, np.int32)
        self.line_start = (int(counting_line[0][0]), int(counting_line[0][1]))
        self.line_end = (int(counting_line[1][0]), int(counting_line[1][1]))
        self.model = YOLO(model_path)
        self.tracked_vehicles = {}
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.class_names = ['Bus', 'Car', 'Motocycle', 'Rickshaw', 'Truck']
        self.colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (255, 0, 255)}
    
    def point_in_polygon(self, point):
        return cv2.pointPolygonTest(self.detection_area, point, False) >= 0
    
    def line_intersection(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.line_start
        x4, y4 = self.line_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def update_tracking(self, track_id, center, class_id, class_name):
        if track_id not in self.tracked_vehicles:
            self.tracked_vehicles[track_id] = {'last_position': center, 'crossed': False, 'class_id': class_id, 'class_name': class_name}
            return False
        vehicle = self.tracked_vehicles[track_id]
        if not vehicle['crossed'] and self.line_intersection(vehicle['last_position'], center):
            vehicle['crossed'] = True
            self.total_count += 1
            self.class_counts[class_name] += 1
            vehicle['last_position'] = center
            return True
        vehicle['last_position'] = center
        return False
    
    def draw_regions(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.detection_area], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [self.detection_area], True, (0, 255, 0), 2)
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 4)
        return frame
    
    def draw_stats(self, frame):
        y_offset = 40
        cv2.rectangle(frame, (10, 10), (300, 40 + len(self.class_names) * 35), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL COUNT: {self.total_count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        for i, class_name in enumerate(self.class_names):
            cv2.putText(frame, f"{class_name}: {self.class_counts[class_name]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
            y_offset += 30
        return frame
    
    def save_to_excel(self, excel_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Vehicle Count Results"
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_alignment = Alignment(horizontal="center", vertical="center")
        ws.merge_cells('A1:C1')
        ws['A1'] = "Vehicle Counting Results"
        ws['A1'].font = Font(bold=True, color="FFFFFF", size=14)
        ws['A1'].alignment = header_alignment
        ws['A1'].fill = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
        ws['A2'] = "Date/Time:"
        ws['B2'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws['A2'].font = Font(bold=True)
        ws['A4'] = "Total Vehicles Counted"
        ws['A4'].font = header_font
        ws['A4'].fill = header_fill
        ws['A4'].alignment = header_alignment
        ws['B4'] = self.total_count
        ws['B4'].font = Font(bold=True, size=12)
        ws['B4'].alignment = Alignment(horizontal="center")
        ws['A6'] = "Vehicle Class"
        ws['B6'] = "Count"
        ws['C6'] = "Percentage"
        for col in ['A', 'B', 'C']:
            ws[f'{col}6'].font = header_font
            ws[f'{col}6'].fill = header_fill
            ws[f'{col}6'].alignment = header_alignment
        row = 7
        for class_name in self.class_names:
            count = self.class_counts[class_name]
            percentage = (count / self.total_count * 100) if self.total_count > 0 else 0
            ws[f'A{row}'] = class_name
            ws[f'B{row}'] = count
            ws[f'C{row}'] = f"{percentage:.1f}%"
            ws[f'B{row}'].alignment = Alignment(horizontal="center")
            ws[f'C{row}'].alignment = Alignment(horizontal="center")
            row += 1
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 15
        wb.save(excel_path)
    
    def process_video(self, video_path, output_path):
        global processing_progress, processing_status, current_frame, total_frames
        try:
            print("Starting video processing...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                processing_status = "error"
                print("Error: Cannot open video")
                return
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video info: {width}x{height}, {fps}fps, {total_frames} frames")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                processing_status = "error"
                print("Error: Cannot create output video")
                cap.release()
                return
            
            processing_status = "processing"
            processing_progress = 0
            current_frame = 0
            processed_count = 0
            
            # Frame skipping: 1 = every frame, 2 = every other frame, 3 = every 3rd frame, etc.
            FRAME_SKIP = 1
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video at frame {current_frame}")
                    break
                
                processed_count += 1
                
                # Run YOLO tracking
                results = self.model.track(source=frame, imgsz=640, conf=0.25, verbose=False, persist=True)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        if self.point_in_polygon(center):
                            class_name = self.class_names[cls]
                            color = self.colors[cls]
                            just_crossed = self.update_tracking(track_id, center, cls, class_name)
                            box_color = (0, 255, 0) if just_crossed else color
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                            cv2.putText(frame, f"{class_name} ID:{track_id}", (int(x1), int(y1) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            cv2.circle(frame, center, 4, (0, 0, 255), -1)
                
                frame = self.draw_regions(frame)
                frame = self.draw_stats(frame)
                out.write(frame)
                
                # Update progress with frame skipping
                current_frame += FRAME_SKIP
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                processing_progress = min(int((current_frame / total_frames) * 100), 100)
                
                if processed_count % 10 == 0:
                    print(f"Progress: {processing_progress}% (frame {current_frame}/{total_frames})")
            
            cap.release()
            out.release()
            processing_status = "complete"
            processing_progress = 100
            current_frame = total_frames
            print(f"Processing complete! Total count: {self.total_count}")
            
        except Exception as e:
            processing_status = "error"
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    return jsonify({'success': True})

@app.route('/first_frame')
def get_first_frame():
    global video_path
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'No video uploaded'}), 400
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': 'Cannot read video'}), 400
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    height, width = frame.shape[:2]
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}', 'width': width, 'height': height})

@app.route('/set_regions', methods=['POST'])
def set_regions():
    global detection_area, counting_line
    data = request.json
    detection_area = data.get('detection_area', [])
    counting_line = data.get('counting_line', [])
    if len(detection_area) != 4 or len(counting_line) != 2:
        return jsonify({'error': 'Invalid regions'}), 400
    return jsonify({'success': True})

@app.route('/process', methods=['POST'])
def process_video():
    global video_path, detection_area, counting_line, counter_instance, processing_status, processing_progress, output_video_path
    if not video_path or not detection_area or not counting_line:
        return jsonify({'error': 'Missing data'}), 400
    print(f"Starting processing with video: {video_path}")
    print(f"Detection area: {detection_area}")
    print(f"Counting line: {counting_line}")
    processing_status = "idle"
    processing_progress = 0
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    print(f"Output will be saved to: {output_video_path}")
    counter_instance = VehicleLineCounter(detection_area, counting_line, model_path="../best.pt")
    def process_thread():
        counter_instance.process_video(video_path, output_video_path)
    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    print("Processing thread started")
    return jsonify({'success': True})

@app.route('/progress')
def get_progress():
    global processing_progress, processing_status, current_frame, total_frames
    print(f"Progress check: {processing_progress}% - Status: {processing_status} - Frame: {current_frame}/{total_frames}")
    return jsonify({
        'progress': processing_progress, 
        'status': processing_status,
        'current_frame': current_frame,
        'total_frames': total_frames
    })

@app.route('/download_video')
def download_video():
    global output_video_path
    try:
        if not output_video_path or not os.path.exists(output_video_path):
            return jsonify({'error': 'No output video found. Processing may not be complete.'}), 404
        print(f"Sending video file: {output_video_path}")
        return send_file(output_video_path, as_attachment=True, download_name='counted_video.mp4', mimetype='video/mp4')
    except Exception as e:
        print(f"Error downloading video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_excel')
def download_excel():
    global counter_instance
    try:
        if not counter_instance:
            return jsonify({'error': 'No results available. Processing may not be complete.'}), 404
        excel_path = os.path.join(app.config['OUTPUT_FOLDER'], f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        print(f"Saving Excel to: {excel_path}")
        counter_instance.save_to_excel(excel_path)
        print(f"Excel file created, sending to browser")
        return send_file(excel_path, as_attachment=True, download_name='counting_results.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        print(f"Error downloading excel: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
