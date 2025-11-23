import os
import re
import cv2
import torch
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import tempfile
import shutil

# OCR support
try:
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
        print("✅ Tesseract OCR tersedia")
    except Exception:
        OCR_AVAILABLE = False
        print("⚠️  Tesseract OCR engine tidak terinstall")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  pytesseract tidak terinstall, OCR tidak tersedia")

app = FastAPI(title="Traffic Video Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "/app/models/YOLOv8s/train/weights/best.pt"
VIDEO_DIR = Path("/app/assets/video/input_vidio")
TEMP_DIR = Path("/tmp/video_processing")
TEMP_DIR.mkdir(exist_ok=True)

# Enable cuDNN optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Global model instance
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ROI Points (from original script)
FIXED_ROI_POINTS = [
    (212, 1022),
    (858, 1085),
    (894, 1369),
    (70, 1296),
    (55, 1274)
]


class VehicleTracker:
    """
    Tracking kendaraan menggunakan IoU + Centroid Distance untuk mencegah double counting.
    Sama seperti di yolo_counting_with_area.py
    """
    def __init__(self):
        self.tracked_objects = {}  # {track_id: {'class': class_name, 'counted': bool, 'positions': [], ...}}
        self.next_object_id = 0
        self.counted_ids = set()

    def reset(self):
        """Reset semua tracking (dipanggil saat menit berubah)"""
        self.tracked_objects.clear()
        self.counted_ids.clear()
        self.next_object_id = 0
        print("[TRACKER] Reset - cleared all tracks")

    def calculate_iou(self, box1, box2):
        """Calculate IoU (Intersection over Union) antara dua box"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection area
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height

        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def calculate_centroid_distance(self, center1, center2):
        """Calculate Euclidean distance antara dua centroid"""
        import math
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def match_detection_to_track(self, detection_box, detection_class, detection_center):
        """
        Match detection ke existing track berdasarkan:
        1. Class harus sama
        2. IoU > 0.2 ATAU centroid distance < 100px
        """
        best_match_id = None
        best_score = 0  # Higher is better

        for track_id, track_data in self.tracked_objects.items():
            # Class harus sama
            if track_data['class'] != detection_class:
                continue

            # Get last position dan center dari track
            if len(track_data['positions']) > 0:
                last_box = track_data['positions'][-1]
                last_center = track_data.get('last_center', (0, 0))

                # Hitung IoU
                iou = self.calculate_iou(detection_box, last_box)

                # Hitung centroid distance
                centroid_dist = self.calculate_centroid_distance(detection_center, last_center)

                # Match jika IoU > 0.2 ATAU centroid distance < 100px
                if iou > 0.2:
                    score = iou + 1  # IoU match gets higher priority
                    if score > best_score:
                        best_score = score
                        best_match_id = track_id
                elif centroid_dist < 100:  # Fallback: centroid distance
                    score = 1 - (centroid_dist / 100)  # Closer = higher score
                    if score > best_score and best_match_id is None:  # Only use if no IoU match
                        best_score = score
                        best_match_id = track_id

        return best_match_id

    def update_tracks(self, current_detections):
        """
        Update tracking objects dan hitung objek baru yang masuk ROI.
        Returns: new_counts dict dengan jumlah kendaraan BARU yang dihitung
        """
        # Tandai semua tracks sebagai not updated
        for track_id in self.tracked_objects:
            self.tracked_objects[track_id]['updated'] = False

        new_counts = defaultdict(int)
        matched_track_ids = set()  # Track IDs yang sudah di-match di frame ini

        # Update atau create new tracks
        for detection in current_detections:
            box = detection['box']
            class_name = detection['class_name']
            center = detection['center']

            # Cari matching track
            matched_id = self.match_detection_to_track(box, class_name, center)

            # Pastikan track belum di-match oleh detection lain di frame ini
            if matched_id is not None and matched_id in matched_track_ids:
                matched_id = None  # Force create new track

            if matched_id is not None:
                # Update existing track
                matched_track_ids.add(matched_id)
                track = self.tracked_objects[matched_id]
                track['positions'].append(box)
                track['last_center'] = center
                track['updated'] = True
                track['frames_missing'] = 0

                # Keep only last N positions untuk memory efficiency
                if len(track['positions']) > 30:
                    track['positions'] = track['positions'][-30:]

                # Jika belum dihitung dan sudah cukup frames, hitung
                if not track['counted'] and len(track['positions']) >= 3:
                    # Objek sudah tracked minimal 3 frame, count as valid
                    track['counted'] = True
                    self.counted_ids.add(matched_id)

                    # Map class_name ke kategori
                    class_lower = class_name.lower()
                    if 'car' in class_lower:
                        new_counts['car'] += 1
                    elif 'motor' in class_lower or 'bike' in class_lower:
                        new_counts['motorcycle'] += 1
                    elif 'truck' in class_lower or 'bus' in class_lower:
                        new_counts['heavy'] += 1

                    print(f"   ✅ COUNTED: ID={matched_id} Class='{class_name}' (tracked {len(track['positions'])} frames)")

            else:
                # Create new track
                new_id = self.next_object_id
                self.next_object_id += 1

                self.tracked_objects[new_id] = {
                    'class': class_name,
                    'positions': [box],
                    'last_center': center,
                    'counted': False,
                    'updated': True,
                    'frames_missing': 0
                }

        # Remove tracks yang hilang terlalu lama
        tracks_to_remove = []
        for track_id, track in self.tracked_objects.items():
            if not track['updated']:
                track['frames_missing'] += 1
                if track['frames_missing'] > 30:  # Lost for 30 frames
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]

        return new_counts

    def get_active_tracks_count(self):
        """Get jumlah tracks yang masih aktif"""
        return len([t for t in self.tracked_objects.values() if t['updated']])

    def get_counted_total(self):
        """Get total objek yang sudah dihitung"""
        return len(self.counted_ids)


class StreamState:
    """Global state untuk tracking OCR dan counting per stream"""
    def __init__(self):
        self.tracker = VehicleTracker()
        self.reset()

    def reset(self):
        self.current_minute = -1
        self.last_known_minute = -1
        self.last_known_hour = 0
        self.minute_counts = {'car': 0, 'motorcycle': 0, 'heavy': 0}
        self.ocr_text = "Waiting..."
        self.tracker.reset()

    def should_reset_counts(self, new_minute: int) -> bool:
        """Cek apakah perlu reset counting (menit berubah)"""
        if self.current_minute == -1:
            return False
        return new_minute != self.current_minute

    def reset_for_new_minute(self):
        """Reset counting dan tracker untuk menit baru"""
        self.minute_counts = {'car': 0, 'motorcycle': 0, 'heavy': 0}
        self.tracker.reset()


# Global stream state
stream_state = StreamState()


def get_dynamic_ocr_roi(current_hour: int = None, last_known_hour: int = 0):
    """Get dynamic OCR ROI based on expected hour format"""
    hour = current_hour if current_hour is not None else last_known_hour
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12

    # 2-digit hour (10, 11, 12) → width 125
    # 1-digit hour (1-9) → width 100
    if hour_12 >= 10:
        return (0, 0, 125, 60)
    else:
        return (0, 0, 100, 60)


def extract_time_from_frame(frame, roi=None, last_known_hour: int = 0):
    """Extract time dari frame menggunakan OCR pada area tertentu"""
    if not OCR_AVAILABLE:
        return None, "OCR not installed"

    # Jika ROI tidak diberikan, gunakan dynamic ROI
    if roi is None:
        roi = get_dynamic_ocr_roi(last_known_hour=last_known_hour)

    try:
        x1, y1, x2, y2 = roi
        time_region = frame[y1:y2, x1:x2]

        if time_region.size == 0:
            return None, "Empty ROI"

        # ENHANCED PREPROCESSING untuk OCR yang lebih robust
        gray = cv2.cvtColor(time_region, cv2.COLOR_BGR2GRAY)

        # Upscale image untuk OCR lebih baik (2x)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Try multiple threshold methods
        methods = [
            ('BINARY', cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)[1]),
            ('BINARY_INV', cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)[1]),
            ('OTSU', cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ('ADAPTIVE', cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        ]

        for method_name, thresh in methods:
            # OCR dengan config untuk single line text
            text = pytesseract.image_to_string(thresh, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:.')

            # Parse time format (HH:MM atau H:MM atau HH.MM)
            time_match = re.search(r'(\d{1,2})[:\.\s](\d{2})', text)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))

                # Validate result (jam 1-12, menit 0-59)
                if 1 <= hour <= 12 and 0 <= minute <= 59:
                    return (hour, minute), None

        return None, "No valid time found"
    except Exception as e:
        return None, f"Error: {str(e)[:30]}"


def convert_12h_to_24h(ocr_hour: int, ocr_minute: int, file_hour: int = None):
    """Convert 12-hour format ke 24-hour berdasarkan file hour"""
    if file_hour is None:
        return ocr_hour, ocr_minute

    # Jika file hour >= 12 dan OCR hour < 12, tambahkan 12
    if file_hour >= 12 and ocr_hour < 12:
        if ocr_hour == 12:
            return ocr_hour, ocr_minute
        return ocr_hour + 12, ocr_minute

    # Jika file hour < 12 dan OCR hour adalah 12
    if file_hour < 12 and ocr_hour == 12:
        return 0, ocr_minute

    return ocr_hour, ocr_minute


def get_model():
    """Load YOLO model lazily"""
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            model.to(device)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model


def get_video_files():
    """Get all video files sorted by time"""
    videos = []
    print(f"[DEBUG] Looking for videos in: {VIDEO_DIR}")
    print(f"[DEBUG] VIDEO_DIR exists: {VIDEO_DIR.exists()}")

    if not VIDEO_DIR.exists():
        print(f"[DEBUG] VIDEO_DIR does not exist!")
        return []

    all_files = list(VIDEO_DIR.glob('*.mp4'))
    print(f"[DEBUG] Found {len(all_files)} mp4 files")

    for f in all_files:
        match = re.search(r'(\d{8})_(\d{6})', f.name)
        if match:
            time_str = match.group(2)
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            videos.append({
                'path': str(f),
                'name': f.name,
                'hour': hour,
                'minute': minute
            })
            print(f"[DEBUG] Video found: {f.name} -> hour={hour}, minute={minute}")
        else:
            print(f"[DEBUG] Video skipped (no match): {f.name}")

    print(f"[DEBUG] Total valid videos: {len(videos)}")
    return sorted(videos, key=lambda x: (x['hour'], x['minute']))


def get_nearest_videos(current_hour: int, current_minute: int):
    """Get videos nearest to current time"""
    videos = get_video_files()
    if not videos:
        return []

    # Determine target hour
    if current_minute < 30:
        target_hour = current_hour
    else:
        target_hour = (current_hour + 1) % 24

    # Find videos for target hour
    hour_videos = [v for v in videos if v['hour'] == target_hour]

    if hour_videos:
        return hour_videos

    # Fallback: find closest hour
    available_hours = list(set(v['hour'] for v in videos))
    if available_hours:
        closest = min(available_hours, key=lambda h: abs(h - target_hour))
        return [v for v in videos if v['hour'] == closest]

    return videos[:6]


def is_point_in_roi(point, roi_polygon):
    """Check if point is inside ROI"""
    if roi_polygon is None:
        return True
    result = cv2.pointPolygonTest(roi_polygon, point, False)
    return result >= 0


def get_class_color(class_name: str):
    """Get color for vehicle class"""
    class_lower = class_name.lower()
    if 'car' in class_lower:
        return (255, 100, 100)  # Blue
    elif 'motor' in class_lower or 'bike' in class_lower:
        return (100, 255, 100)  # Green
    elif 'truck' in class_lower or 'bus' in class_lower:
        return (100, 100, 255)  # Red
    return (100, 200, 255)  # Orange


def process_frame_with_detection(frame, yolo_model, roi_polygon, draw_stats=True):
    """
    Process single frame with YOLO detection.
    TIDAK melakukan counting - hanya deteksi dan return boxes untuk tracker.
    """
    detected_boxes = []  # Store box data for tracker

    results = yolo_model.predict(
        frame,
        conf=0.25,
        iou=0.45,
        device=device,
        verbose=False
    )

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if not is_point_in_roi(center, roi_polygon):
                continue

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[cls]

            # Store box data untuk tracker dan drawing
            # 'box' tuple untuk IoU calculation di tracker
            detected_boxes.append({
                'box': (x1, y1, x2, y2),  # Tuple untuk tracker
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center': center,
                'class_name': class_name,
                'conf': conf,
                'color': get_class_color(class_name)
            })

    # Draw all boxes
    frame = draw_detection_boxes(frame, detected_boxes)

    # Draw ROI
    if roi_polygon is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [roi_polygon], True, (0, 255, 0), 2)

    # Return frame dan detected_boxes (TANPA counts - counting dilakukan oleh tracker)
    return frame, detected_boxes


def draw_detection_boxes(frame, boxes):
    """Draw cached detection boxes on frame"""
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        center = box['center']
        color = box['color']
        class_name = box['class_name']
        conf = box['conf']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # Label
        label = f'{class_name} {conf:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_stats_overlay(frame, counts):
    """Draw stats overlay on the LEFT side of the frame"""
    height, width = frame.shape[:2]

    # Draw stats on LEFT side (large area) with bigger font
    box_width = 280
    box_height = 160
    margin = 15

    # Semi-transparent background on LEFT
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_width, margin + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text with larger font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    y_start = margin + 35
    line_height = 35

    cv2.putText(frame, f"Car: {counts.get('car', 0)}", (margin + 15, y_start),
                font, font_scale, (255, 100, 100), thickness)
    cv2.putText(frame, f"Motorcycle: {counts.get('motorcycle', 0)}", (margin + 15, y_start + line_height),
                font, font_scale, (100, 255, 100), thickness)
    cv2.putText(frame, f"Heavy: {counts.get('heavy', 0)}", (margin + 15, y_start + line_height * 2),
                font, font_scale, (100, 100, 255), thickness)
    cv2.putText(frame, f"Total: {sum(counts.values())}", (margin + 15, y_start + line_height * 3),
                font, font_scale, (255, 255, 255), thickness)

    return frame


def draw_stats_overlay_with_ocr(frame, counts, ocr_text):
    """Draw stats overlay dengan info waktu OCR dan tracking info"""
    global stream_state
    height, width = frame.shape[:2]

    # Draw stats on LEFT side
    box_width = 320
    box_height = 230
    margin = 15

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_width, margin + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    y_start = margin + 30
    line_height = 30

    # OCR Time
    cv2.putText(frame, f"Time: {ocr_text}", (margin + 15, y_start),
                font, font_scale, (255, 0, 255), thickness)  # Magenta

    # Counts (accumulated per minute)
    cv2.putText(frame, f"Car: {counts.get('car', 0)}", (margin + 15, y_start + line_height),
                font, font_scale, (255, 100, 100), thickness)
    cv2.putText(frame, f"Motorcycle: {counts.get('motorcycle', 0)}", (margin + 15, y_start + line_height * 2),
                font, font_scale, (100, 255, 100), thickness)
    cv2.putText(frame, f"Heavy: {counts.get('heavy', 0)}", (margin + 15, y_start + line_height * 3),
                font, font_scale, (100, 100, 255), thickness)
    cv2.putText(frame, f"Total: {sum(counts.values())}", (margin + 15, y_start + line_height * 4),
                font, font_scale, (255, 255, 255), thickness)

    # Tracking info
    active_tracks = stream_state.tracker.get_active_tracks_count()
    total_tracked = stream_state.tracker.get_counted_total()
    cv2.putText(frame, f"Active Tracks: {active_tracks}", (margin + 15, y_start + line_height * 5),
                font, 0.5, (0, 255, 255), 1)  # Cyan
    cv2.putText(frame, f"Unique Counted: {total_tracked}", (margin + 15, y_start + line_height * 5 + 20),
                font, 0.5, (0, 255, 255), 1)

    # Info: Reset on minute change
    cv2.putText(frame, "Reset: per minute change", (margin + 15, y_start + line_height * 6 + 10),
                font, 0.45, (150, 150, 150), 1)

    return frame


def generate_video_stream(video_path: str, with_detection: bool = True, file_hour: int = None):
    """Generate video stream with optional YOLO detection and OCR-based minute reset"""
    import time
    global stream_state

    # Reset tracker saat stream baru dimulai
    print(f"[STREAM] Starting new stream: {video_path}")
    stream_state.reset()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    # Get original FPS for proper playback speed
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25
    frame_delay = 1.0 / original_fps

    roi_polygon = np.array(FIXED_ROI_POINTS, dtype=np.int32) if FIXED_ROI_POINTS else None
    yolo_model = get_model() if with_detection else None

    # PENTING: Deteksi setiap frame untuk tracking yang akurat (tidak ada frame_skip)
    ocr_interval = 30  # Run OCR every 30 frames (1 second at 30fps)
    frame_count = 0

    # Cache last detection results for drawing
    last_boxes = []
    last_scaled_roi = None

    # Initialize file_hour from filename if not provided
    if file_hour is None:
        match = re.search(r'(\d{8})_(\d{6})', video_path)
        if match:
            time_str = match.group(2)
            file_hour = int(time_str[:2])

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                # Loop video - reset tracker karena objek akan muncul ulang dari awal
                print("[STREAM] Video loop - resetting tracker to prevent double counting")
                stream_state.tracker.reset()
                stream_state.minute_counts = {'car': 0, 'motorcycle': 0, 'heavy': 0}
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue

            frame_count += 1
            original_frame = frame.copy()  # Keep original for OCR

            # Resize frame for streaming
            height, width = frame.shape[:2]
            scale = 720 / width if width > 720 else 1
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

            # Run OCR every ocr_interval frames to detect minute change
            if frame_count % ocr_interval == 1 and OCR_AVAILABLE:
                ocr_result, error_msg = extract_time_from_frame(
                    original_frame,
                    last_known_hour=stream_state.last_known_hour
                )

                if ocr_result:
                    ocr_hour_raw, ocr_minute_raw = ocr_result
                    hour_24, minute_24 = convert_12h_to_24h(ocr_hour_raw, ocr_minute_raw, file_hour)

                    stream_state.last_known_hour = hour_24
                    stream_state.last_known_minute = minute_24
                    stream_state.ocr_text = f"{hour_24:02d}:{minute_24:02d}"

                    # Cek apakah menit berubah - RESET counting DAN tracker
                    if stream_state.should_reset_counts(minute_24):
                        print(f"[OCR] Menit berubah: {stream_state.current_minute:02d} -> {minute_24:02d}, RESET counting dan tracker!")
                        stream_state.reset_for_new_minute()

                    stream_state.current_minute = minute_24
                else:
                    # OCR gagal - gunakan last known
                    if stream_state.last_known_minute != -1:
                        stream_state.ocr_text = f"LAST: {stream_state.last_known_hour:02d}:{stream_state.last_known_minute:02d}"

            # Apply detection SETIAP FRAME untuk tracking yang akurat
            if with_detection and yolo_model:
                scaled_roi = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in FIXED_ROI_POINTS], dtype=np.int32)
                last_scaled_roi = scaled_roi

                # Run detection setiap frame - PENTING untuk tracking
                frame, detected_boxes = process_frame_with_detection(frame, yolo_model, scaled_roi, draw_stats=False)
                last_boxes = detected_boxes

                # Update tracker dengan deteksi baru
                # Tracker akan:
                # 1. Match detection ke existing track (IoU atau centroid distance)
                # 2. Hanya menghitung objek yang sudah di-track minimal 3 frame
                # 3. Objek yang sudah dihitung TIDAK akan dihitung lagi (counted=True)
                new_counts = stream_state.tracker.update_tracks(detected_boxes)

                # Tambahkan new_counts ke minute_counts
                for k, v in new_counts.items():
                    if k in stream_state.minute_counts:
                        stream_state.minute_counts[k] += v

                # Draw stats from accumulated minute_counts
                frame = draw_stats_overlay_with_ocr(frame, stream_state.minute_counts, stream_state.ocr_text)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Control playback speed
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()


@app.get("/")
async def root():
    return {"status": "ok", "device": device, "model_loaded": model is not None}


@app.get("/api/videos")
async def list_videos():
    """List available videos"""
    videos = get_video_files()
    return {"success": True, "videos": videos}


@app.get("/api/video/stream/{hour}")
async def stream_video_by_hour(hour: int, detection: bool = True):
    """Stream video for specific hour with YOLO detection"""
    now = datetime.now()
    videos = get_nearest_videos(hour, now.minute)

    if not videos:
        raise HTTPException(status_code=404, detail="No videos found")

    video_path = videos[0]['path']

    return StreamingResponse(
        generate_video_stream(video_path, with_detection=detection),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/api/video/current")
async def stream_current_video(detection: bool = True):
    """Stream video for current hour"""
    now = datetime.now()
    videos = get_nearest_videos(now.hour, now.minute)

    if not videos:
        raise HTTPException(status_code=404, detail="No videos found")

    video_path = videos[0]['path']

    return StreamingResponse(
        generate_video_stream(video_path, with_detection=detection),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.post("/api/predict")
async def predict_video(
    file: UploadFile = File(...),
    date: Optional[str] = None,
    hour: Optional[str] = None
):
    """Process uploaded video with YOLO detection"""
    # Save uploaded file with unique timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
    temp_input = TEMP_DIR / f"input_{timestamp}_{safe_filename}"
    temp_output = TEMP_DIR / f"output_{timestamp}_{safe_filename}"

    cap = None
    out = None

    try:
        # Ensure temp directory exists
        TEMP_DIR.mkdir(exist_ok=True)

        # Save uploaded file
        with open(temp_input, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"[PREDICT] Input saved: {temp_input}")

        # Process video
        cap = cv2.VideoCapture(str(temp_input))
        if not cap.isOpened():
            return JSONResponse({
                "success": False,
                "error": "Cannot open video file"
            }, status_code=400)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[PREDICT] Video: {width}x{height} @ {fps}fps")

        # Use H264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

        if not out.isOpened():
            return JSONResponse({
                "success": False,
                "error": "Cannot create output video writer"
            }, status_code=500)

        roi_polygon = np.array(FIXED_ROI_POINTS, dtype=np.int32) if FIXED_ROI_POINTS else None
        yolo_model = get_model()

        if yolo_model is None:
            print("[PREDICT] WARNING: Model failed to load!")
        else:
            print(f"[PREDICT] Model loaded on {device}")

        total_counts = defaultdict(int)
        frame_count = 0

        # Tracker untuk video predict (terpisah dari stream)
        predict_tracker = VehicleTracker()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every frame with detection if model is available
            if yolo_model:
                frame, detected_boxes = process_frame_with_detection(frame, yolo_model, roi_polygon, draw_stats=True)

                # Update tracker dan dapatkan new counts
                new_counts = predict_tracker.update_tracks(detected_boxes)
                for k, v in new_counts.items():
                    total_counts[k] += v

                # Draw stats on frame
                draw_stats_overlay(frame, total_counts)

            out.write(frame)

        cap.release()
        cap = None
        out.release()
        out = None

        print(f"[PREDICT] Output saved: {temp_output}, exists: {temp_output.exists()}")

        # Clean up old output files (keep only last 5)
        output_files = sorted(TEMP_DIR.glob("output_*.mp4"), key=lambda p: p.stat().st_mtime)
        for old_file in output_files[:-5]:
            try:
                old_file.unlink()
            except:
                pass

        return JSONResponse({
            "success": True,
            "message": "Video processed successfully",
            "total_counts": dict(total_counts),
            "frames_processed": frame_count,
            "model_loaded": yolo_model is not None
        })

    except Exception as e:
        import traceback
        print(f"[PREDICT] Error: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

    finally:
        # Release resources if still open
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        # Cleanup input file only
        if temp_input.exists():
            temp_input.unlink()


@app.get("/api/predict/stream")
async def stream_predicted_video():
    """Stream the last processed video"""
    # Ensure temp directory exists
    TEMP_DIR.mkdir(exist_ok=True)

    output_files = list(TEMP_DIR.glob("output_*.mp4"))
    print(f"[STREAM] Looking for output files in {TEMP_DIR}, found: {len(output_files)}")

    if not output_files:
        # Return a simple error image instead of 404
        return JSONResponse({
            "error": "No processed video found",
            "detail": "Please upload and process a video first"
        }, status_code=404)

    latest = max(output_files, key=lambda p: p.stat().st_mtime)
    print(f"[STREAM] Streaming file: {latest}")

    if not latest.exists():
        return JSONResponse({
            "error": "Output file not found",
            "detail": str(latest)
        }, status_code=404)

    return StreamingResponse(
        generate_video_stream(str(latest), with_detection=False),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
