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
    """Process single frame with YOLO detection"""
    counts = defaultdict(int)
    detected_boxes = []  # Store box data for caching

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

            # Count vehicles
            if 'car' in class_name.lower():
                counts['car'] += 1
            elif 'motor' in class_name.lower() or 'bike' in class_name.lower():
                counts['motorcycle'] += 1
            elif 'truck' in class_name.lower() or 'bus' in class_name.lower():
                counts['heavy'] += 1

            # Store box data for caching
            detected_boxes.append({
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

    # Draw stats only if requested (for backward compatibility)
    if draw_stats:
        y_offset = 30
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Car: {counts['car']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(frame, f"Motorcycle: {counts['motorcycle']}", (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, f"Heavy: {counts['heavy']}", (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        cv2.putText(frame, f"Total: {sum(counts.values())}", (20, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, counts, detected_boxes


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


def generate_video_stream(video_path: str, with_detection: bool = True):
    """Generate video stream with optional YOLO detection"""
    import time

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    # Get original FPS for proper playback speed
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25
    frame_delay = 1.0 / original_fps  # Delay between frames

    roi_polygon = np.array(FIXED_ROI_POINTS, dtype=np.int32) if FIXED_ROI_POINTS else None
    yolo_model = get_model() if with_detection else None

    frame_skip = 2  # Process every 2nd frame for better smoothness
    frame_count = 0

    # Cache last detection results to prevent flickering
    last_counts = {'car': 0, 'motorcycle': 0, 'heavy': 0}
    last_boxes = []  # Cache detected boxes

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # Resize frame for streaming
            height, width = frame.shape[:2]
            scale = 720 / width if width > 720 else 1
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

            # Apply detection every N frames, but always draw cached results
            if with_detection and yolo_model:
                scaled_roi = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in FIXED_ROI_POINTS], dtype=np.int32)

                if frame_count % frame_skip == 0:
                    # Run detection and update cached counts and boxes
                    frame, counts, detected_boxes = process_frame_with_detection(frame, yolo_model, scaled_roi, draw_stats=False)
                    last_counts = dict(counts)
                    last_boxes = detected_boxes
                else:
                    # Draw cached boxes on non-detection frames
                    frame = draw_detection_boxes(frame, last_boxes)

                    # Draw ROI
                    if scaled_roi is not None:
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [scaled_roi], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                        cv2.polylines(frame, [scaled_roi], True, (0, 255, 0), 2)

                # Always draw stats from cached counts (prevents flickering)
                frame = draw_stats_overlay(frame, last_counts)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Control playback speed - wait to match original FPS
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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every frame with detection if model is available
            if yolo_model:
                frame, counts, _ = process_frame_with_detection(frame, yolo_model, roi_polygon)
                for k, v in counts.items():
                    total_counts[k] += v

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
    uvicorn.run(app, host="0.0.0.0", port=8001)
