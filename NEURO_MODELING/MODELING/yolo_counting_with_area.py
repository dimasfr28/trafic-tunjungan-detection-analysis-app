#!/usr/bin/env python3
"""
YOLOv8 Object Counting dengan Area Detection (ROI)
Menggunakan GPU RTX 3050 untuk inference dengan cuDNN acceleration
"""


import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from collections import defaultdict
from datetime import datetime
import csv
import re
import shutil
try:
   import pytesseract
   # Test if tesseract executable is available
   try:
       pytesseract.get_tesseract_version()
       OCR_AVAILABLE = True
   except Exception:
       OCR_AVAILABLE = False
       print("⚠️  Tesseract OCR engine tidak terinstall")
       print("   Install dengan: sudo apt install tesseract-ocr")
except ImportError:
   OCR_AVAILABLE = False
   print("⚠️  pytesseract tidak terinstall, OCR tidak tersedia")


# Enable cuDNN optimizations untuk performa maksimal
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Otomatis pilih algoritma tercepat untuk hardware ini


class ObjectCounterWithROI:
   def __init__(self, model_path, confidence=0.25, iou=0.45, device='cuda', display_width=1280):
       """
       Initialize YOLOv8 Object Counter dengan ROI


       Args:
           model_path: Path ke model YOLOv8 (.pt)
           confidence: Confidence threshold untuk deteksi
           iou: IOU threshold untuk NMS
           device: 'cuda' untuk GPU atau 'cpu'
           display_width: Lebar display window (default 1280, 0 = original size)
       """
       # Check CUDA availability
       if device == 'cuda' and not torch.cuda.is_available():
           print("⚠️  CUDA tidak tersedia, menggunakan CPU")
           device = 'cpu'
       else:
           print(f"✅ Menggunakan device: {device}")
           if device == 'cuda':
               print(f"   GPU: {torch.cuda.get_device_name(0)}")
               print(f"   CUDA: {torch.version.cuda}")
               print(f"   cuDNN: v{torch.backends.cudnn.version()}")
               print(f"   cuDNN Benchmark: {'Enabled' if torch.backends.cudnn.benchmark else 'Disabled'}")


       # Load model
       self.model = YOLO(model_path)
       self.model.to(device)
       self.device = device
       self.confidence = confidence
       self.iou = iou


       # Display settings
       self.display_width = display_width
       self.scale_factor = 1.0


       # ROI (Region of Interest) - akan diset lewat mouse
       self.roi_points = []
       self.roi_polygon = None
       self.drawing_roi = False


       # Counter
       self.object_counts = defaultdict(int)
       self.total_detections = 0


       # Tracking untuk menghindari double counting
       self.tracked_objects = {}  # {track_id: {'class': class_name, 'counted': bool, 'positions': []}}
       self.next_object_id = 0
       self.counted_ids = set()  # Set of IDs yang sudah dihitung


       # Persistent OCR state (tidak direset antar video)
       self.current_minute = -1
       self.last_known_minute = -1
       self.last_known_hour = 0
       self.detection_active = False
       self.minute_counts = {'car': 0, 'motorcycle': 0, 'truck': 0}


   def resize_frame(self, frame):
       """Resize frame untuk display, maintain aspect ratio"""
       if self.display_width == 0:
           return frame, 1.0


       height, width = frame.shape[:2]
       if width <= self.display_width:
           return frame, 1.0


       scale_factor = self.display_width / width
       new_width = self.display_width
       new_height = int(height * scale_factor)


       resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
       return resized, scale_factor


   def set_roi_mouse_callback(self, event, x, y, flags, param):
       """Callback untuk menggambar ROI dengan mouse"""
       if event == cv2.EVENT_LBUTTONDOWN:
           # Scale coordinates back to original size
           original_x = int(x / self.scale_factor)
           original_y = int(y / self.scale_factor)
           self.roi_points.append((original_x, original_y))
           print(f"Point {len(self.roi_points)}: ({original_x}, {original_y})")


       elif event == cv2.EVENT_RBUTTONDOWN:
           if len(self.roi_points) >= 3:
               self.roi_polygon = np.array(self.roi_points, dtype=np.int32)
               self.drawing_roi = False
               print(f"✅ ROI selesai dibuat dengan {len(self.roi_points)} titik")
           else:
               print("⚠️  Minimal 3 titik untuk membuat ROI")


   def draw_roi_on_frame(self, frame, scale_factor=1.0):
       """Gambar ROI di frame dengan scale factor"""
       if len(self.roi_points) > 0:
           # Gambar titik-titik dengan scaling
           for point in self.roi_points:
               scaled_point = (int(point[0] * scale_factor), int(point[1] * scale_factor))
               cv2.circle(frame, scaled_point, 5, (0, 255, 255), -1)


           # Gambar garis antar titik
           for i in range(len(self.roi_points) - 1):
               pt1 = (int(self.roi_points[i][0] * scale_factor), int(self.roi_points[i][1] * scale_factor))
               pt2 = (int(self.roi_points[i+1][0] * scale_factor), int(self.roi_points[i+1][1] * scale_factor))
               cv2.line(frame, pt1, pt2, (0, 255, 255), 2)


           # Jika sudah lebih dari 2 titik, hubungkan titik terakhir ke pertama
           if len(self.roi_points) > 2:
               pt1 = (int(self.roi_points[-1][0] * scale_factor), int(self.roi_points[-1][1] * scale_factor))
               pt2 = (int(self.roi_points[0][0] * scale_factor), int(self.roi_points[0][1] * scale_factor))
               cv2.line(frame, pt1, pt2, (0, 255, 255), 2)


       # Jika ROI sudah jadi, gambar polygon transparan
       if self.roi_polygon is not None:
           overlay = frame.copy()
           scaled_polygon = np.array([[int(p[0] * scale_factor), int(p[1] * scale_factor)]
                                     for p in self.roi_polygon], dtype=np.int32)
           cv2.fillPoly(overlay, [scaled_polygon], (0, 255, 0))
           cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
           cv2.polylines(frame, [scaled_polygon], True, (0, 255, 0), 2)


       return frame


   def is_point_in_roi(self, point):
       """Cek apakah titik berada di dalam ROI"""
       if self.roi_polygon is None:
           return True  # Jika tidak ada ROI, semua terdeteksi


       result = cv2.pointPolygonTest(self.roi_polygon, point, False)
       return result >= 0


   def setup_roi(self, video_path):
       """Setup ROI dengan menggunakan frame pertama video"""
       cap = cv2.VideoCapture(video_path)
       ret, frame = cap.read()
       cap.release()


       if not ret:
           print("❌ Gagal membaca video")
           return False


       # Resize frame untuk display
       display_frame_orig, self.scale_factor = self.resize_frame(frame)


       print("\n📐 SETUP ROI (Region of Interest)")
       print("=" * 50)
       print(f"Original size: {frame.shape[1]}x{frame.shape[0]}")
       print(f"Display size: {display_frame_orig.shape[1]}x{display_frame_orig.shape[0]}")
       print(f"Scale factor: {self.scale_factor:.2f}")
       print("=" * 50)
       print("Instruksi:")
       print("1. Klik kiri (LEFT CLICK) untuk menambah titik ROI")
       print("2. Klik kanan (RIGHT CLICK) setelah selesai menggambar area")
       print("3. Tekan 's' untuk skip ROI (deteksi seluruh area)")
       print("4. Tekan 'q' untuk keluar")
       print("=" * 50)


       self.drawing_roi = True
       clone = display_frame_orig.copy()


       cv2.namedWindow('Setup ROI')
       cv2.setMouseCallback('Setup ROI', self.set_roi_mouse_callback)


       while self.drawing_roi:
           display_frame = clone.copy()
           display_frame = self.draw_roi_on_frame(display_frame, self.scale_factor)


           # Tampilkan instruksi di frame
           cv2.putText(display_frame, "LEFT CLICK: Tambah titik | RIGHT CLICK: Selesai | 'S': Skip | 'Q': Quit",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
           cv2.putText(display_frame, f"Titik: {len(self.roi_points)}",
                      (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


           cv2.imshow('Setup ROI', display_frame)


           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
               cv2.destroyAllWindows()
               return False
           elif key == ord('s'):
               print("⏭️  ROI diskip, akan mendeteksi seluruh area")
               self.roi_polygon = None
               break


       cv2.destroyAllWindows()
       return True


   def get_center_point(self, box):
       """Dapatkan titik tengah dari bounding box"""
       x1, y1, x2, y2 = box
       center_x = int((x1 + x2) / 2)
       center_y = int((y1 + y2) / 2)
       return (center_x, center_y)


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


   def match_detection_to_track(self, detection_box, detection_class):
       """Match detection ke existing track berdasarkan IoU dan class"""
       best_match_id = None
       best_iou = 0.3  # Minimum IoU threshold


       for track_id, track_data in self.tracked_objects.items():
           if track_data['class'] != detection_class:
               continue


           # Get last position dari track
           if len(track_data['positions']) > 0:
               last_box = track_data['positions'][-1]
               iou = self.calculate_iou(detection_box, last_box)


               if iou > best_iou:
                   best_iou = iou
                   best_match_id = track_id


       return best_match_id


   def update_tracks(self, current_detections):
       """Update tracking objects dan hitung objek baru yang masuk ROI"""
       # Tandai semua tracks sebagai not updated
       for track_id in self.tracked_objects:
           self.tracked_objects[track_id]['updated'] = False


       new_counts = defaultdict(int)


       # Update atau create new tracks
       for detection in current_detections:
           box, class_name, center = detection


           # Cari matching track
           matched_id = self.match_detection_to_track(box, class_name)


           if matched_id is not None:
               # Update existing track
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
                   new_counts[class_name] += 1
                   print(f"   ✅ COUNTED: ID={matched_id} Class='{class_name}'")


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
               if track['frames_missing'] > 30:  # Lost for 30 frames (1 second at 30fps)
                   tracks_to_remove.append(track_id)


       for track_id in tracks_to_remove:
           del self.tracked_objects[track_id]


       return new_counts


   def get_class_color(self, class_name, is_counted=False):
       """
       Get bounding box color based on vehicle class
       - Car: Biru (Blue)
       - Motorcycle: Hijau (Green)
       - Heavy (Truck/Bus): Merah (Red)


       Returns brighter color if object is counted
       """
       class_lower = class_name.lower()


       # Determine base color by class
       if 'car' in class_lower or 'vehicle' in class_lower:
           # Car = Blue
           base_color = (255, 0, 0)  # BGR: Blue
           counted_color = (255, 100, 100)  # Light blue
       elif 'motor' in class_lower or 'bike' in class_lower:
           # Motorcycle = Green
           base_color = (0, 255, 0)  # BGR: Green
           counted_color = (100, 255, 100)  # Light green
       elif 'truck' in class_lower or 'bus' in class_lower or 'heavy' in class_lower:
           # Heavy = Red
           base_color = (0, 0, 255)  # BGR: Red
           counted_color = (100, 100, 255)  # Light red
       else:
           # Default = Orange
           base_color = (0, 165, 255)  # BGR: Orange
           counted_color = (100, 200, 255)  # Light orange


       return counted_color if is_counted else base_color


   def get_dynamic_ocr_roi(self, current_hour=None):
       """
       Get dynamic OCR ROI based on expected hour format
       - Jam >= 10 (2 digit): ROI width 125 (contoh: "10:00", "11:00", "12:00")
       - Jam < 10 (1 digit): ROI width 100 (contoh: "1:00", "2:00", "9:00")
       """
       # Gunakan last known hour jika ada, atau default ke 2-digit
       hour = current_hour if current_hour is not None else self.last_known_hour


       # Convert ke 12-hour format untuk check digit
       hour_12 = hour % 12
       if hour_12 == 0:
           hour_12 = 12


       # 2-digit hour (10, 11, 12) → width 125
       # 1-digit hour (1-9) → width 100
       if hour_12 >= 10:
           return (0, 0, 125, 60)
       else:
           return (0, 0, 100, 60)


   def extract_time_from_frame(self, frame, roi=None):
       """Extract time dari frame menggunakan OCR pada area tertentu"""
       if not OCR_AVAILABLE:
           return None, "OCR not installed"


       # Jika ROI tidak diberikan, gunakan dynamic ROI
       if roi is None:
           roi = self.get_dynamic_ocr_roi()


       try:
           x1, y1, x2, y2 = roi
           time_region = frame[y1:y2, x1:x2]


           # ENHANCED PREPROCESSING untuk OCR yang lebih robust
           gray = cv2.cvtColor(time_region, cv2.COLOR_BGR2GRAY)


           # Upscale image untuk OCR lebih baik (2x)
           gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


           # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
           clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
           enhanced = clahe.apply(gray)


           # Denoise
           denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)


           # Try multiple threshold methods untuk lebih robust
           methods = [
               ('BINARY', cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)[1]),
               ('BINARY_INV', cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY_INV)[1]),
               ('OTSU', cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
               ('ADAPTIVE', cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
           ]


           # Try OCR with each method until we get a valid result
           best_result = None
           for method_name, thresh in methods:
               # OCR dengan config untuk single line text
               text = pytesseract.image_to_string(thresh, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:.')


               # Debug: print raw OCR text
               if text.strip():
                   print(f"   [DEBUG OCR {method_name}] Raw text: '{text.strip()}'")


               # Parse time format (HH:MM atau H:MM atau HH.MM)
               time_match = re.search(r'(\d{1,2})[:\.\s](\d{2})', text)
               if time_match:
                   hour = int(time_match.group(1))
                   minute = int(time_match.group(2))


                   # Validate result (jam 1-12, menit 0-59)
                   if 1 <= hour <= 12 and 0 <= minute <= 59:
                       print(f"   ✅ OCR SUCCESS with {method_name}: {hour}:{minute:02d}")
                       return (hour, minute), None


               # Jika belum dapat hasil, simpan untuk fallback
               if best_result is None and text.strip():
                   best_result = text.strip()


           # Jika semua method gagal
           return None, f"No match: {best_result[:20] if best_result else 'empty'}"
       except Exception as e:
           return None, f"Error: {str(e)[:30]}"


       return None, "Unknown error"


   def parse_datetime_from_filename(self, filename):
       """Parse datetime dari nama file: screen_recording_20251104_140237_seg2.mp4"""
       # Pattern: YYYYMMDD_HHMMSS
       match = re.search(r'(\d{8})_(\d{6})', filename)
       if match:
           date_str = match.group(1)  # YYYYMMDD
           time_str = match.group(2)  # HHMMSS


           year = int(date_str[0:4])
           month = int(date_str[4:6])
           day = int(date_str[6:8])
           hour = int(time_str[0:2])
           minute = int(time_str[2:4])
           second = int(time_str[4:6])


           return datetime(year, month, day, hour, minute, second)
       return None


   def convert_12h_to_24h(self, ocr_hour, ocr_minute, filename_datetime):
       """Convert 12-hour format ke 24-hour berdasarkan filename"""
       if filename_datetime is None:
           return ocr_hour, ocr_minute


       file_hour = filename_datetime.hour


       # Jika file hour >= 12 dan OCR hour < 12, tambahkan 12
       if file_hour >= 12 and ocr_hour < 12:
           # Kecuali jika ocr_hour adalah 12 (12 PM = 12, 12 AM = 0)
           if ocr_hour == 12:
               return ocr_hour, ocr_minute
           return ocr_hour + 12, ocr_minute


       # Jika file hour < 12 dan OCR hour adalah 12
       if file_hour < 12 and ocr_hour == 12:
           return 0, ocr_minute


       return ocr_hour, ocr_minute


   def should_detect(self, minute):
       """Cek apakah deteksi aktif berdasarkan menit (00-10 saja)"""
       return 0 <= minute <= 10


   def save_to_csv(self, csv_path, datetime_str, count_car, count_motorcycle, count_heavy, ocr_raw=""):
       """Append data ke CSV file dengan OCR raw untuk validasi"""
       file_exists = Path(csv_path).exists()


       with open(csv_path, 'a', newline='') as csvfile:
           fieldnames = ['datetime', 'ocr_raw', 'count_of_car', 'count_of_motorcycle', 'count_of_heavy']
           writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


           # Write header jika file baru
           if not file_exists:
               writer.writeheader()


           # Write data
           writer.writerow({
               'datetime': datetime_str,
               'ocr_raw': ocr_raw,
               'count_of_car': count_car,
               'count_of_motorcycle': count_motorcycle,
               'count_of_heavy': count_heavy
           })


   def process_video(self, video_path, output_dir=None, show_live=True, fixed_roi_points=None, csv_output_path=None):
       """
       Process video dengan YOLOv8 dan counting


       Args:
           video_path: Path ke video input
           output_dir: Direktori untuk memindahkan video yang sudah diproses
           show_live: Tampilkan preview real-time
           fixed_roi_points: List of ROI points [(x1,y1), (x2,y2), ...] untuk skip manual drawing
           csv_output_path: Path untuk menyimpan hasil counting ke CSV
       """
       # Setup ROI dulu
       if fixed_roi_points is not None:
           # Gunakan ROI yang sudah fix
           print("\n✅ Menggunakan Fixed ROI Points")
           print(f"   Total points: {len(fixed_roi_points)}")
           for i, point in enumerate(fixed_roi_points, 1):
               print(f"   Point {i}: {point}")


           self.roi_points = list(fixed_roi_points)
           self.roi_polygon = np.array(self.roi_points, dtype=np.int32)
           print("✅ ROI berhasil diset!\n")
       else:
           # Gambar ROI secara manual
           if not self.setup_roi(video_path):
               print("❌ Setup ROI dibatalkan")
               return


       # Open video
       cap = cv2.VideoCapture(video_path)


       if not cap.isOpened():
           print(f"❌ Gagal membuka video: {video_path}")
           return


       # Video properties
       fps = int(cap.get(cv2.CAP_PROP_FPS))
       width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


       print(f"\n🎥 Processing Video: {Path(video_path).name}")
       print(f"   Resolusi: {width}x{height}")
       print(f"   FPS: {fps}")
       print(f"   Total Frames: {total_frames}")
       print(f"   Device: {self.device.upper()}")


       frame_count = 0


       # Reset counters dan tracking
       self.object_counts.clear()
       self.total_detections = 0
       self.tracked_objects.clear()
       self.counted_ids.clear()
       self.next_object_id = 0


       # Parse filename datetime
       filename_datetime = self.parse_datetime_from_filename(Path(video_path).name)
       if filename_datetime:
           print(f"   Filename datetime: {filename_datetime.strftime('%Y/%m/%d %H:%M:%S')}")


       # OCR raw text (per-video variable)
       ocr_raw_text = "Waiting..."


       # PRE-CHECK: Cek menit di awal video untuk skip jika di luar rentang 55-10
       print("\n🔍 Pre-checking video minute range...")
       ret, first_frame = cap.read()
       if ret:
           # Gunakan ROI berdasarkan filename hour jika ada
           initial_hour = filename_datetime.hour if filename_datetime else None
           ocr_roi = self.get_dynamic_ocr_roi(initial_hour)
           print(f"   🔍 Using dynamic OCR ROI: {ocr_roi} (based on filename hour: {initial_hour})")
           ocr_result, error_msg = self.extract_time_from_frame(first_frame, roi=ocr_roi)
           if ocr_result:
               ocr_hour_raw, ocr_minute_raw = ocr_result
               hour_24, minute_24 = self.convert_12h_to_24h(ocr_hour_raw, ocr_minute_raw, filename_datetime)


               # Cek apakah di rentang 55-10 (55-59 atau 00-10)
               in_valid_range = (minute_24 >= 55) or (minute_24 <= 10)


               if not in_valid_range:
                   print(f"   ⏭️  Video dimulai di menit {minute_24:02d} (di luar rentang 55-10)")
                   print(f"   ⏭️  SKIPPING video ini untuk efisiensi waktu...")
                   cap.release()
                   return
               else:
                   print(f"   ✅ Video dimulai di menit {minute_24:02d} (dalam rentang 55-10)")
           else:
               print(f"   ⚠️  Pre-check OCR gagal: {error_msg}")
               print(f"   ➡️  Melanjutkan processing (asumsi valid)...")


           # Reset video ke awal
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


       print("\n⏳ Memproses video...")
       print("💡 Mode: Full OCR (deteksi aktif menit 00-10, pause menit 11-59)")
       print("   📌 State persistent across videos (tidak reset)")
       print("Tekan 'q' untuk berhenti\n")


       while True:
           ret, frame = cap.read()
           if not ret:
               break


           frame_count += 1


           # FULL OCR MODE: Jalan terus setiap 30 frames (1 detik)
           if frame_count % 30 == 1:
               # Gunakan dynamic ROI (otomatis adjust berdasarkan last known hour)
               ocr_result, error_msg = self.extract_time_from_frame(frame)
               if ocr_result:
                   # OCR SUCCESS
                   ocr_hour_raw, ocr_minute_raw = ocr_result
                   ocr_raw_text = f"{ocr_hour_raw:02d}:{ocr_minute_raw:02d}"
                   hour_24, minute_24 = self.convert_12h_to_24h(ocr_hour_raw, ocr_minute_raw, filename_datetime)


                   # Simpan sebagai last known value (PERSISTENT)
                   self.last_known_hour = hour_24
                   self.last_known_minute = minute_24


                   # Cek apakah menit berubah
                   if minute_24 != self.current_minute:
                       # Menit berubah
                       if self.current_minute != -1 and self.current_minute <= 10 and csv_output_path:
                           # Save data menit sebelumnya ke CSV (hanya jika masih di 00-10)
                           if filename_datetime:
                               dt_str = f"{filename_datetime.year}/{filename_datetime.month:02d}/{filename_datetime.day:02d} {self.last_known_hour:02d}:{self.current_minute:02d}"
                               self.save_to_csv(
                                   csv_output_path,
                                   dt_str,
                                   self.minute_counts.get('car', 0),
                                   self.minute_counts.get('motorcycle', 0),
                                   self.minute_counts.get('truck', 0),
                                   ocr_raw_text
                               )
                               print(f"   💾 Saved to CSV: {dt_str} (OCR:{ocr_raw_text}) - Car:{self.minute_counts.get('car', 0)} Moto:{self.minute_counts.get('motorcycle', 0)} Truck:{self.minute_counts.get('truck', 0)}")


                       # Update current minute (PERSISTENT)
                       self.current_minute = minute_24


                       # Reset counts untuk menit baru (hanya jika menit 00)
                       if minute_24 == 0:
                           self.minute_counts = {'car': 0, 'motorcycle': 0, 'truck': 0}


                       # SKIP VIDEO saat menit > 10 (efisiensi waktu)
                       if minute_24 > 10 and minute_24 < 55:
                           print(f"   ⏭️  MENIT {minute_24:02d} DETECTED (>10 dan <55)! Skipping to next video...")
                           break  # Skip video ini, lanjut ke video berikutnya


                       # STOP RECORDING saat menit 10 terdeteksi
                       if minute_24 == 10:
                           print(f"   🛑 MENIT 10 DETECTED! Stopping recording for this video...")
                           print(f"   💾 Final save: {dt_str if 'dt_str' in locals() else 'N/A'}")
                           break  # Keluar dari loop video, lanjut ke video berikutnya


                       # Cek apakah deteksi aktif (00-10) atau pause (11-59)
                       self.detection_active = self.should_detect(minute_24)
                       if self.detection_active:
                           print(f"   🟢 Detection ACTIVE - Minute: {minute_24:02d}")
                       else:
                           print(f"   🔴 Detection PAUSED - Minute: {minute_24:02d} (deteksi hanya di menit 00-10)")
               else:
                   # OCR FAILED - Gunakan nilai terakhir yang diketahui
                   if self.last_known_minute != -1:
                       # Gunakan last known value
                       self.current_minute = self.last_known_minute
                       ocr_raw_text = f"LAST: {self.last_known_hour:02d}:{self.last_known_minute:02d}"
                       if frame_count % 300 == 1:
                           print(f"   ⚠️  OCR Failed, using last known: {self.last_known_hour:02d}:{self.last_known_minute:02d}")
                   else:
                       # Belum pernah ada OCR yang sukses
                       ocr_raw_text = f"FAIL: {error_msg}"
                       if frame_count % 300 == 1:
                           print(f"   ⚠️  OCR Failed: {error_msg}")


           # Initialize
           current_detections = []


           # Skip detection jika tidak dalam rentang menit 00-10
           if self.detection_active or self.current_minute == -1:
               # YOLOv8 inference dengan GPU
               results = self.model.predict(
                   frame,
                   conf=self.confidence,
                   iou=self.iou,
                   device=self.device,
                   verbose=False
               )


               # Collect current frame detections yang ada di dalam ROI
               for result in results:
                   boxes = result.boxes


                   for box in boxes:
                       # Get bounding box coordinates
                       x1, y1, x2, y2 = map(int, box.xyxy[0])


                       # Get center point
                       center = self.get_center_point((x1, y1, x2, y2))


                       # Cek apakah center point ada di dalam ROI
                       if not self.is_point_in_roi(center):
                           continue  # Skip jika di luar ROI


                       # Get class name and confidence
                       cls = int(box.cls[0])
                       conf = float(box.conf[0])
                       class_name = self.model.names[cls]


                       # Simpan detection
                       current_detections.append(((x1, y1, x2, y2), class_name, center))


               # Update tracking dan dapatkan count baru
               new_counts = self.update_tracks(current_detections)


               # Update total counts dan minute counts
               for class_name, count in new_counts.items():
                   self.object_counts[class_name] += count
                   self.total_detections += count


                   # Update minute counts untuk CSV
                   if class_name.lower() in self.minute_counts:
                       self.minute_counts[class_name.lower()] += count
                   elif 'car' in class_name.lower() or 'vehicle' in class_name.lower():
                       self.minute_counts['car'] += count
                   elif 'motor' in class_name.lower() or 'bike' in class_name.lower():
                       self.minute_counts['motorcycle'] += count
                   elif 'truck' in class_name.lower() or 'bus' in class_name.lower() or 'heavy' in class_name.lower():
                       self.minute_counts['truck'] += count
                   else:
                       # Log jika ada class yang tidak dikenali
                       print(f"   ⚠️  Class tidak dikenali untuk CSV: '{class_name}'")


           # Draw detections dengan tracking ID
           for detection in current_detections:
               box, class_name, center = detection
               x1, y1, x2, y2 = box


               # Find track ID untuk detection ini
               track_id = None
               for tid, track in self.tracked_objects.items():
                   if (track['class'] == class_name and
                       track['last_center'] == center):
                       track_id = tid
                       break


               # Warna berdasarkan class dan status counted
               is_counted = track_id in self.counted_ids
               color = self.get_class_color(class_name, is_counted)


               # Draw bounding box
               cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


               # Draw center point
               cv2.circle(frame, center, 4, (0, 0, 255), -1)


               # Label dengan track ID
               if track_id is not None:
                   label = f'ID:{track_id} {class_name}'
               else:
                   label = f'{class_name}'


               (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
               cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
               cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


           # Draw ROI (at original scale)
           frame = self.draw_roi_on_frame(frame, scale_factor=1.0)


           # Draw OCR area indicator (kiri atas) - Dynamic ROI based on current hour
           ocr_roi = self.get_dynamic_ocr_roi()
           ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_roi
           cv2.rectangle(frame, (ocr_x1, ocr_y1), (ocr_x2, ocr_y2), (255, 0, 255), 2)  # Magenta box
           # Tampilkan OCR raw result dibawah box dengan info ROI
           cv2.putText(frame, f"OCR: {ocr_raw_text} (ROI:{ocr_x2}x{ocr_y2})", (ocr_x1 + 5, ocr_y2 + 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


           # Draw statistics - PINDAH KE KANAN ATAS
           # Hitung posisi X dari kanan
           stats_width = 450
           stats_x = width - stats_width - 10  # 10px dari kanan


           y_offset = 30
           active_tracks = len([t for t in self.tracked_objects.values() if t['updated']])
           lines_count = len(self.object_counts) + 6  # Extra lines untuk info tambahan


           # Background box di kanan atas
           cv2.rectangle(frame, (stats_x, 10), (width - 10, 30 + lines_count * 25), (0, 0, 0), -1)


           # Semua text dimulai dari stats_x + 10
           text_x = stats_x + 10


           cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (text_x, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
           y_offset += 25


           # OCR Raw (untuk validasi)
           cv2.putText(frame, f"OCR Raw: {ocr_raw_text}", (text_x, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
           y_offset += 22


           # Current time dan detection status
           if self.current_minute != -1:
               time_str = f"Time: {self.last_known_hour:02d}:{self.current_minute:02d}"
               status_color = (0, 255, 0) if self.detection_active else (0, 0, 255)
               status_text = "ACTIVE" if self.detection_active else "PAUSED"
               cv2.putText(frame, f"{time_str} [{status_text}]", (text_x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
           else:
               cv2.putText(frame, "Time: Reading OCR...", (text_x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
           y_offset += 25


           cv2.putText(frame, f"Active Tracks: {active_tracks}", (text_x, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
           y_offset += 25


           cv2.putText(frame, f"Total Counted: {self.total_detections}", (text_x, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           y_offset += 25


           # Minute counts (untuk CSV)
           cv2.putText(frame, f"This Minute:", (text_x, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
           y_offset += 20
           cv2.putText(frame, f"Car:{self.minute_counts['car']} Moto:{self.minute_counts['motorcycle']} Truck:{self.minute_counts['truck']}",
                      (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
           y_offset += 25


           for class_name, count in sorted(self.object_counts.items()):
               cv2.putText(frame, f"  {class_name}: {count}", (text_x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
               y_offset += 25


           # Video output tidak dibuat lagi (hanya tracking)


           # Show live (resized for display)
           if show_live:
               display_frame, _ = self.resize_frame(frame)
               cv2.imshow('YOLOv8 Counting with ROI', display_frame)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   print("\n⏹️  Dihentikan oleh user")
                   break


           # Progress
           if frame_count % 30 == 0:
               progress = (frame_count / total_frames) * 100
               print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')


       # Cleanup
       cap.release()
       cv2.destroyAllWindows()

       # Pindahkan video yang sudah diproses ke output directory
       if output_dir:
           try:
               output_dir_path = Path(output_dir)
               output_dir_path.mkdir(exist_ok=True)

               video_filename = Path(video_path).name
               destination = output_dir_path / video_filename

               shutil.move(str(video_path), str(destination))
               print(f"\n✅ Video dipindahkan ke: {destination}")
           except Exception as e:
               print(f"\n⚠️  Gagal memindahkan video: {e}")


       # Print summary
       print("\n\n" + "=" * 60)
       print("📊 HASIL COUNTING")
       print("=" * 60)
       print(f"Total Detections (in ROI): {self.total_detections}")
       print(f"\nBreakdown per Class:")
       for class_name, count in sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True):
           print(f"  • {class_name}: {count}")
       print("=" * 60)
       print("💾 Hasil disimpan ke CSV (tidak ada JSON/MP4 output)\n")




def main():
   """Main function"""
   # Configuration
   MODEL_PATH = "/home/dimas/neurocomputing/model/YOLOv8s/train/weights/best.pt"
   VIDEO_DIR = "/home/dimas/neurocomputing/assets/video/input_vidio"
   OUTPUT_DIR = "/home/dimas/neurocomputing/assets/video/output_counting"


   # Fixed ROI Points - Set None untuk menggambar manual
   # Format: [(x1, y1), (x2, y2), (x3, y3), ...]
   FIXED_ROI_POINTS = [
       (212, 1022),
       (858, 1085),
       (894, 1369),
       (70, 1296),
       (55, 1274)
   ]
   # Set ke None jika ingin menggambar ROI secara manual setiap video
   # FIXED_ROI_POINTS = None


   # Create output directory
   Path(OUTPUT_DIR).mkdir(exist_ok=True)


   # CSV output path
   CSV_OUTPUT_PATH = "/home/dimas/neurocomputing/assets/file/vehicle_counts.csv"


   # Get all video files
   video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
   video_files = []
   for ext in video_extensions:
       video_files.extend(Path(VIDEO_DIR).glob(ext))


   video_files = sorted([str(v) for v in video_files])


   if not video_files:
       print("❌ Tidak ada video ditemukan!")
       return


   print("\n" + "=" * 60)
   print("🚀 YOLOv8 Object Counting dengan Area Detection (ROI)")
   print("=" * 60)
   print(f"Model: {Path(MODEL_PATH).name}")
   print(f"Video ditemukan: {len(video_files)}")
   if FIXED_ROI_POINTS:
       print(f"ROI Mode: Fixed ROI ({len(FIXED_ROI_POINTS)} points)")
   else:
       print("ROI Mode: Manual Drawing")
   print("=" * 60)


   # Initialize counter
   counter = ObjectCounterWithROI(
       model_path=MODEL_PATH,
       confidence=0.25,
       iou=0.45,
       device='cuda',  # Menggunakan GPU RTX 3050
       display_width=600  # Resize display ke 600px (ubah ke 800, 960, 1280 jika ingin lebih besar)
   )


   # Process each video
   for i, video_path in enumerate(video_files, 1):
       print(f"\n[{i}/{len(video_files)}] Processing: {Path(video_path).name}")


       try:
           counter.process_video(
               video_path=video_path,
               output_dir=OUTPUT_DIR,  # Direktori untuk memindahkan video yang sudah diproses
               show_live=True,
               fixed_roi_points=FIXED_ROI_POINTS,  # Gunakan Fixed ROI
               csv_output_path=CSV_OUTPUT_PATH  # CSV output
           )
       except Exception as e:
           print(f"❌ Error processing {Path(video_path).name}: {e}")
           import traceback
           traceback.print_exc()
           continue


       # Auto-lanjut ke video berikutnya (no confirmation)
       print(f"\n➡️  Auto-continue to next video...")
       # ROI sudah fixed, tidak perlu reset


   print("\n✅ Semua video selesai diproses!")
   print(f"📁 Output disimpan di: {OUTPUT_DIR}")
if __name__ == "__main__":
   main()