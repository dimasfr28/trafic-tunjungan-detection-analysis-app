# NeuroTraffic: Traffic Analysis System 🚦

<div align="center">

![Traffic Analysis](https://img.shields.io/badge/Traffic-Analysis-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Sistem Analisis Lalu Lintas Berbasis AI untuk Jl. Tunjungan Surabaya**

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Contributors](#-contributors)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [API Documentation](#-api-documentation)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Overview

**NeuroTraffic** adalah sistem analisis lalu lintas real-time yang menggunakan teknologi **Deep Learning** dan **Computer Vision** untuk mendeteksi, menghitung, dan memprediksi kepadatan lalu lintas di Jl. Tunjungan, Surabaya.

Sistem ini menggunakan:
- **YOLOv8** untuk deteksi kendaraan real-time
- **Deep Neural Network (DNN)** untuk klasifikasi kepadatan lalu lintas
- **Vehicle Tracking** dengan IoU-based matching
- **OCR (Tesseract)** untuk ekstraksi timestamp dari video

### 🎥 Demo

Sistem dapat:
- ✅ Mendeteksi 3 jenis kendaraan: **Car**, **Motorcycle**, **Heavy Vehicle**
- ✅ Tracking kendaraan dengan unique ID untuk menghindari double counting
- ✅ Prediksi kepadatan: **Lancar** (Class 1), **Padat** (Class 2), **Macet** (Class 3)
- ✅ Visualisasi data real-time dengan charts interaktif
- ✅ Analisis per menit dengan history 60 menit terakhir

---

## ✨ Features

### 🚗 Vehicle Detection & Tracking
- **YOLOv8-based detection** dengan 3 class: `car`, `motorcycle`, `heavy`
- **Real-time tracking** menggunakan IoU (Intersection over Union)
- **ROI (Region of Interest)** filtering untuk fokus pada area tertentu
- **Unique counting** - setiap kendaraan hanya dihitung sekali

### 📊 Traffic Analysis
- **Per-minute statistics** dengan automatic reset setiap menit berganti
- **DNN-based prediction** untuk klasifikasi kepadatan (Lancar/Padat/Macet)
- **Multi-video streaming** dari berbagai jam secara otomatis
- **Interactive charts** menggunakan Chart.js:
  - Line Chart: Traffic density per jam
  - Pie Chart: Cluster distribution
  - Bar Chart: Total vehicles per day
  - Heatmap: Traffic intensity (Hari x Jam)

### 🎯 Prediction System
- Upload video untuk analisis
- Deteksi YOLO real-time dengan output video
- Prediksi kepadatan menggunakan model DNN
- Export hasil per menit

### 🌐 Web Interface
- **Home Page**: Live video stream dengan real-time counting dan tabel data per menit
- **Traffic Analysis**: Visualisasi data historis dengan filter tanggal dan hari
- **Prediction**: Upload dan analisis video custom

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT (Browser)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐          │
│  │   Home   │  │ Analysis │  │   Prediction     │          │
│  └──────────┘  └──────────┘  └──────────────────┘          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────┴────────────────────────────────────────┐
│                  BACKEND SERVICES                            │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │   Django (Port 8001) │    │  FastAPI (Port 8002)    │   │
│  │  - Web UI            │◄───┤  - Video Streaming      │   │
│  │  - Templates         │    │  - YOLO Detection       │   │
│  │  - Static Files      │    │  - DNN Prediction       │   │
│  │  - Data Aggregation  │    │  - Vehicle Tracking     │   │
│  └──────────────────────┘    └─────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  YOLOv8     │  │  DNN Model   │  │  Video Files     │   │
│  │  Model      │  │  (SGD)       │  │  (.mp4)          │   │
│  │  (.pt)      │  │  (.pkl)      │  │                  │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐                         │
│  │  CSV Data   │  │  Tesseract   │                         │
│  │  (results)  │  │  OCR Engine  │                         │
│  └─────────────┘  └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### Component Flow

```
Video Input → YOLO Detection → ROI Filtering → Vehicle Tracking
                                                      ↓
                                              Unique Counting
                                                      ↓
                                              OCR Timestamp
                                                      ↓
                                              Per-Minute Stats
                                                      ↓
                                              DNN Prediction
                                                      ↓
                                              Database/Display
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** (v0.104.1) - High-performance async API framework
- **Django** (v4.2) - Web framework untuk UI
- **PyTorch** (v2.1.0) - Deep learning framework
- **Ultralytics YOLOv8** - Object detection
- **OpenCV** (cv2) - Video processing
- **Tesseract OCR** - Time extraction dari video
- **NumPy** & **Pandas** - Data processing

### Frontend
- **HTML5** & **CSS3** - Modern responsive design
- **JavaScript (ES6+)** - Interactive features
- **Bootstrap 5.3** - UI components
- **Chart.js** - Data visualization

### Infrastructure
- **Docker** & **Docker Compose** - Containerization
- **CUDA** - GPU acceleration (optional)
- **NVIDIA Container Toolkit** - GPU support in Docker

### Models
- **YOLOv8s Custom** - Trained untuk 3 class (car, motorcycle, heavy)
- **Deep Neural Network (SGD)** - Traffic density classifier
  - Architecture: 12→128→64→32→3
  - Optimizer: SGD
  - Output: 3 classes (Lancar, Padat, Macet)

---

## 📦 Installation

### Prerequisites

- **Docker** & **Docker Compose** installed
- **NVIDIA GPU** (optional, for faster processing)
- **NVIDIA Container Toolkit** (optional, for GPU support)
- **Git**

### 1. Clone Repository

```bash
git clone https://github.com/your-username/neurotraffic.git
cd neurotraffic
```

### 2. Directory Structure

```
neurocomputing/
├── traffic_app/
│   ├── django_app/                 # Django web interface
│   │   ├── templates/              # HTML templates
│   │   ├── static/                 # CSS, JS, images
│   │   └── traffic_app/            # Django app
│   ├── fastapi_service/            # FastAPI backend
│   │   ├── main.py                 # Main API endpoints
│   │   └── Dockerfile
│   ├── assets/
│   │   └── video/
│   │       └── input_vidio/        # Video files
│   ├── models/
│   │   ├── YOLOv8s/
│   │   │   └── train/weights/best.pt
│   │   └── best_deep_neural_network_sgd.pkl
│   └── docker-compose.yml
└── NEURO_MODELING/
    └── best_deep_neural_network_sgd.pkl
```

### 3. Prepare Models

Place your trained models:

```bash
# YOLOv8 model
mkdir -p traffic_app/models/YOLOv8s/train/weights/
cp your_yolo_model.pt traffic_app/models/YOLOv8s/train/weights/best.pt

# DNN model
cp NEURO_MODELING/best_deep_neural_network_sgd.pkl traffic_app/models/
```

### 4. Prepare Video Files

Place video files in format `YYYYMMDD_HHMMSS_*.mp4`:

```bash
mkdir -p traffic_app/assets/video/input_vidio/
# Example: screen_recording_20251105_170236_seg2.mp4
```

### 5. Build & Run with Docker

```bash
cd traffic_app

# Build containers (with host network for pip install)
docker compose build

# Start services
docker compose up -d

# Check logs
docker compose logs -f
```

### 6. Access the Application

- **Home Page**: http://localhost:8001/
- **Traffic Analysis**: http://localhost:8001/analysis/
- **Prediction**: http://localhost:8001/prediction/
- **API Documentation**: http://localhost:8002/docs

---

## 🚀 Usage

### 1. Live Traffic Monitoring

Navigate to **Home Page** (http://localhost:8001/):

1. View real-time video stream with vehicle detection
2. See live counting: Car, Motorcycle, Heavy Vehicle
3. Check per-minute statistics table below video
4. Toggle detection ON/OFF

### 2. Historical Analysis

Go to **Traffic Analysis** (http://localhost:8001/analysis/):

1. Select date range
2. Choose day of week (optional)
3. View interactive charts:
   - Traffic density trends
   - Vehicle type distribution
   - Daily totals
   - Heatmap of traffic patterns

### 3. Video Prediction

Access **Prediction** (http://localhost:8001/prediction/):

1. Upload video file (MP4, AVI, MOV)
2. Select date and hour
3. Click "Process Video"
4. View results:
   - Processed video with detections
   - Per-minute vehicle counts
   - DNN traffic density predictions

---

## 🤖 Model Information

### YOLOv8 Detection Model

```yaml
Model: YOLOv8s (small)
Classes: 3
  - car: Mobil penumpang
  - motorcycle: Sepeda motor
  - heavy: Kendaraan berat (bus, truck)
Input Size: 640x640
Confidence Threshold: 0.25
IoU Threshold: 0.45
```

**Color Coding:**
- 🔵 **Blue**: Car
- 🟢 **Green**: Motorcycle
- 🔴 **Red**: Heavy Vehicle

### DNN Traffic Classifier

```yaml
Model: Deep Neural Network (PyTorch)
Input Features: 12
  1. count_of_car
  2. count_of_motorcycle
  3. count_of_heavy
  4. vehicle_count
  5. motorcycle_ratio
  6. car_ratio
  7. heavy_ratio
  8. car_motorcycle_interaction
  9. density_score
  10. vehicle_count_squared
  11. car_squared
  12. vehicle_variety

Architecture:
  - Input: 12 features
  - Hidden 1: 128 neurons (BatchNorm + ReLU + Dropout 0.3)
  - Hidden 2: 64 neurons (BatchNorm + ReLU + Dropout 0.3)
  - Hidden 3: 32 neurons (BatchNorm + ReLU + Dropout 0.3)
  - Output: 3 classes

Optimizer: SGD
Scaler: MinMaxScaler

Output Classes:
  - Class 1: Lancar (Low/Smooth) - < 50 vehicles
  - Class 2: Padat (Medium/Dense) - 50-150 vehicles
  - Class 3: Macet (High/Congested) - > 150 vehicles
```

**Class Mapping:**

| Model Index | Label | Class Number | Badge Color |
|-------------|-------|--------------|-------------|
| 0 | Lancar | 1 | 🟢 Green |
| 1 | Macet | 3 | 🔴 Red |
| 2 | Padat | 2 | 🟡 Yellow |

---

## 📡 API Documentation

### FastAPI Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns system status and model information.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": true
}
```

#### 2. Video Stream
```http
GET /api/video/stream/{hour}?detection=true
```
Streams video for specific hour with optional detection.

**Parameters:**
- `hour` (int): Target hour (0-23)
- `detection` (bool): Enable/disable detection (default: true)

**Response:** MJPEG stream

#### 3. Stream Data (Per-Minute Statistics)
```http
GET /api/stream/data?limit=20
```
Returns per-minute vehicle counts and DNN predictions.

**Parameters:**
- `limit` (int): Number of minutes to return (default: 20)

**Response:**
```json
{
  "success": true,
  "current_time": "17:23",
  "current_counts": {
    "car": 15,
    "motorcycle": 42,
    "heavy": 3
  },
  "history": [
    {
      "time": "17:22",
      "hour": 17,
      "minute": 22,
      "counts": {
        "car": 12,
        "motorcycle": 38,
        "heavy": 2
      },
      "prediction": {
        "class": 2,
        "label": "Padat",
        "confidence": 0.95
      },
      "timestamp": "2025-01-25T17:22:30.123456"
    }
  ],
  "total_minutes": 20,
  "totals": {
    "car": 240,
    "motorcycle": 760,
    "heavy": 40,
    "total": 1040
  }
}
```

#### 4. Video Prediction
```http
POST /api/predict
Content-Type: multipart/form-data
```

Upload video for analysis and prediction.

**Form Data:**
- `file` (file): Video file
- `date` (string): Date in YYYY-MM-DD format
- `hour` (int): Hour (0-23)

**Response:**
```json
{
  "success": true,
  "results_per_minute": [
    {
      "time": "14:35",
      "counts": {
        "car": 25,
        "motorcycle": 45,
        "heavy": 3
      },
      "prediction": {
        "class": 2,
        "label": "Padat",
        "confidence": 0.92
      }
    }
  ],
  "total_minutes": 5,
  "total_counts": {
    "car": 120,
    "motorcycle": 215,
    "heavy": 12
  },
  "frames_processed": 7500,
  "model_loaded": true
}
```

#### 5. Prediction Stream Output
```http
GET /api/predict/stream
```
Streams processed video with detections.

**Response:** MJPEG stream

---

## 👥 Contributors

<div align="center">

### **Tim Neurocomputing - Traffic Analysis Project**

<table>
  <tr>
    <th>Nama</th>
    <th>NRP</th>
    <th>Role</th>
    <th>Kontribusi</th>
  </tr>
  <tr>
    <td><b>Dimas Firmansyah</b></td>
    <td>3323600034</td>
    <td>Project Lead & ML Engineer</td>
    <td>
      • Model Development (YOLO & DNN)<br>
      • System Architecture<br>
      • Backend Integration
    </td>
  </tr>
  <tr>
    <td><b>Afif Hanifuddin</b></td>
    <td>3323600050</td>
    <td>Data Engineer</td>
    <td>
      • Data Collection & Preprocessing<br>
      • Feature Engineering<br>
      • Model Training Pipeline
    </td>
  </tr>
  <tr>
    <td><b>Adriyans Jusa H</b></td>
    <td>3323600052</td>
    <td>Backend Developer</td>
    <td>
      • FastAPI Development<br>
      • Video Processing<br>
      • API Design
    </td>
  </tr>
  <tr>
    <td><b>M. Ariel Sulton</b></td>
    <td>3323600054</td>
    <td>Frontend Developer</td>
    <td>
      • Web Interface Design<br>
      • Chart Visualization<br>
      • UI/UX Implementation
    </td>
  </tr>
  <tr>
    <td><b>Wahyu Rohman Dwi Putra</b></td>
    <td>3323600043</td>
    <td>DevOps & Testing</td>
    <td>
      • Docker Configuration<br>
      • System Integration<br>
      • Testing & Debugging
    </td>
  </tr>
</table>

</div>

### 🙏 Acknowledgments

Proyek ini dikembangkan sebagai bagian dari mata kuliah **Neurocomputing** dengan dukungan:
- **Institut Teknologi Sepuluh Nopember (ITS)**
- **Departemen Teknik Komputer**
- **Dosen Pembimbing**: [Nama Dosen]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 NeuroTraffic Team - ITS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

### Project Repository
- **GitHub**: [github.com/your-username/neurotraffic](https://github.com/your-username/neurotraffic)
- **Issues**: [github.com/your-username/neurotraffic/issues](https://github.com/your-username/neurotraffic/issues)

### Team Contact
- **Email**: neurotraffic.team@its.ac.id
- **Institution**: Institut Teknologi Sepuluh Nopember (ITS)
- **Location**: Surabaya, Indonesia

---

## 🔮 Future Improvements

- [ ] Real-time alert system untuk kemacetan
- [ ] Mobile application (iOS & Android)
- [ ] Integration dengan Google Maps API
- [ ] Multi-camera support
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Advanced analytics dengan time-series forecasting
- [ ] Integration dengan traffic light control system
- [ ] REST API untuk third-party integration

---

## 📊 Project Statistics

- **Lines of Code**: ~10,000+
- **Development Time**: 4 months
- **Models Trained**: 2 (YOLO + DNN)
- **Accuracy**: 92%+ (DNN), 85%+ (YOLO)
- **Video Processing**: Real-time (30 FPS)
- **Dataset**: 50+ hours of traffic video

---

<div align="center">

**Made with ❤️ by NeuroTraffic Team**

**Institut Teknologi Sepuluh Nopember** | **2025**

[⬆ Back to Top](#neurotraffic-traffic-analysis-system-)

</div>
