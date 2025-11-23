# Traffic Analysis Application
## Analisis Kemacetan Jl. Tunjungan Surabaya

Aplikasi berbasis web untuk analisis lalu lintas menggunakan YOLO (You Only Look Once) dan machine learning clustering.

## Teknologi

- **Backend Web**: Django 4.2
- **API Video Processing**: FastAPI
- **Object Detection**: YOLOv8/YOLO11 (Ultralytics)
- **Frontend**: HTML, CSS, JavaScript Native
- **Container**: Docker Compose
- **GPU Acceleration**: CUDA/cuDNN (RTX 3050)

## Struktur Aplikasi

```
traffic_app/
├── django_app/           # Django web application
│   ├── traffic_project/  # Django project settings
│   ├── traffic_app/      # Main Django app
│   ├── templates/        # HTML templates
│   └── static/           # CSS & JS files
├── fastapi_service/      # FastAPI video processing service
├── docker-compose.yml    # Docker config (GPU)
├── docker-compose.cpu.yml # Docker config (CPU only)
└── run_local.sh          # Local development script
```

## Pages

1. **Home** - Tampilan video CCTV dengan deteksi YOLO real-time
2. **Traffic Analysis** - Dashboard analisis dengan 4 metric cards dan charts
3. **Prediction** - Upload video untuk prediksi deteksi kendaraan

## Cara Menjalankan

### Option 1: Docker Compose (Recommended)

**Dengan GPU NVIDIA:**
```bash
cd /home/dimas/neurocomputing/traffic_app
docker-compose up --build
```

**Tanpa GPU (CPU Only):**
```bash
cd /home/dimas/neurocomputing/traffic_app
docker-compose -f docker-compose.cpu.yml up --build
```

### Option 2: Local Development

```bash
cd /home/dimas/neurocomputing/traffic_app
./run_local.sh
```

Atau jalankan service secara terpisah:
```bash
# Terminal 1 - FastAPI
./run_local.sh fastapi

# Terminal 2 - Django
./run_local.sh django
```

## Akses Aplikasi

- **Web Interface**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8001/docs

## API Endpoints

### Django API
- `GET /api/metrics/` - Traffic metrics
- `GET /api/chart-data/?day=Monday` - Chart data per hari
- `GET /api/video/<hour>/` - Video list for specific hour

### FastAPI API
- `GET /api/videos` - List available videos
- `GET /api/video/stream/{hour}` - Stream video dengan detection
- `GET /api/video/current` - Stream video untuk jam saat ini
- `POST /api/predict` - Upload dan process video

## Features

### Home Page
- Video stream berdasarkan jam terdekat
- YOLO detection overlay (bounding boxes)
- Toggle detection ON/OFF
- Live indicator

### Traffic Analysis Page
- **4 Metric Cards:**
  1. Average Vehicle Count per Nearest Time Period
  2. Peak Hour Range
  3. Peak Day
  4. Dominant Vehicle Type

- **4 Charts:**
  1. Line Chart - Traffic Density per Jam
  2. Pie Chart - Cluster Distribution
  3. Bar Chart - Total Vehicles per Day
  4. Heatmap - Traffic Intensity (Day x Hour)

### Prediction Page
- Upload video file
- Input tanggal dan jam
- YOLO detection processing
- Output video stream
- Detection results (Car, Motorcycle, Heavy Vehicle counts)

## Data Source

- **Video**: `/home/dimas/neurocomputing/assets/video/input_vidio/`
- **CSV Data**: `/home/dimas/neurocomputing/assets/excel/vehicle_counts_fuzzy_clustered.csv`
- **YOLO Model**: `/home/dimas/neurocomputing/NEURO_MODELING/YOLOv8s/train/weights/best.pt`

## Nearest Time Logic

Penentuan video berdasarkan jam:
- Menit 00-29: Ambil video jam saat ini (13:22 → video jam 13:00)
- Menit 30-59: Ambil video jam berikutnya (13:38 → video jam 14:00)

## Cluster Interpretation

- **Cluster 1**: Low traffic (hijau)
- **Cluster 2**: Medium traffic (kuning)
- **Cluster 3**: High traffic (merah)

## Requirements

- Python 3.10+
- CUDA 11.8+ (untuk GPU)
- Docker & Docker Compose
- 8GB+ RAM
- GPU: NVIDIA RTX 3050 atau lebih tinggi (optional)
