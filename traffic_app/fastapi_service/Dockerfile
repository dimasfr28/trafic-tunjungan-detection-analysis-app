FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install heavy packages first to avoid re-download on failure
RUN pip3 install --no-cache-dir --timeout 7200 torch==2.1.0 || \
    pip3 install --no-cache-dir --timeout 7200 torch==2.1.0
RUN pip3 install --no-cache-dir --timeout 3600 torchvision==0.16.0
RUN pip3 install --no-cache-dir --timeout 3600 -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]