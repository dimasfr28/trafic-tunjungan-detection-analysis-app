#!/bin/bash

# Traffic Analysis Application - Local Development Script
# Usage: ./run_local.sh [django|fastapi|all]

BASE_DIR="/home/dimas/neurocomputing"
APP_DIR="$BASE_DIR/traffic_app"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Traffic Analysis Application${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed${NC}"
    exit 1
fi

# Create virtual environment if not exists
VENV_DIR="$APP_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q django==4.2 requests pandas python-dateutil
pip install -q fastapi uvicorn python-multipart opencv-python-headless numpy aiofiles
pip install -q torch torchvision ultralytics

# Function to run Django
run_django() {
    echo -e "${GREEN}Starting Django server on http://localhost:8001${NC}"
    cd "$APP_DIR/django_app"
    export DEBUG=1
    export FASTAPI_URL=http://localhost:8002
    python manage.py runserver 0.0.0.0:8001
}

# Function to run FastAPI
run_fastapi() {
    echo -e "${GREEN}Starting FastAPI server on http://localhost:8002${NC}"
    cd "$APP_DIR/fastapi_service"
    uvicorn main:app --host 0.0.0.0 --port 8002 --reload
}

# Main
case "$1" in
    django)
        run_django
        ;;
    fastapi)
        run_fastapi
        ;;
    all|*)
        echo -e "${YELLOW}Starting both services...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""

        # Run FastAPI in background
        cd "$APP_DIR/fastapi_service"
        uvicorn main:app --host 0.0.0.0 --port 8002 &
        FASTAPI_PID=$!

        # Wait a bit for FastAPI to start
        sleep 2

        # Run Django in foreground
        cd "$APP_DIR/django_app"
        export DEBUG=1
        export FASTAPI_URL=http://localhost:8002
        python manage.py runserver 0.0.0.0:8001 &
        DJANGO_PID=$!

        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Services Running:${NC}"
        echo -e "${GREEN}  - Django:  http://localhost:8001${NC}"
        echo -e "${GREEN}  - FastAPI: http://localhost:8002${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""

        # Wait for Ctrl+C
        trap "kill $FASTAPI_PID $DJANGO_PID 2>/dev/null; exit" INT
        wait
        ;;
esac
