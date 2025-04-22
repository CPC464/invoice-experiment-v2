#!/bin/bash

# Set the base directory to the script location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Application directory
APP_DIR="$BASE_DIR/invoice-parser"

# Terminal colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Invoice Parser Startup Script      ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Check for .env.local file first
if [ ! -f "$APP_DIR/.env.local" ]; then
    echo -e "${RED}Error: .env.local file not found in $APP_DIR${NC}"
    echo -e "${YELLOW}Please create .env.local file by copying .env.template and filling in your API keys${NC}"
    exit 1
fi

# Extract PORT from .env.local file
PORT=$(grep -E "^PORT=" "$APP_DIR/.env.local" | cut -d '=' -f2)
if [ -z "$PORT" ]; then
    echo -e "${RED}Error: PORT not defined in .env.local${NC}"
    exit 1
fi
echo -e "${GREEN}Using PORT: ${PORT}${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Tesseract OCR (required for duplicate detection)
check_tesseract() {
    echo -e "${YELLOW}Checking for Tesseract OCR installation...${NC}"
    if command_exists tesseract; then
        TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n 1)
        echo -e "${GREEN}✓ Tesseract OCR is installed: ${TESSERACT_VERSION}${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Tesseract OCR is not installed or not in PATH${NC}"
        echo -e "${YELLOW}The duplicate detection feature will have limited functionality.${NC}"
        echo -e "${YELLOW}To install Tesseract OCR:${NC}"
        echo -e "${YELLOW}  - Ubuntu/Debian: sudo apt install tesseract-ocr${NC}"
        echo -e "${YELLOW}  - macOS: brew install tesseract${NC}"
        echo -e "${YELLOW}  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki${NC}"
        echo -e "${YELLOW}             and ensure it's in your PATH${NC}"
        return 1
    fi
}

# Check for Python
if ! command_exists python3; then
    echo -e "${YELLOW}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check Python version
PY_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}Using Python version: ${PY_VERSION}${NC}"

# Determine Python binary to use for venv creation (prefer 3.12 if available)
PYTHON_BIN="python3"
if command_exists python3.12; then
    PYTHON_BIN="python3.12"
    echo -e "${GREEN}Python 3.12 found and will be used to create the virtual environment${NC}"
elif [[ $PY_VERSION == 3.13* ]]; then
    echo -e "${YELLOW}Note: You're using Python ${PY_VERSION}. Some libraries may have compatibility issues.${NC}"
    echo -e "${YELLOW}Installing Python 3.12 is recommended for best compatibility.${NC}"
fi

# Set up virtual environment
VENV_DIR="$BASE_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment using ${PYTHON_BIN}...${NC}"
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows with Git Bash or similar
    source "$VENV_DIR/Scripts/activate"
else
    # macOS, Linux, etc.
    source "$VENV_DIR/bin/activate"
fi

# Check if requirements are installed
echo -e "${YELLOW}Checking requirements...${NC}"
python3 -m pip install --quiet --upgrade pip
python3 -c "import flask" >/dev/null 2>&1 || MISSING_REQS=1
python3 -c "import streamlit" >/dev/null 2>&1 || MISSING_REQS=1

if [ "$MISSING_REQS" == "1" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r "$APP_DIR/requirements.txt"
else
    echo -e "${GREEN}All requirements are already installed${NC}"
fi

# Check for faiss (required for duplicate detection)
echo -e "${YELLOW}Checking for FAISS (vector similarity) installation...${NC}"
if ! python3 -c "import faiss" >/dev/null 2>&1; then
    echo -e "${YELLOW}FAISS not found, installing faiss-cpu...${NC}"
    pip install faiss-cpu
    
    # Verify installation
    if ! python3 -c "import faiss" >/dev/null 2>&1; then
        echo -e "${RED}⚠️  FAISS installation failed. Duplicate detection will not work.${NC}"
        echo -e "${YELLOW}If you're on an M1/M2 Mac, try:${NC}"
        echo -e "${YELLOW}  pip install faiss-cpu --no-cache-dir${NC}"
        echo -e "${YELLOW}or install using conda:${NC}"
        echo -e "${YELLOW}  conda install -c conda-forge faiss${NC}"
    else
        echo -e "${GREEN}✓ FAISS installed successfully${NC}"
    fi
else
    echo -e "${GREEN}✓ FAISS is already installed${NC}"
fi

# Check for sentence-transformers (required for duplicate detection)
echo -e "${YELLOW}Checking for sentence-transformers installation...${NC}"
if ! python3 -c "import sentence_transformers" >/dev/null 2>&1; then
    echo -e "${YELLOW}sentence-transformers not found, installing...${NC}"
    pip install sentence-transformers
    
    # Verify installation
    if ! python3 -c "import sentence_transformers" >/dev/null 2>&1; then
        echo -e "${RED}⚠️  sentence-transformers installation failed. Duplicate detection will not work.${NC}"
    else
        echo -e "${GREEN}✓ sentence-transformers installed successfully${NC}"
    fi
else
    echo -e "${GREEN}✓ sentence-transformers is already installed${NC}"
fi

# Check for pytesseract Python module (different from system Tesseract)
echo -e "${YELLOW}Checking for pytesseract Python module...${NC}"
if ! python3 -c "import pytesseract" >/dev/null 2>&1; then
    echo -e "${YELLOW}pytesseract Python module not found, installing...${NC}"
    pip install pytesseract
    
    # Verify installation
    if ! python3 -c "import pytesseract" >/dev/null 2>&1; then
        echo -e "${RED}⚠️  pytesseract installation failed. OCR will not work.${NC}"
    else
        echo -e "${GREEN}✓ pytesseract Python module installed successfully${NC}"
    fi
else
    echo -e "${GREEN}✓ pytesseract Python module is already installed${NC}"
fi

# Check for Tesseract OCR
check_tesseract

# Make sure upload and results directories exist
mkdir -p "$APP_DIR/uploads"
mkdir -p "$APP_DIR/results"
mkdir -p "$APP_DIR/thumbnails"

# Make log directory
mkdir -p "$APP_DIR/logs"

# Get the full path to the virtual environment's Python
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"

# Start both services in background
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Starting Flask backend on port ${PORT}...${NC}"
cd "$APP_DIR" && $VENV_PYTHON app.py &
FLASK_PID=$!

# Wait a bit for Flask to start
sleep 2

echo -e "${GREEN}Starting Streamlit frontend on port 8501...${NC}"
cd "$APP_DIR" && $VENV_STREAMLIT run streamlit_app.py > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Services started successfully!${NC}"
echo -e "${YELLOW}Backend API:${NC} http://localhost:${PORT}"
echo -e "${YELLOW}Frontend UI:${NC} http://localhost:8501"
echo -e "${YELLOW}Log files:${NC} $APP_DIR/logs/flask.log and $APP_DIR/logs/streamlit.log"
echo -e "${BLUE}=====================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Trap Ctrl+C to kill both processes
trap 'kill $FLASK_PID $STREAMLIT_PID 2>/dev/null' INT

# Wait for both processes to finish
wait $FLASK_PID $STREAMLIT_PID 