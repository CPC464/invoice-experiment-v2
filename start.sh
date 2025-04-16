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
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Invoice Parser Startup Script      ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
if ! command_exists python3; then
    echo -e "${YELLOW}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check Python version
PY_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}Using Python version: ${PY_VERSION}${NC}"

# Set up virtual environment
VENV_DIR="$BASE_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
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

# Make sure upload and results directories exist
mkdir -p "$APP_DIR/uploads"
mkdir -p "$APP_DIR/results"
mkdir -p "$APP_DIR/thumbnails"

# Make log directory
mkdir -p "$APP_DIR/logs"

# Start both services in background
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Starting Flask backend on port 5002...${NC}"
cd "$APP_DIR" && python3 app.py > logs/flask.log 2>&1 &
FLASK_PID=$!

# Wait a bit for Flask to start
sleep 2

echo -e "${GREEN}Starting Streamlit frontend on port 8501...${NC}"
cd "$APP_DIR" && streamlit run streamlit_app.py > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Services started successfully!${NC}"
echo -e "${YELLOW}Backend API:${NC} http://localhost:5002"
echo -e "${YELLOW}Frontend UI:${NC} http://localhost:8501"
echo -e "${YELLOW}Log files:${NC} $APP_DIR/logs/flask.log and $APP_DIR/logs/streamlit.log"
echo -e "${BLUE}=====================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Trap Ctrl+C to kill both processes
trap 'kill $FLASK_PID $STREAMLIT_PID 2>/dev/null' INT

# Wait for both processes to finish
wait $FLASK_PID $STREAMLIT_PID 