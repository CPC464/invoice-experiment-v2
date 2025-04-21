#!/bin/bash

# Set app directory
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/invoice-parser"

# Stop any running processes on Ctrl+C
trap "exit" INT TERM
trap "kill 0" EXIT

# Check for Python and required packages
if ! command -v python3 &> /dev/null; then
  echo "Python 3 is required but not installed. Please install it first."
  exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
  
  echo "Installing dependencies..."
  source venv/bin/activate
  pip install -r "$APP_DIR/requirements.txt"
else
  source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f "$APP_DIR/.env" ]; then
  echo "Creating .env file from template..."
  cp "$APP_DIR/.env.template" "$APP_DIR/.env"
  echo "Please edit the .env file with your API keys before continuing."
  exit 1
fi

# Start the Flask backend
echo "Starting Flask backend..."
cd "$APP_DIR" && python app.py &
FLASK_PID=$!
echo "Flask server running with PID: $FLASK_PID"

# Wait a moment for Flask to start
sleep 2

# Start the Streamlit frontend
echo "Starting Streamlit frontend..."
cd "$APP_DIR" && streamlit run streamlit_app.py &
STREAMLIT_PID=$!
echo "Streamlit server running with PID: $STREAMLIT_PID"

echo "Both servers are now running."
echo "Press Ctrl+C to stop all servers."

# Wait for processes to finish
wait 