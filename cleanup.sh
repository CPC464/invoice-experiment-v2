#!/bin/bash
# Quick script to kill all processes using port 5002
# Use with caution!

# Set the base directory to the script location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Application directory
APP_DIR="$BASE_DIR/invoice-parser"

# Read port from .env.local
if [ ! -f "$APP_DIR/.env.local" ]; then
    echo "Error: .env.local file not found in $APP_DIR"
    exit 1
fi

# Extract PORT from .env.local
PORT=$(grep -E "^PORT=" "$APP_DIR/.env.local" | cut -d "=" -f2)

if [ -z "$PORT" ]; then
    echo "Error: PORT not defined in .env.local"
    exit 1
fi

echo "Looking for processes on port $PORT..."

# Get PIDs of processes using the port (macOS specific)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    PIDS=$(lsof -i:$PORT -t)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    PIDS=$(lsof -i:$PORT -t)
else
    echo "Unsupported OS. Please use cleanup.py instead."
    exit 1
fi

if [ -z "$PIDS" ]; then
    echo "No processes found using port $PORT."
else
    echo "Found processes with PIDs: $PIDS"
    echo "Terminating processes..."

    for PID in $PIDS; do
        echo "Killing process $PID..."
        kill -15 $PID
        # Check if it worked, if not force kill
        sleep 0.5
        if kill -0 $PID 2>/dev/null; then
            echo "Process $PID didn't terminate gracefully, force killing..."
            kill -9 $PID
        fi
    done

    echo "Done. All processes on port $PORT should be terminated."
fi

# Define the directories to clean
LOGS_DIR="$APP_DIR/logs"
THUMBNAILS_DIR="$APP_DIR/thumbnails"
UPLOADS_DIR="$APP_DIR/uploads"
RESULTS_DIR="$APP_DIR/results"

# Clean up log files
if [ -d "$LOGS_DIR" ]; then
    echo "Cleaning up log files in $LOGS_DIR..."
    rm -f $LOGS_DIR/*.log
    echo "Log files removed."
else
    echo "Logs directory not found at $LOGS_DIR."
fi

# Clean up thumbnails
if [ -d "$THUMBNAILS_DIR" ]; then
    echo "Cleaning up thumbnails in $THUMBNAILS_DIR..."
    rm -f $THUMBNAILS_DIR/*.jpg $THUMBNAILS_DIR/*.jpeg $THUMBNAILS_DIR/*.png
    echo "Thumbnails removed."
else
    echo "Thumbnails directory not found at $THUMBNAILS_DIR."
fi

# Clean up uploads
if [ -d "$UPLOADS_DIR" ]; then
    echo "Cleaning up uploads in $UPLOADS_DIR..."
    rm -f $UPLOADS_DIR/*.pdf $UPLOADS_DIR/*.jpg $UPLOADS_DIR/*.jpeg $UPLOADS_DIR/*.png $UPLOADS_DIR/*.tiff
    echo "Uploads removed."
else
    echo "Uploads directory not found at $UPLOADS_DIR."
fi

# Clean up results
if [ -d "$RESULTS_DIR" ]; then
    echo "Cleaning up results in $RESULTS_DIR..."
    rm -f $RESULTS_DIR/*.json
    echo "Results removed."
else
    echo "Results directory not found at $RESULTS_DIR."
fi

echo "Cleanup complete. You can now restart your application." 