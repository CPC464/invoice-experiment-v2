from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
import threading
import queue
import shutil
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from invoice_processor import process_invoice

# Load environment variables
load_dotenv(dotenv_path=".env.local")  # Updated to use .env.local instead of .env

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Setup directories from environment or use defaults
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")
THUMBNAILS_FOLDER = os.getenv("THUMBNAILS_FOLDER", "thumbnails")
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff"}
CONSOLIDATED_RESULTS_FILE = os.path.join(RESULTS_FOLDER, "all_results.json")

# Configure app
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

# Create a processing queue
processing_queue = queue.Queue()
# Dictionary to track processing status
processing_status = {}


def allowed_file(filename):
    """
    Check if the file extension is in the allowed list
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_consolidated_results():
    """
    Load the consolidated results JSON file or create a new one if it doesn't exist
    """
    if os.path.exists(CONSOLIDATED_RESULTS_FILE):
        try:
            with open(CONSOLIDATED_RESULTS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # File exists but is not valid JSON
            return {"results": []}
    else:
        # Create new results structure
        return {"results": []}


def save_consolidated_results(results_data):
    """
    Save the consolidated results to the JSON file
    """
    with open(CONSOLIDATED_RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=2)


# Worker function to process items in the queue
def process_queue_worker():
    """
    Background worker to process invoices from the queue
    """
    while True:
        try:
            # Get item from queue
            job_id, file_path, original_filename = processing_queue.get()

            # Update status to processing
            processing_status[job_id] = {
                "status": "processing",
                "file": os.path.basename(file_path),
                "original_filename": original_filename,
            }

            try:
                # Process the invoice
                result = process_invoice(file_path)

                # Generate thumbnail
                thumbnail_path = os.path.join(THUMBNAILS_FOLDER, f"{job_id}.jpg")
                from invoice_processor import generate_thumbnail

                generate_thumbnail(file_path, thumbnail_path)

                # Add additional metadata
                result["job_id"] = job_id
                result["original_filename"] = original_filename
                result["thumbnail_path"] = thumbnail_path

                # Load the consolidated results
                consolidated_results = load_consolidated_results()

                # Add this result to the array
                consolidated_results["results"].append(result)

                # Save the updated consolidated results
                save_consolidated_results(consolidated_results)

                # Update status to completed
                processing_status[job_id] = {
                    "status": "completed",
                    "file": os.path.basename(file_path),
                    "original_filename": original_filename,
                    "thumbnail_path": thumbnail_path,
                    "result": result,  # Store result in the status dictionary
                }
            except Exception as e:
                # Update status with error
                processing_status[job_id] = {
                    "status": "error",
                    "file": os.path.basename(file_path),
                    "original_filename": original_filename,
                    "error": str(e),
                }

            # Mark task as done
            processing_queue.task_done()
        except Exception as e:
            print(f"Error in worker: {e}")


# Start worker threads
num_worker_threads = int(os.getenv("NUMBER_OF_WORKER_THREADS", "10"))
for i in range(num_worker_threads):
    t = threading.Thread(target=process_queue_worker, daemon=True)
    t.start()


@app.route("/api/upload-invoice", methods=["POST"])
def upload_file():
    """
    API endpoint to upload and queue an invoice for processing
    """
    # Check if file is included in request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Validate file type
    if file and allowed_file(file.filename):
        # Generate unique ID and secure the filename
        job_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filename = f"{job_id}_{original_filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the file
        file.save(file_path)

        # Add to processing queue
        processing_queue.put((job_id, file_path, original_filename))

        # Update status
        processing_status[job_id] = {
            "status": "queued",
            "file": filename,
            "original_filename": original_filename,
        }

        # Return response
        return (
            jsonify(
                {
                    "message": "File uploaded successfully",
                    "job_id": job_id,
                    "status": "processing",
                }
            ),
            202,
        )

    # Invalid file type
    return jsonify({"error": "File type not allowed"}), 400


@app.route("/api/status/<job_id>", methods=["GET"])
def check_status(job_id):
    """
    API endpoint to check the status of a processing job
    """
    if job_id in processing_status:
        status = processing_status[job_id]

        # If job is completed, include the result directly from the status dictionary
        if status["status"] == "completed":
            return jsonify(status)
        else:
            return jsonify(status)

    # Job not found
    return jsonify({"error": "Job not found"}), 404


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """
    API endpoint to list all jobs and their statuses
    """
    return jsonify(processing_status)


@app.route("/api/all-results", methods=["GET"])
def get_all_results():
    """
    API endpoint to get all processed results
    """
    consolidated_results = load_consolidated_results()
    return jsonify(consolidated_results)


@app.route("/thumbnails/<filename>", methods=["GET"])
def get_thumbnail(filename):
    """
    Serve thumbnail files
    """
    return send_from_directory(THUMBNAILS_FOLDER, filename)


@app.route("/api/clear-all", methods=["POST"])
def clear_all_data():
    """
    API endpoint to clear all data (uploads, thumbnails, results, and logs)
    """
    try:
        global processing_status

        # Clear processing queue and status
        with processing_queue.mutex:
            processing_queue.queue.clear()
        processing_status = {}

        # Function to safely delete files in a directory
        def clear_directory(directory):
            if os.path.exists(directory) and os.path.isdir(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")

        # Clear directories
        clear_directory(UPLOAD_FOLDER)
        clear_directory(THUMBNAILS_FOLDER)

        # Clear results directory but don't delete the consolidated results file yet
        if os.path.exists(RESULTS_FOLDER) and os.path.isdir(RESULTS_FOLDER):
            for filename in os.listdir(RESULTS_FOLDER):
                if filename != os.path.basename(
                    CONSOLIDATED_RESULTS_FILE
                ):  # Skip the consolidated file for now
                    file_path = os.path.join(RESULTS_FOLDER, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")

        # Also clear log files
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            # Clear all log files except the directory itself
            for filename in os.listdir(logs_dir):
                file_path = os.path.join(logs_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        # Truncate the file instead of deleting it
                        with open(file_path, "w") as f:
                            pass  # Truncate to 0 bytes
                        print(f"Cleared log file: {file_path}")
                except Exception as e:
                    print(f"Error clearing log file {file_path}: {str(e)}")

        # Create empty consolidated results file - make sure the directory exists
        os.makedirs(os.path.dirname(CONSOLIDATED_RESULTS_FILE), exist_ok=True)
        save_consolidated_results({"results": []})
        print(
            f"Recreated empty consolidated results file at {CONSOLIDATED_RESULTS_FILE}"
        )

        return jsonify({"message": "All data has been cleared successfully"}), 200
    except Exception as e:
        print(f"Error in clear_all_data: {str(e)}")
        return jsonify({"error": f"Error clearing data: {str(e)}"}), 500


if __name__ == "__main__":
    # Get port from environment variable (centrally defined in .env file)
    port = int(os.getenv("PORT", "5002"))

    # Register signal handlers for graceful shutdown
    import signal
    import sys

    def signal_handler(sig, frame):
        print("Received shutdown signal, exiting gracefully...")
        sys.exit(0)

    # Register handlers for SIGINT (Ctrl+C) and SIGTERM (kill command)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Use threaded=False for better shutdown behavior
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True, use_reloader=True)
