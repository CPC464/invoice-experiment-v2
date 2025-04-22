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
from duplicate_detector import get_duplicate_detector
import requests
from datetime import datetime

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
            job_id, file_path, original_filename, is_duplicate = processing_queue.get()

            # Skip processing if this is a duplicate
            if is_duplicate:
                # Make sure the job is already marked as skipped - it should be from the upload step
                if (
                    job_id in processing_status
                    and processing_status[job_id].get("status") != "skipped"
                ):
                    processing_status[job_id]["status"] = "skipped"

                # Mark task as done and continue
                processing_queue.task_done()
                continue

            # Update status to processing
            processing_status[job_id] = {
                "status": "processing",
                "file": os.path.basename(file_path),
                "original_filename": original_filename,
            }

            try:
                # Process the invoice - enable duplicate detection by default
                result = process_invoice(
                    file_path, check_duplicates=True, auto_reject_duplicates=True
                )

                # Check if this was rejected as a duplicate
                if result.get("is_duplicate", False):
                    processing_status[job_id] = {
                        "status": "duplicate",
                        "file": os.path.basename(file_path),
                        "original_filename": original_filename,
                        "duplicate_info": result.get("duplicate_info", {}),
                    }
                    processing_queue.task_done()
                    continue

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
    print("Upload request received")  # Debug log
    # Check if file is included in request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Check if this is a duplicate file (passed from frontend)
    is_duplicate = request.form.get("is_duplicate", "false").lower() == "true"
    print(f"Is duplicate flag: {is_duplicate}")  # Debug log

    # Validate file type
    if file and allowed_file(file.filename):
        # Generate unique ID and secure the filename
        job_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filename = f"{job_id}_{original_filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Save the file
        file.save(file_path)
        print(f"File saved: {file_path}")  # Debug log

        # Always generate thumbnail for all files
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, f"{job_id}.jpg")
        try:
            from invoice_processor import generate_thumbnail

            generate_thumbnail(file_path, thumbnail_path)
            print(f"Thumbnail generated: {thumbnail_path}")  # Debug log
        except Exception as e:
            print(f"Error generating thumbnail: {str(e)}")
            thumbnail_path = None

        # Get current time for timestamps
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle duplicate files differently
        if is_duplicate:
            print(f"Processing as duplicate: {job_id}")  # Debug log
            # Check for duplicate info
            duplicate_info = None
            similarity_score = None

            try:
                # Get duplicate info directly
                duplicate_detector = get_duplicate_detector()
                duplicate_info = duplicate_detector.check_for_duplicates(file_path)
                similarity_score = duplicate_info.get("highest_score", 100)
                print(f"Duplicate detection result: {similarity_score}%")  # Debug log
            except Exception as e:
                print(f"Error checking for duplicates: {str(e)}")
                # Default values if duplicate check fails
                duplicate_info = {"is_duplicate": True, "highest_score": 100}
                similarity_score = 100

            # Update status to skipped (for duplicates)
            skipped_status = {
                "status": "skipped",
                "file": filename,
                "original_filename": original_filename,
                "thumbnail_path": thumbnail_path,
                "is_duplicate": True,
                "time": current_time,
                "duplicate_info": duplicate_info,
                "similarity_score": similarity_score,
            }

            # Store in processing status
            processing_status[job_id] = skipped_status
            print(
                f"Added to processing_status with status 'skipped': {job_id}"
            )  # Debug log

            # Return response for duplicate with thumbnail path
            return (
                jsonify(
                    {
                        "message": "File uploaded and marked as duplicate",
                        "job_id": job_id,
                        "status": "skipped",
                        "thumbnail_path": thumbnail_path,
                        "time": current_time,
                        "duplicate_info": duplicate_info,
                        "similarity_score": similarity_score,
                    }
                ),
                202,
            )

        # Add to processing queue for normal processing
        processing_queue.put((job_id, file_path, original_filename, is_duplicate))
        print(f"Added to processing queue: {job_id}")  # Debug log

        # Update status
        processing_status[job_id] = {
            "status": "queued",
            "file": filename,
            "original_filename": original_filename,
            "thumbnail_path": thumbnail_path if thumbnail_path else None,
            "time": current_time,
        }
        print(f"Added to processing_status with status 'queued': {job_id}")  # Debug log

        # Return response with thumbnail path
        return (
            jsonify(
                {
                    "message": "File uploaded successfully",
                    "job_id": job_id,
                    "status": "processing",
                    "thumbnail_path": thumbnail_path if thumbnail_path else None,
                    "time": current_time,
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
    print(f"Current jobs in processing_status: {len(processing_status)}")  # Debug log
    # Print each job's ID and status for debugging
    for job_id, job_info in processing_status.items():
        print(f"Job {job_id}: status={job_info.get('status', 'unknown')}")

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
        clear_directory(RESULTS_FOLDER)

        # Explicitly delete vector index and metadata files that may be outside the results folder
        vector_index_path = "invoice-parser/results/vector_index"
        vector_metadata_path = "invoice-parser/results/vector_metadata.json"

        for path in [vector_index_path, vector_metadata_path]:
            if os.path.exists(path):
                try:
                    if os.path.isfile(path):
                        os.unlink(path)
                        print(f"Deleted {path}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Deleted directory {path}")
                except Exception as e:
                    print(f"Error deleting {path}: {str(e)}")

        # Create empty consolidated results file - make sure the directory exists
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        save_consolidated_results({"results": []})
        print(
            f"Recreated empty consolidated results file at {CONSOLIDATED_RESULTS_FILE}"
        )

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

        # Reset duplicate detector index if it exists
        try:
            # Actually recreate the duplicate detector to ensure we get a fresh index
            from duplicate_detector import InvoiceDuplicateDetector

            # Initialize a new detector with default paths (will create empty files)
            new_detector = InvoiceDuplicateDetector()
            # Save the empty index
            new_detector._save_index()
            print("Created fresh duplicate detector index")
        except Exception as e:
            print(f"Error resetting duplicate detector: {str(e)}")

        return jsonify({"message": "All data has been cleared successfully"}), 200
    except Exception as e:
        print(f"Error in clear_all_data: {str(e)}")
        return jsonify({"error": f"Error clearing data: {str(e)}"}), 500


@app.route("/api/check-duplicate", methods=["POST"])
def check_duplicate():
    """
    API endpoint to check if an invoice is a duplicate without processing it
    """
    # Check if file is included in request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Validate file type
    if file and allowed_file(file.filename):
        # Generate unique temporary ID and secure the filename
        temp_id = f"temp_{str(uuid.uuid4())}"
        original_filename = secure_filename(file.filename)
        filename = f"{temp_id}_{original_filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            # Save the file temporarily
            file.save(file_path)

            # Check for duplicates
            duplicate_detector = get_duplicate_detector()
            duplicate_check = duplicate_detector.check_for_duplicates(file_path)

            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Return the duplicate check results
            return (
                jsonify(
                    {
                        "filename": original_filename,
                        "is_duplicate": duplicate_check.get("is_duplicate", False),
                        "is_related": duplicate_check.get("is_related", False),
                        "highest_score": duplicate_check.get("highest_score", 0),
                        "similar_documents": duplicate_check.get(
                            "similar_documents", []
                        ),
                    }
                ),
                200,
            )

        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Error checking for duplicates: {str(e)}"}), 500

    # Invalid file type
    return jsonify({"error": "File type not allowed"}), 400


@app.route("/api/duplicate-stats", methods=["GET"])
def get_duplicate_stats():
    """
    API endpoint to get statistics about the duplicate detection index
    """
    try:
        duplicate_detector = get_duplicate_detector()

        # Get basic statistics
        num_documents = (
            duplicate_detector.index.ntotal
            if hasattr(duplicate_detector, "index")
            else 0
        )

        # Get model info
        model_info = {
            "name": (
                duplicate_detector.model_name
                if hasattr(duplicate_detector, "model_name")
                else "unknown"
            ),
            "vector_dimension": (
                duplicate_detector.vector_dim
                if hasattr(duplicate_detector, "vector_dim")
                else 0
            ),
        }

        # Get threshold settings
        from duplicate_detector import IDENTICAL_THRESHOLD, RELATED_THRESHOLD

        threshold_info = {
            "identical_threshold": IDENTICAL_THRESHOLD,
            "related_threshold": RELATED_THRESHOLD,
        }

        return (
            jsonify(
                {
                    "num_documents": num_documents,
                    "model_info": model_info,
                    "threshold_info": threshold_info,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": f"Error getting duplicate stats: {str(e)}"}), 500


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
