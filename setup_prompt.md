# Invoice Parser Development Prompt

## Project Overview

Create a comprehensive invoice parsing application that can extract structured data from invoices in PDF or image format. The application should use LangChain with vision-capable LLMs to directly process visual invoice content, identify key information fields, and return the data in a standardized JSON format.

## Technical Requirements

### Core Functionality

1. Develop a system that accepts invoice files in PDF or common image formats (JPG, PNG, TIFF)
2. Use LangChain to integrate with vision-capable LLMs (like GPT-4 Vision, Claude 3, or Gemini) to directly process visual content
3. Extract and structure information from the invoices without requiring separate OCR processes
4. Identify and extract the following specific invoice fields:
   - vendor_name (the company or entity that issued the invoice)
   - due_date (when payment is required)
   - paid_date (when the invoice was actually paid, if applicable)
   - service_from (start date of the service period)
   - service_to (end date of the service period)
   - currency (the currency used for the invoice)
   - net_amount (the amount before tax/VAT)
   - vat_amount (the value-added tax or similar tax amount)
   - gross_amount (the total amount including taxes)
5. Output the extracted data in a well-structured JSON format
6. Include validation mechanisms to ensure data quality and completeness

### Technical Stack

1. LangChain as the primary framework for building the AI workflow
2. Integration with vision-capable LLMs (such as OpenAI's GPT-4 Vision, Anthropic's Claude 3, or Google's Gemini)
3. Python for all development
4. Flask for creating a simple backend API server
5. Streamlit for developing a minimalistic user interface
6. PDF handling libraries for processing multi-page PDF documents if needed
7. JSON processing libraries for output formatting

## Implementation Steps

Please provide code and explanations for:

1. **Environment Setup**

   - Required dependencies and libraries
   - Configuration for LangChain and vision-capable LLM integration

2. **File Handling**

   - Functions to accept and validate input files
   - Handling different file formats (PDFs vs. images)
   - Techniques for processing multi-page PDFs if necessary

3. **Vision LLM Integration**

   - Setting up vision model connections through LangChain
   - Encoding and transmitting visual content to the LLM
   - Managing token/input size limitations for large invoices

4. **Prompt Engineering**

   - Designing effective prompts to guide the vision LLM
   - Structuring system prompts for consistent field extraction
   - Using LangChain's prompt templates for reliable extraction

5. **User Interface Development**

   - Creating a Streamlit application for invoice uploads
   - Designing a clean, minimalistic interface
   - Building a results table to display extracted invoice data
   - Adding download options for the extracted JSON data

6. **Backend API Development**

   - Setting up a Flask server to handle requests
   - Creating API endpoints for invoice processing
   - Managing file uploads and validation
   - Implementing error handling and response formatting

7. **Validation**

   - Data validation mechanisms
   - Confidence scores for extracted fields
   - Handling of edge cases and anomalies

8. **Usage Examples**
   - Sample code demonstrating how to use the system
   - Example inputs and expected outputs

## Project Requirements

Create a complete Python-based invoice parsing application with the following components:

1. **Flask Backend API**

   - Create a RESTful API service to process invoice files
   - Implement endpoints for file upload, processing, and result retrieval
   - Implement a background processing queue for handling multiple invoices simultaneously
   - Track processing status for each uploaded invoice
   - Store processing results in JSON files within the project directory
   - Handle authentication and security concerns if necessary

2. **Streamlit Frontend**

   - Design a clean, minimalistic UI for uploading invoice files
   - Create a file upload component that supports PDFs and images
   - Display a table showing extracted information from processed invoices
   - Implement options to download results as JSON
   - Add basic validation and error messaging for user feedback

3. **LangChain Integration**

   - Set up LangChain components to work with vision-capable LLMs
   - Design effective prompt templates for invoice field extraction
   - Implement chain components to process the LLM responses into structured data
   - Add caching mechanisms to improve performance

4. **Invoice Processing Logic**

   - Develop processing pipelines for both PDF and image formats
   - Extract the specified invoice fields accurately
   - Handle various invoice formats and layouts
   - Implement validation for extracted data

5. **Data Management**

   - Structure the JSON output format to match required fields
   - Implement data normalization (especially for dates and currency amounts)
   - Add confidence scores for extracted fields when possible

6. **Deployment and Configuration**
   - Include instructions for setting up development and production environments
   - Provide configuration options for API keys and model selection
   - Document scaling considerations for production use

## Sample Code Structure

Please structure your solution with code similar to the following examples:

### 1. Flask Backend (`app.py`)

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
import json
import threading
import queue
from invoice_processor import process_invoice

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Setup directories
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Create a processing queue
processing_queue = queue.Queue()
# Dictionary to track processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Worker function to process items in the queue
def process_queue_worker():
    while True:
        try:
            # Get item from queue
            job_id, file_path = processing_queue.get()

            # Update status
            processing_status[job_id] = {
                'status': 'processing',
                'file': os.path.basename(file_path)
            }

            try:
                # Process the invoice
                result = process_invoice(file_path)

                # Save result to JSON file
                result_path = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)

                # Update status
                processing_status[job_id] = {
                    'status': 'completed',
                    'file': os.path.basename(file_path),
                    'result_path': result_path
                }
            except Exception as e:
                # Update status with error
                processing_status[job_id] = {
                    'status': 'error',
                    'file': os.path.basename(file_path),
                    'error': str(e)
                }

            # Mark task as done
            processing_queue.task_done()
        except Exception as e:
            print(f"Error in worker: {e}")

# Start worker threads
num_worker_threads = 3
for i in range(num_worker_threads):
    t = threading.Thread(target=process_queue_worker, daemon=True)
    t.start()

@app.route('/api/upload-invoice', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)

        # Add to processing queue
        processing_queue.put((job_id, file_path))

        # Update status
        processing_status[job_id] = {
            'status': 'queued',
            'file': filename
        }

        return jsonify({
            'message': 'File uploaded successfully',
            'job_id': job_id
        }), 202

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    if job_id in processing_status:
        status = processing_status[job_id]
        if status['status'] == 'completed':
            # Read the result JSON
            result_path = status['result_path']
            with open(result_path, 'r') as f:
                result = json.load(f)
            return jsonify({
                'status': status['status'],
                'file': status['file'],
                'result': result
            })
        else:
            return jsonify(status)

    return jsonify({'error': 'Job not found'}), 404

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    return jsonify(processing_status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 2. Streamlit Frontend (`streamlit_app.py`)

```python
import streamlit as st
import requests
import pandas as pd
import json
import time
import os
from datetime import datetime

API_URL = "http://localhost:5000"

st.set_page_config(page_title="Invoice Parser", layout="wide")
st.title("Invoice Parser")

# Session state to store job information
if 'jobs' not in st.session_state:
    st.session_state.jobs = {}

# File uploader component
st.subheader("Upload Invoice")
uploaded_files = st.file_uploader("Choose invoice files (PDF or Image)",
                                 type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
                                 accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Invoices"):
        for uploaded_file in uploaded_files:
            # Create a form data object
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}

            # Send the file to the API
            response = requests.post(f"{API_URL}/api/upload-invoice", files=files)

            if response.status_code == 202:
                data = response.json()
                job_id = data['job_id']

                # Store job in session state
                st.session_state.jobs[job_id] = {
                    'file': uploaded_file.name,
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'queued',
                    'result': None
                }

                st.success(f"File '{uploaded_file.name}' uploaded and queued for processing!")
            else:
                st.error(f"Error uploading file '{uploaded_file.name}': {response.json().get('error', 'Unknown error')}")

# Display jobs and their status
st.subheader("Processing Status")

# Poll for updates on jobs
if st.session_state.jobs:
    for job_id, job_info in list(st.session_state.jobs.items()):
        if job_info['status'] not in ['completed', 'error']:
            # Check status from API
            response = requests.get(f"{API_URL}/api/status/{job_id}")
            if response.status_code == 200:
                status_data = response.json()
                job_status = status_data.get('status', 'unknown')

                # Update job info
                job_info['status'] = job_status

                # If job is complete, get the result
                if job_status == 'completed' and 'result' in status_data:
                    job_info['result'] = status_data['result']

    # Create a list for display
    job_display = []
    for job_id, job_info in st.session_state.jobs.items():
        job_display.append({
            'ID': job_id[:8] + '...',  # Shorten ID for display
            'File': job_info['file'],
            'Uploaded': job_info['time'],
            'Status': job_info['status'],
            'Actions': job_id  # We'll use this for buttons
        })

    # Display as a table
    if job_display:
        job_df = pd.DataFrame(job_display)
        st.dataframe(job_df.set_index('ID'), use_container_width=True)
    else:
        st.info("No jobs submitted yet.")

# Results section
st.subheader("Invoice Results")

# Select a completed job to view
completed_jobs = {job_id: job_info['file']
                  for job_id, job_info in st.session_state.jobs.items()
                  if job_info['status'] == 'completed'}

if completed_jobs:
    selected_job = st.selectbox("Select a processed invoice to view:",
                                list(completed_jobs.keys()),
                                format_func=lambda x: completed_jobs[x])

    if selected_job:
        job_info = st.session_state.jobs[selected_job]
        if job_info['result']:
            # Display the extracted information
            result = job_info['result']

            # Convert result to DataFrame for display
            result_df = pd.DataFrame([result])
            st.dataframe(result_df, use_container_width=True)

            # Add download button for JSON
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{job_info['file'].split('.')[0]}_extracted.json",
                mime="application/json"
            )
else:
    st.info("No completed jobs available yet.")
```

### 3. Invoice Processor with LangChain (`invoice_processor.py`)

````python
from langchain.llms import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
import base64
import os
import json
from datetime import datetime
from PIL import Image
import io
import fitz  # PyMuPDF for PDF handling
from typing import Dict, Any, Optional, List

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the vision model
vision_model = ChatOpenAI(
    model="gpt-4-vision-preview",  # Use GPT-4 with Vision capabilities
    api_key=OPENAI_API_KEY,
    max_tokens=4096,
    temperature=0.1  # Lower temperature for more factual responses
)

# System prompt for invoice extraction
SYSTEM_PROMPT = """You are an AI assistant that specializes in extracting information from invoices.
Your task is to analyze the given invoice and extract the following fields:

1. vendor_name: The company or entity that issued the invoice
2. due_date: When payment is required (in YYYY-MM-DD format)
3. paid_date: When the invoice was actually paid, if applicable (in YYYY-MM-DD format)
4. service_from: Start date of the service period (in YYYY-MM-DD format)
5. service_to: End date of the service period (in YYYY-MM-DD format)
6. currency: The currency used for the invoice (e.g., USD, EUR, GBP)
7. net_amount: The amount before tax/VAT (numerical value only)
8. vat_amount: The value-added tax or similar tax amount (numerical value only)
9. gross_amount: The total amount including taxes (numerical value only)

Respond with a JSON object containing these fields. If a field is not found, use null for its value.
Do not include any explanations, just the JSON object."""

# Human message template
HUMAN_PROMPT = """Please extract the required invoice information from this image:
{image_data}"""

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_images(pdf_path: str) -> List[bytes]:
    """
    Convert PDF pages to images
    """
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
        img_bytes = pix.pil_tobytes(format="JPEG")
        images.append(img_bytes)

    return images

def process_invoice(file_path: str) -> Dict[str, Any]:
    """
    Process an invoice file (PDF or image) and extract information using LangChain and a vision model
    """
    # Determine file type
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        # Handle PDF: Convert to images first
        images = pdf_to_images(file_path)

        # For now, we'll just process the first page
        # In a more advanced implementation, we might combine results from all pages
        image_data = images[0]

        # Convert bytes to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    else:
        # Handle image formats
        image_base64 = encode_image(file_path)

    # Create messages for the chat model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": HUMAN_PROMPT.format(image_data="")},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        )
    ]

    # Process with the vision model
    response = vision_model.invoke(messages)

    # Extract JSON from response
    try:
        # Sometimes the model might include ```json and ``` around the response
        response_text = response.content
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()

        result = json.loads(json_text)

        # Validate required fields
        for field in ["vendor_name", "due_date", "paid_date", "service_from",
                      "service_to", "currency", "net_amount", "vat_amount", "gross_amount"]:
            if field not in result:
                result[field] = None

        # Add metadata
        result["processed_at"] = datetime.now().isoformat()
        result["filename"] = os.path.basename(file_path)

        return result
    except json.JSONDecodeError as e:
        # If we can't parse the JSON, return an error
        return {
            "error": f"Failed to parse JSON from model response: {str(e)}",
            "raw_response": response.content,
            "processed_at": datetime.now().isoformat(),
            "filename": os.path.basename(file_path)
        }
    except Exception as e:
        # Handle any other exceptions
        return {
            "error": f"Error processing invoice: {str(e)}",
            "processed_at": datetime.now().isoformat(),
            "filename": os.path.basename(file_path)
        }

# Example usage if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing invoice: {file_path}")
        result = process_invoice(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Please provide an invoice file path")
````

## Installation and Setup Instructions

To set up the invoice parsing system, please include setup instructions similar to the following:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY=your_api_key_here  # On Windows, use: set OPENAI_API_KEY=your_api_key_here

# Run the Flask backend (in one terminal)
python app.py

# Run the Streamlit frontend (in another terminal)
streamlit run streamlit_app.py
```

And include a requirements.txt file with content similar to:

```
flask==2.2.3
flask-cors==3.0.10
streamlit==1.22.0
langchain==0.0.267
openai==0.27.8
pillow==9.5.0
pandas==2.0.2
pymupdf==1.22.5
python-dotenv==1.0.0
requests==2.31.0
```

## Expected Output

The final deliverable should include:

1. **Complete Source Code** divided into the following components:

   - Flask API server code (`app.py` or similar)
   - Streamlit frontend application (`streamlit_app.py` or similar)
   - LangChain components for invoice processing
   - Utility functions for file handling and data processing
   - Configuration files and environment templates

2. **Setup Instructions**

   - Step-by-step installation guide
   - Required Python packages (requirements.txt)
   - Environment variable configuration
   - Instructions for obtaining and configuring API keys

3. **Code Documentation**

   - Clear inline comments explaining code functionality
   - Function and class documentation
   - Architecture overview explaining how components interact

4. **Usage Examples**

   - Sample commands to run both the Flask server and Streamlit app
   - Example API requests for testing
   - Screenshots or descriptions of the expected UI flow

5. **Project Structure**
   - Logical organization of files and directories
   - Separation of concerns between frontend, backend, and processing logic
   - Modular design allowing for future enhancements

Please provide code that demonstrates a complete working system that can be run locally for development and testing purposes. The code should be production-ready but focused on delivering the core functionality without unnecessary complexity.
