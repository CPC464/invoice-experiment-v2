# Invoice Parser

An AI-powered application that extracts structured data from invoice documents using LangChain and vision-capable language models.

## Features

- Extract key information from invoice PDF or image files
- Process multiple invoices in a queue
- View results with thumbnails in a user-friendly interface
- Download individual or consolidated results (JSON/CSV)
- Clear all data with a single click
- Duplicate invoice detection using vector embeddings and OCR

## Extracted Information

The application extracts the following data from invoices:

- Vendor Name
- Due Date
- Paid Date
- Service Period (From/To)
- Currency
- Net Amount
- VAT/Tax Amount
- Gross Amount

## Requirements

### API Keys

- OpenAI API key (required for invoice parsing)

### Software Dependencies

- Python 3.8+ (3.12 recommended)
- Tesseract OCR (required for duplicate detection with image-based files)
- FAISS (Facebook AI Similarity Search) for vector similarity search
- sentence-transformers for generating document embeddings

#### Installing Tesseract OCR

For the duplicate detection feature to work with image files and scanned PDFs, you need:

1. **Tesseract OCR** (system binary installed on your machine)
2. **pytesseract** (Python package that interfaces with Tesseract - automatically installed by our script)

**Installing the Tesseract OCR system binary:**

**Ubuntu/Debian:**

```bash
sudo apt install tesseract-ocr
```

**macOS:**

```bash
brew install tesseract
```

**Windows:**

1. Download and install from https://github.com/UB-Mannheim/tesseract/wiki
2. Ensure the installation path (e.g., `C:\Program Files\Tesseract-OCR`) is added to your PATH environment variable

You can verify your Tesseract installation by running:

```bash
tesseract --version
```

**Note:** The startup script will automatically install the Python `pytesseract` package, but it requires the Tesseract OCR binary to be installed on your system and available in your PATH.

#### Note for FAISS on Apple Silicon (M1/M2) Macs

If you're using an Apple Silicon Mac and encounter issues with FAISS installation, try:

```bash
pip install faiss-cpu --no-cache-dir
```

Or install using conda:

```bash
conda install -c conda-forge faiss
```

## Quick Start

1. Create an .env.local file in the /invoice-parser dir, copy the contents of .env.example to this file, and insert your Open AI API key (The anthropic key is not used yet)

2. Start the application by running:

```bash
./start.sh
```

This will:

1. Set up a Python virtual environment in the root directory (if needed)
2. Install all required dependencies
3. Check for Tesseract OCR (will warn if not installed)
4. Start the Flask backend server on port 5002
5. Start the Streamlit frontend on port 8501
6. Create a log file for the steamlit service in the logs directory. The flask app logs directly to the console

Press Ctrl+C to stop both services when you're done.

## Configuration

The application configuration is stored in `invoice-parser/.env.local`. Key settings include:

- `PORT`: The port for the Flask backend (default: 5002)
- `OPENAI_API_KEY`: Your OpenAI API key for invoice parsing
- `LLM_PROVIDER`: The LLM provider to use (openai or anthropic)
- `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4.1-mini)

The cleanup script uses the port value from `.env.local` to find and terminate any processes using that port.

## Project Structure

```
invoice-experiment/
├── start.sh                # Single command to start everything
├── run.sh                  # Alternative run script
├── venv/                   # Python virtual environment (created by start.sh)
├── invoice-parser/         # Application directory
    ├── app.py              # Flask backend API
    ├── streamlit_app.py    # Streamlit frontend
    ├── invoice_processor.py # Core processing logic
    ├── uploads/            # Uploaded invoice files
    ├── results/            # Processing results
    ├── thumbnails/         # Invoice thumbnails
    ├── logs/               # Application logs
    └── requirements.txt    # Project dependencies
```

## Manual Setup

If you prefer to set up manually:

1. Create a virtual environment in the root directory:

   ```bash
   # From the project root
   python -m venv venv
   ```

2. Activate the virtual environment:

   - On macOS/Linux:
     ```bash
     # From the project root
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     # From the project root
     venv\Scripts\activate
     ```

3. Install dependencies:

   ```bash
   # From the invoice-parser directory
   pip install -r invoice-parser/requirements.txt
   ```

4. Start the Flask backend:

   ```bash
   # From the invoice-parser directory
   cd invoice-parser && python app.py
   ```

5. In a new terminal, start the Streamlit frontend:
   ```bash
   # From the invoice-parser directory
   cd invoice-parser && streamlit run streamlit_app.py
   ```

## Usage

1. Navigate to http://localhost:8501 in your web browser
2. Upload invoice documents (PDF, PNG, JPG, JPEG, TIFF)
3. Files are automatically processed in the background
4. Use the "Refresh Status" button to update processing status
5. View and download results once processing is complete

## API Endpoints

The backend API is available at http://localhost:5002 with the following endpoints:

- POST `/api/upload-invoice`: Upload and process an invoice
- GET `/api/status/<job_id>`: Check the status of a specific job
- GET `/api/jobs`: List all jobs and their statuses
- GET `/api/all-results`: Get all processed results
- POST `/api/clear-all`: Clear all data (uploads, thumbnails, and results)

## License

This project is licensed under the MIT License.
