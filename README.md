# Invoice Parser

An AI-powered application that extracts structured data from invoice documents using LangChain and vision-capable language models.

## Features

- Extract key information from invoice PDF or image files
- Process multiple invoices in a queue
- View results with thumbnails in a user-friendly interface
- Download individual or consolidated results (JSON/CSV)
- Clear all data with a single click

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

## Quick Start

To start the invoice parser application, simply run:

```bash
./start.sh
```

This will:

1. Set up a Python virtual environment in the root directory (if needed)
2. Install all required dependencies
3. Start the Flask backend server on port 5002
4. Start the Streamlit frontend on port 8501
5. Create log files for both services in the logs directory

The application will be available at: http://localhost:8501

Press Ctrl+C to stop both services when you're done.

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
