import streamlit as st
import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from invoice_processor import required_fields

# Load environment variables
load_dotenv(dotenv_path=".env.local")  # Updated to use .env.local instead of .env

# Configuration
API_URL = f"http://localhost:{os.getenv('PORT')}"

# Page setup
st.set_page_config(page_title="Invoice Parser", page_icon="🔍", layout="wide")

with st.container():

    # Header
    st.title("🔍 Invoice Parser")
    st.markdown(
        """
    This application extracts key information from invoice documents.
    Upload a PDF or image of an invoice to get started.
    """
    )

    # Initialize session state for tracking jobs and uploaded files
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "clear_uploader" not in st.session_state:
        st.session_state.clear_uploader = False
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Helper function to clear file uploader state
    def reset_file_uploader():
        # This will help reset the file uploader state
        if "file_uploader" in st.session_state:
            st.session_state.file_uploader = []
        st.session_state.clear_uploader = True

    def load_existing_jobs():
        """Load all existing jobs from the API and update session state"""
        try:
            # Fetch all jobs from the API
            response = requests.get(f"{API_URL}/api/jobs")

            if response.status_code == 200:
                jobs_data = response.json()

                # Update session state with existing jobs
                for job_id, job_data in jobs_data.items():
                    # Only add if not already in session state
                    if job_id not in st.session_state.jobs:
                        # Extract info directly from the jobs endpoint data
                        file_name = job_data.get(
                            "original_filename", job_data.get("file", "Unknown")
                        )

                        # Add job to session state
                        st.session_state.jobs[job_id] = {
                            "file": file_name,
                            "time": job_data.get(
                                "time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ),
                            "status": job_data.get("status", "unknown"),
                            "result": job_data.get("result"),
                            "thumbnail": None,
                        }

                        # Add filename to processed files to prevent re-upload
                        st.session_state.processed_files.add(file_name)

                        # Add thumbnail if available
                        if "thumbnail_path" in job_data:
                            st.session_state.jobs[job_id][
                                "thumbnail"
                            ] = f"{API_URL}/thumbnails/{os.path.basename(job_data['thumbnail_path'])}"

                # Print diagnostic info
                print(f"Loaded {len(jobs_data)} jobs from API")
                print(f"Session state now has {len(st.session_state.jobs)} jobs")
                print(f"Processed files: {st.session_state.processed_files}")

                return True
            return False
        except Exception as e:
            print(f"Error loading existing jobs: {str(e)}")
            return False

    def update_job_statuses():
        """Update the status of all jobs"""
        for job_id, job_info in list(st.session_state.jobs.items()):
            try:
                # Check status from API
                response = requests.get(f"{API_URL}/api/status/{job_id}")

                if response.status_code == 200:
                    status_data = response.json()
                    job_status = status_data.get("status", "unknown")

                    # Update job info
                    job_info["status"] = job_status

                    # If job is complete, get the result and thumbnail
                    if job_status == "completed":
                        if "result" in status_data:
                            job_info["result"] = status_data["result"]
                        if "thumbnail_path" in status_data:
                            job_info["thumbnail"] = (
                                f"{API_URL}/thumbnails/{os.path.basename(status_data['thumbnail_path'])}"
                            )
            except Exception as e:
                print(f"Could not update status for job {job_id}: {str(e)}")

    # Always load existing jobs at startup
    try:
        with st.spinner("Loading existing jobs..."):
            load_existing_jobs()
            update_job_statuses()
            st.session_state.initialized = True
    except Exception as e:
        print(f"Error in initial job loading: {str(e)}")

    def process_uploaded_file(uploaded_file):
        """Process a single uploaded file"""
        # Check if file was already processed to avoid duplicates
        if uploaded_file.name in st.session_state.processed_files:
            return False

        # Create form data with the file
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

        try:
            # Send the file to the API
            response = requests.post(f"{API_URL}/api/upload-invoice", files=files)

            if response.status_code == 202:
                data = response.json()
                job_id = data["job_id"]

                # Store job in session state
                st.session_state.jobs[job_id] = {
                    "file": uploaded_file.name,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "queued",
                    "result": None,
                    "thumbnail": None,
                }

                # Mark file as processed
                st.session_state.processed_files.add(uploaded_file.name)

                st.success(
                    f"File '{uploaded_file.name}' uploaded and queued for processing!"
                )
                return True
            else:
                st.error(
                    f"Error uploading '{uploaded_file.name}': {response.json().get('error', 'Unknown error')}"
                )
                return False
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return False

    # File uploader section
    with (
        st.container(border=True)
        if hasattr(st, "container") and "border" in st.container.__code__.co_varnames
        else st.container()
    ):
        # CSS to hide the file list that appears below the upload widget

        # Check if we need to clear the uploader
        if st.session_state.clear_uploader:
            # Reset the flag
            st.session_state.clear_uploader = False
            # The empty uploader will be shown automatically

        uploaded_files = st.file_uploader(
            "Upload your invoices here",
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        # Process files automatically when uploaded
        if uploaded_files:
            processed_any = False
            with st.spinner("Uploading and processing files..."):
                for uploaded_file in uploaded_files:
                    if process_uploaded_file(uploaded_file):
                        processed_any = True

            # Clear the uploader after processing
            if processed_any:
                st.rerun()

    # Job status section
    st.subheader("Processing Status")

    # Manual refresh button
    refresh_button = st.button("Refresh Status")
    if refresh_button:
        with st.spinner("Refreshing job statuses..."):
            # Load any new jobs first
            load_existing_jobs()
            # Then update all job statuses
            update_job_statuses()

    # Display job statuses
    if st.session_state.jobs:
        # Create a data frame for display
        job_display = []
        for job_id, job_info in st.session_state.jobs.items():
            # Get token and cost info if available
            token_info = ""
            cost_info = ""
            time_info = ""
            completed_fields = f"0/{len(required_fields)}"  # Default value

            if job_info["status"] == "completed" and job_info.get("result"):
                result = job_info["result"]
                if "total_tokens" in result and result["total_tokens"] is not None:
                    token_info = f"{result['total_tokens']:,} tokens"
                if "total_cost" in result and result["total_cost"] is not None:
                    cost_info = f"{result['total_cost']:.4f}"
                if (
                    "processing_time" in result
                    and result["processing_time"] is not None
                ):
                    time_info = f"{result['processing_time']} ms"

                # Use the existing completed_fields attribute from the results object
                if (
                    "completed_fields" in result
                    and result["completed_fields"] is not None
                ):
                    completed_fields = (
                        f"{result['completed_fields']}/{len(required_fields)}"
                    )

            # Check if this job is the selected invoice
            is_selected = False
            if (
                "selected_invoice_id" in st.session_state
                and job_id == st.session_state.get("selected_invoice_id")
            ):
                is_selected = True

            job_display.append(
                {
                    "ID": job_id[:8] + "...",  # Truncate ID for display
                    "File": job_info["file"],
                    "Uploaded": job_info["time"],
                    "Status": job_info["status"].capitalize(),
                    "Token Usage": token_info,
                    "Cost ($ cents)": cost_info,
                    "Processing Time": time_info,
                    "Completed Fields": completed_fields,  # Add the completed fields count
                }
            )

        # Remove _selected field if it exists (from previous attempts)
        for item in job_display:
            if "_selected" in item:
                del item["_selected"]

        # Display as a dataframe with highlighting
        if job_display:
            # Create a DataFrame for display
            display_df = pd.DataFrame(job_display)

            # Define a function to highlight the selected row
            def highlight_selected_row(x):
                # Create an empty DataFrame of same shape to hold styling
                df_style = pd.DataFrame("", index=x.index, columns=x.columns)

                # Check each row to find selected one
                for i, row in x.iterrows():
                    job_id_prefix = row["ID"].split("...")[0]  # Get ID prefix

                    # Find if this is the selected invoice
                    for full_job_id in st.session_state.jobs.keys():
                        if (
                            full_job_id.startswith(job_id_prefix)
                            and "selected_invoice_id" in st.session_state
                            and full_job_id
                            == st.session_state.get("selected_invoice_id")
                        ):
                            # Apply highlighting to entire row
                            df_style.loc[i, :] = (
                                "background-color: #f0f8ff; border-left: 3px solid #1e90ff;"
                            )

                return df_style

            # Apply styling
            styled_df = display_df.style.apply(highlight_selected_row, axis=None)

            # Display the styled DataFrame
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No jobs submitted yet.")
    else:
        st.info("No jobs submitted yet.")

    # Results section
    st.subheader("Invoice Results")

    # Get all completed jobs
    completed_jobs = {
        job_id: job_info
        for job_id, job_info in st.session_state.jobs.items()
        if job_info["status"] == "completed"
    }

    if completed_jobs:
        # Initialize selection state if not exists
        if "selected_invoice_id" not in st.session_state:
            # Default to the first completed job
            st.session_state.selected_invoice_id = list(completed_jobs.keys())[0]

        # Get all job IDs in a fixed order for consistent navigation
        all_job_ids = list(completed_jobs.keys())

        # Find current index in the list
        current_index = all_job_ids.index(st.session_state.selected_invoice_id)

        # Custom CSS for button styling
        st.markdown(
            """
        <style>
        .stButton > button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: auto;
            padding: 0.25rem 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Create a centered row with buttons
        button_cols = st.columns([2, 1, 1, 2])

        # Previous button
        with button_cols[1]:
            prev_disabled = current_index == 0
            if st.button(
                "◀ Previous", disabled=prev_disabled, key="prev_button", type="primary"
            ):
                # Move to previous invoice
                st.session_state.selected_invoice_id = all_job_ids[current_index - 1]
                st.rerun()

        # Next button
        with button_cols[2]:
            next_disabled = current_index == len(all_job_ids) - 1
            if st.button(
                "Next ▶", disabled=next_disabled, key="next_button", type="primary"
            ):
                # Move to next invoice
                st.session_state.selected_invoice_id = all_job_ids[current_index + 1]
                st.rerun()

        # Get the selected job info
        selected_job_id = st.session_state.selected_invoice_id
        job_info = completed_jobs[selected_job_id]
        result = job_info["result"]

        if result:

            # Create two columns - left for image, right for data
            col1, col2 = st.columns(
                [3, 2]
            )  # Adjusted ratio to give more space to the invoice image

            with col1:
                # Display thumbnail with larger size
                if job_info.get("thumbnail"):
                    st.image(
                        job_info["thumbnail"], use_container_width=True
                    )  # Using full container width for better visibility
                else:
                    st.info("No thumbnail available")

            with col2:
                # Display data in the requested order
                st.markdown("### Extracted Data")

                # Vendor information
                st.markdown(f"**Vendor:** {result.get('vendor_name', 'N/A')}")
                st.markdown(
                    f"**Document Type(s):** {', '.join(t for t in result.get('document_type', 'N/A'))}"
                )
                st.markdown("---")

                # Date information
                st.markdown(f"**Due Date:** {result.get('due_date', 'N/A')}")
                st.markdown(f"**Paid Date:** {result.get('paid_date', 'N/A')}")
                st.markdown(f"**Service From:** {result.get('service_from', 'N/A')}")
                st.markdown(f"**Service To:** {result.get('service_to', 'N/A')}")
                st.markdown("---")

                # Currency
                st.markdown(f"**Currency:** {result.get('currency', 'N/A')}")

                # Amount information
                st.markdown(f"**Net Amount:** {result.get('net_amount', 'N/A')}")
                st.markdown(f"**VAT Amount:** {result.get('vat_amount', 'N/A')}")
                st.markdown(f"**Gross Amount:** {result.get('gross_amount', 'N/A')}")
                st.markdown("---")

                # Processing metrics
                st.markdown("### Processing Metrics")

                if (
                    "processing_time" in result
                    and result["processing_time"] is not None
                ):
                    st.markdown(f"**Processing Time:** {result['processing_time']} ms")

                if "total_tokens" in result and result["total_tokens"] is not None:
                    st.markdown(f"**Token Usage:** {result['total_tokens']:,} tokens")

                if "total_cost" in result and result["total_cost"] is not None:
                    st.markdown(
                        f"**Cost:** ${result['total_cost']/100:.4f}"
                    )  # Convert cents to dollars
                if (
                    "completed_fields" in result
                    and result["completed_fields"] is not None
                ):
                    st.markdown(
                        f"**Completed Fields:** {result['completed_fields']}/{len(required_fields)}"
                    )
    else:
        st.info("No completed jobs available yet.")

    # Get all results in one file
    st.markdown("---")
    st.subheader("Download All Results")

    try:
        # Get consolidated results
        response = requests.get(f"{API_URL}/api/all-results")
        if response.status_code == 200:
            all_results = response.json()
            if all_results and "results" in all_results and all_results["results"]:
                # Enable download of consolidated results
                json_str = json.dumps(all_results, indent=2)
                st.download_button(
                    label="Download All Results (JSON)",
                    data=json_str,
                    file_name="all_invoice_results.json",
                    mime="application/json",
                    key="download_all_json",
                )

                # Create a CSV option
                if all_results["results"]:
                    # Flatten results for CSV
                    flat_results = []
                    for result in all_results["results"]:
                        flat_result = {
                            "original_filename": result.get("original_filename", ""),
                            "vendor_name": result.get("vendor_name", ""),
                            "currency": result.get("currency", ""),
                            "net_amount": result.get("net_amount", ""),
                            "vat_amount": result.get("vat_amount", ""),
                            "gross_amount": result.get("gross_amount", ""),
                            "due_date": result.get("due_date", ""),
                            "paid_date": result.get("paid_date", ""),
                            "service_from": result.get("service_from", ""),
                            "service_to": result.get("service_to", ""),
                            "processed_at": result.get("processed_at", ""),
                            # Add AI metrics
                            "processing_time_ms": result.get("processing_time", ""),
                            "prompt_tokens": result.get("prompt_tokens", ""),
                            "completion_tokens": result.get("completion_tokens", ""),
                            "total_tokens": result.get("total_tokens", ""),
                            "total_cost_cents": result.get("total_cost", ""),
                        }
                        flat_results.append(flat_result)

                    # Create DataFrame for CSV download
                    df = pd.DataFrame(flat_results)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download All Results (CSV)",
                        data=csv,
                        file_name="all_invoice_results.csv",
                        mime="text/csv",
                        key="download_all_csv",
                    )
            else:
                st.info("No results available for download yet.")
    except Exception as e:
        st.error(f"Could not retrieve all results: {str(e)}")

    # Display API status
    try:
        response = requests.get(f"{API_URL}/api/jobs")
        if response.status_code == 200:
            st.sidebar.success("✅ API is connected")
        else:
            st.sidebar.error("❌ API returned an error")
    except:
        st.sidebar.error("❌ API is not connected")

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown(
        """
    This application uses AI to extract structured data from invoice documents.
    
    **Supported Fields:**
    - Vendor Name
    - Due Date
    - Paid Date
    - Service Period
    - Currency
    - Net Amount
    - VAT Amount
    - Gross Amount
    """
    )
