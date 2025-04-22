import streamlit as st
import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from invoice_processor import required_fields
import uuid
import hashlib

# Load environment variables
load_dotenv(dotenv_path=".env.local")  # Updated to use .env.local instead of .env

# Configuration
API_URL = f"http://localhost:{os.getenv('PORT')}"

# Page setup
st.set_page_config(page_title="Invoice Parser", page_icon="��", layout="wide")

# Create tabs for main sections
tab1, tab2 = st.tabs(["Invoice Upload & Processing", "Duplicate Detection Stats"])

with tab1:
    # Main invoice processing UI
    with st.container():

        # Header
        st.title("🔍 Invoice Parser")
        st.markdown(
            """
        This application extracts key information from invoice documents.
        Upload a PDF or image of an invoice to get started.
        
        **How It Works:**
        1. When you upload invoices, we automatically check for duplicates
        2. Unique invoices are processed by AI to extract structured data
        3. Duplicate invoices are tagged and skipped to avoid redundant processing
        """
        )

        # Initialize session state
        if "jobs" not in st.session_state:
            st.session_state.jobs = {}
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "processed_file_hashes" not in st.session_state:
            st.session_state.processed_file_hashes = (
                set()
            )  # To track exact same file content
        if "clear_uploader" not in st.session_state:
            st.session_state.clear_uploader = False
        if "initialized" not in st.session_state:
            st.session_state.initialized = False
        if "refreshing" not in st.session_state:
            st.session_state.refreshing = False
        if "last_uploaded_files" not in st.session_state:
            st.session_state.last_uploaded_files = []

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
                            # st.session_state.processed_files.add(file_name)

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

                        # If job is complete or skipped, get the thumbnail
                        if job_status in ["completed", "skipped"]:
                            # Update result if available
                            if "result" in status_data:
                                job_info["result"] = status_data["result"]

                            # Update thumbnail if available
                            if "thumbnail_path" in status_data:
                                job_info["thumbnail"] = (
                                    f"{API_URL}/thumbnails/{os.path.basename(status_data['thumbnail_path'])}"
                                )

                            # Update duplicate info if available
                            if "duplicate_info" in status_data:
                                job_info["duplicate_info"] = status_data[
                                    "duplicate_info"
                                ]
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
            print(f"Processing file: {uploaded_file.name}")  # Debug logging

            # Get file content for duplicate detection
            file_content = uploaded_file.getvalue()

            # Create form data with the file
            files = {"file": (uploaded_file.name, file_content)}

            try:
                # First check for duplicates - always do this check regardless of hash
                print("Checking for duplicates...")
                duplicate_response = requests.post(
                    f"{API_URL}/api/check-duplicate", files=files
                )

                # Track if this is a duplicate
                is_duplicate = False
                duplicate_info = None
                similarity_score = None

                if duplicate_response.status_code == 200:
                    check_result = duplicate_response.json()
                    is_duplicate = check_result.get("is_duplicate", False)
                    similarity_score = check_result.get("highest_score", 0)
                    duplicate_info = check_result
                    print(
                        f"Duplicate check result: is_duplicate={is_duplicate}, score={similarity_score}%"
                    )

                # Always upload the file but include a flag if it's a duplicate
                print(f"Uploading file with is_duplicate={is_duplicate}")

                # Convert boolean to string for form data
                is_duplicate_str = "true" if is_duplicate else "false"

                # Try to get additional fields from duplicate_info for better debugging
                related_docs = []
                if duplicate_info and "similar_documents" in duplicate_info:
                    related_docs = [
                        doc.get("document_id", "unknown")
                        for doc in duplicate_info.get("similar_documents", [])
                    ]

                if is_duplicate:
                    print(
                        f"File is a duplicate with score {similarity_score}%. Related documents: {related_docs}"
                    )

                # Always upload to store the file
                response = requests.post(
                    f"{API_URL}/api/upload-invoice",
                    files=files,
                    data={"is_duplicate": is_duplicate_str},
                )

                if response.status_code == 202:
                    data = response.json()
                    job_id = data["job_id"]
                    print(
                        f"File uploaded with job_id: {job_id}, API status: {data.get('status', 'unknown')}"
                    )

                    # Create appropriate status message based on duplicate status
                    status_message = "skipped" if is_duplicate else "queued"
                    status_display = f"File '{uploaded_file.name}' uploaded and marked as {status_message}."

                    if is_duplicate:
                        status_display += (
                            f" (Duplicate similarity: {similarity_score}%)"
                        )

                    # Store job in session state
                    st.session_state.jobs[job_id] = {
                        "file": uploaded_file.name,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": status_message,
                        "result": None,
                        "thumbnail": None,
                        "similarity_score": similarity_score,
                        "duplicate_info": duplicate_info if is_duplicate else None,
                    }

                    # Add thumbnail path if returned in response
                    if "thumbnail_path" in data:
                        thumbnail_path = data["thumbnail_path"]
                        st.session_state.jobs[job_id][
                            "thumbnail"
                        ] = f"{API_URL}/thumbnails/{os.path.basename(thumbnail_path)}"
                        print(f"Thumbnail added: {thumbnail_path}")

                    # Don't add to processed_file_hashes to allow re-uploads
                    # We'll rely on duplicate detection instead

                    # Show appropriate message
                    if is_duplicate:
                        st.warning(status_display)
                    else:
                        st.success(status_display)
                    return True
                else:
                    print(
                        f"Error response from API: {response.status_code} - {response.text}"
                    )
                    st.error(
                        f"Error uploading '{uploaded_file.name}': {response.json().get('error', 'Unknown error')}"
                    )
                    return False
            except Exception as e:
                print(f"Exception in process_uploaded_file: {str(e)}")
                st.error(f"Connection error: {str(e)}")
                return False

        # File uploader section
        with (
            st.container(border=True)
            if hasattr(st, "container")
            and "border" in st.container.__code__.co_varnames
            else st.container()
        ):
            # Check if we need to clear the uploader
            if st.session_state.clear_uploader:
                # Reset the flag
                st.session_state.clear_uploader = False
                # The empty uploader will be shown automatically

            # Track if we're in a refresh operation
            is_refreshing = (
                "refreshing" in st.session_state and st.session_state.refreshing
            )

            # Only enable uploading when not refreshing
            if not is_refreshing:
                uploaded_files = st.file_uploader(
                    "Upload your invoices here",
                    type=["pdf", "png", "jpg", "jpeg", "tiff"],
                    accept_multiple_files=True,
                    key="file_uploader",
                )

                # Check if uploaded_files has changed from last time
                if uploaded_files and (
                    len(uploaded_files) != len(st.session_state.last_uploaded_files)
                    or any(
                        file.name
                        not in [f.name for f in st.session_state.last_uploaded_files]
                        for file in uploaded_files
                    )
                ):
                    # Files have changed, process them
                    processed_any = False
                    # Remember this set of files to avoid reprocessing
                    st.session_state.last_uploaded_files = uploaded_files.copy()

                    with st.spinner("Uploading and processing files..."):
                        for uploaded_file in uploaded_files:
                            # Get file hash to detect exact duplicates in memory
                            file_content = uploaded_file.getvalue()
                            file_hash = hashlib.md5(file_content).hexdigest()

                            # Skip if this exact file content was already processed in this session
                            if file_hash in st.session_state.processed_file_hashes:
                                continue

                            # Process the file
                            if process_uploaded_file(uploaded_file):
                                processed_any = True
                                # Add to processed hashes
                                st.session_state.processed_file_hashes.add(file_hash)

                    # Only clear the uploader and rerun if files were processed successfully
                    # This prevents infinite loops
                    if processed_any:
                        # Set a flag to clear uploader on next run but don't rerun immediately
                        st.session_state.clear_uploader = True
            else:
                # During refresh, show a placeholder message instead of the uploader
                st.info(
                    "File uploader temporarily disabled during refresh. Please wait..."
                )

        # Job status section
        st.subheader("Processing Status")

        # Manual refresh button
        refresh_col1, refresh_col2, debug_col = st.columns([1, 8, 1])
        with refresh_col1:
            refresh_button = st.button("🔄 Refresh")

        with refresh_col2:
            # Show last refresh time
            if "last_refresh_time" not in st.session_state:
                st.session_state.last_refresh_time = datetime.now().strftime("%H:%M:%S")

            st.write(f"Last refreshed: {st.session_state.last_refresh_time}")

        # Add debug button
        with debug_col:
            debug_button = st.button("🔍 Debug")

        # Handle refresh button
        if refresh_button:
            # Set refreshing flag to prevent uploading files during refresh
            st.session_state.refreshing = True

            with st.spinner("Refreshing job statuses..."):
                # Load any new jobs first
                load_existing_jobs()
                # Then update all job statuses
                update_job_statuses()
                # Update last refresh time
                st.session_state.last_refresh_time = datetime.now().strftime("%H:%M:%S")
                # Clear the refreshing flag
                st.session_state.refreshing = False

            # Tell the user refresh is complete
            st.success("Refresh complete!")
            # Use a short auto-clearing notification
            time.sleep(0.5)

        # Handle debug button - forces API refresh and show results
        if debug_button:
            st.info("Running debug checks...")

            # Direct API call without caching
            try:
                response = requests.get(f"{API_URL}/api/jobs")
                if response.status_code == 200:
                    api_jobs = response.json()

                    # Display raw API response in an expandable section
                    with st.expander("Raw API Response"):
                        st.json(api_jobs)

                    # Count job types
                    job_counts = {}
                    for job_id, job_info in api_jobs.items():
                        status = job_info.get("status", "unknown")
                        if status not in job_counts:
                            job_counts[status] = 0
                        job_counts[status] += 1

                    # Show job type counts
                    st.write("### Job Status Counts")
                    for status, count in job_counts.items():
                        st.write(f"- {status.capitalize()}: {count}")

                    # Highlight any skipped jobs
                    skipped_jobs = {
                        job_id: job_info
                        for job_id, job_info in api_jobs.items()
                        if job_info.get("status") == "skipped"
                    }

                    if skipped_jobs:
                        st.write("### Skipped Jobs Details")
                        for job_id, job_info in skipped_jobs.items():
                            st.write(
                                f"- {job_id} ({job_info.get('original_filename', 'Unknown')})"
                            )
                    else:
                        st.warning("No skipped jobs found in API response")

                    # Update session state with these jobs
                    st.session_state.jobs = api_jobs

                    # Force re-render after updating session state
                    st.rerun()
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"API Connection Error: {str(e)}")

            # Update last refresh time
            st.session_state.last_refresh_time = datetime.now().strftime("%H:%M:%S")

        # Display job statuses
        if True:  # Always try to display jobs, even if session state is empty
            # Directly fetch all jobs from API to ensure we have everything
            try:
                response = requests.get(f"{API_URL}/api/jobs")
                if response.status_code == 200:
                    all_jobs = response.json()
                    print(f"Fetched {len(all_jobs)} jobs from API")  # Debug logging

                    # Check if we received any skipped jobs
                    skipped_count = sum(
                        1
                        for job_info in all_jobs.values()
                        if job_info.get("status") == "skipped"
                    )
                    print(
                        f"Found {skipped_count} jobs with 'skipped' status"
                    )  # Debug logging

                    # Update session state with jobs
                    for job_id, job_info in all_jobs.items():
                        # Print debug info for each job
                        print(
                            f"API job: {job_id} - Status: {job_info.get('status', 'unknown')}"
                        )

                        if job_id not in st.session_state.jobs:
                            st.session_state.jobs[job_id] = job_info
                            print(f"Added new job to session state: {job_id}")
                        else:
                            # Update status and other critical fields
                            prev_status = st.session_state.jobs[job_id].get(
                                "status", "unknown"
                            )
                            new_status = job_info.get("status", prev_status)
                            st.session_state.jobs[job_id]["status"] = new_status
                            print(
                                f"Updated job status: {job_id} - {prev_status} → {new_status}"
                            )

                        # Update thumbnail path if available
                        if "thumbnail_path" in job_info:
                            st.session_state.jobs[job_id][
                                "thumbnail"
                            ] = f"{API_URL}/thumbnails/{os.path.basename(job_info['thumbnail_path'])}"
            except Exception as e:
                st.error(f"Error fetching jobs: {str(e)}")
                print(f"Error in API call: {str(e)}")  # Debug logging

            # Create a data frame for display
            job_display = []

            # Print debug info about session state
            print(f"Session state has {len(st.session_state.jobs)} jobs")
            for job_id, job_info in st.session_state.jobs.items():
                print(
                    f"Session job: {job_id} - Status: {job_info.get('status', 'unknown')}"
                )

            # Sort jobs by time, most recent first
            sorted_jobs = sorted(
                st.session_state.jobs.items(),
                key=lambda x: x[1].get("time", ""),
                reverse=True,
            )

            for job_id, job_info in sorted_jobs:
                # Double-check status to make sure we display skipped jobs
                status = job_info.get("status", "unknown")
                print(f"Processing job for display: {job_id} - Status: {status}")

                # Get token and cost info if available
                token_info = ""
                cost_info = ""
                time_info = ""
                completed_fields = f"0/{len(required_fields)}"  # Default value
                similarity_score = job_info.get(
                    "similarity_score", ""
                )  # Get similarity score

                if similarity_score:
                    similarity_score = f"{similarity_score}%"  # Format as percentage

                # For skipped status, get similarity score from duplicate_info if available
                if status == "skipped":
                    print(f"Formatting skipped job: {job_id}")
                    if (
                        "duplicate_info" in job_info
                        and isinstance(job_info["duplicate_info"], dict)
                        and "highest_score" in job_info["duplicate_info"]
                    ):
                        similarity_score = (
                            f"{job_info['duplicate_info']['highest_score']}%"
                        )
                    # Set other metrics to None for skipped jobs
                    token_info = "None"
                    cost_info = "None"
                    time_info = "None"
                    completed_fields = "None"
                elif status == "completed" and job_info.get("result"):
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

                    # Check for duplicate info in results
                    if (
                        "duplicate_check" in result
                        and result["duplicate_check"].get("highest_score") is not None
                    ):
                        similarity_score = (
                            f"{result['duplicate_check'].get('highest_score')}%"
                        )

                # Get better filename display
                display_filename = job_info.get(
                    "original_filename", job_info.get("file", "Unknown")
                )

                # Check if this job is the selected invoice
                is_selected = False
                if (
                    "selected_invoice_id" in st.session_state
                    and job_id == st.session_state.get("selected_invoice_id")
                ):
                    is_selected = True

                # Add color coding for status
                status_display = status.capitalize()
                if status.lower() == "completed":
                    status_display = "✅ " + status_display
                elif status.lower() == "skipped":
                    status_display = "⚠️ " + status_display
                elif status.lower() == "error":
                    status_display = "❌ " + status_display
                elif status.lower() == "queued":
                    status_display = "⏳ " + status_display
                elif status.lower() == "processing":
                    status_display = "🔄 " + status_display

                # Add the job to the display
                job_item = {
                    "ID": job_id[:8] + "...",  # Truncate ID for display
                    "File": display_filename,
                    "Uploaded": job_info.get("time", ""),
                    "Status": status_display,
                    "Similarity": similarity_score,  # Add similarity score column
                    "Token Usage": token_info,
                    "Cost ($ cents)": cost_info,
                    "Processing Time": time_info,
                    "Completed Fields": completed_fields,  # Add the completed fields count
                }

                # Add job to display list
                job_display.append(job_item)
                print(f"Added job to display: {job_id} - Status: {status_display}")

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

        # Get all completed jobs and skipped jobs (with thumbnails)
        displayable_jobs = {
            job_id: job_info
            for job_id, job_info in st.session_state.jobs.items()
            if job_info["status"] in ["completed", "skipped"]
            and job_info.get("thumbnail")
        }

        if displayable_jobs:
            # Initialize selection state if not exists
            if (
                "selected_invoice_id" not in st.session_state
                or st.session_state.selected_invoice_id not in displayable_jobs
            ):
                # Default to the first displayable job
                st.session_state.selected_invoice_id = list(displayable_jobs.keys())[0]

            # Get all job IDs in a fixed order for consistent navigation
            all_job_ids = list(displayable_jobs.keys())

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
                    "◀ Previous",
                    disabled=prev_disabled,
                    key="prev_button",
                    type="primary",
                ):
                    # Move to previous invoice
                    st.session_state.selected_invoice_id = all_job_ids[
                        current_index - 1
                    ]
                    st.rerun()

            # Next button
            with button_cols[2]:
                next_disabled = current_index == len(all_job_ids) - 1
                if st.button(
                    "Next ▶", disabled=next_disabled, key="next_button", type="primary"
                ):
                    # Move to next invoice
                    st.session_state.selected_invoice_id = all_job_ids[
                        current_index + 1
                    ]
                    st.rerun()

            # Get the selected job info
            selected_job_id = st.session_state.selected_invoice_id
            job_info = displayable_jobs[selected_job_id]

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
                st.markdown("### File Information")

                # Show status with appropriate formatting
                status = job_info.get("status", "unknown")
                if status == "completed":
                    st.markdown("**Status:** ✅ Completed")
                elif status == "skipped":
                    st.markdown("**Status:** ⚠️ Skipped (Duplicate)")
                else:
                    st.markdown(f"**Status:** {status.capitalize()}")

                st.markdown(f"**Filename:** {job_info.get('file', 'Unknown')}")
                st.markdown(f"**Uploaded:** {job_info.get('time', 'Unknown')}")

                # Show duplicate info if available for skipped files
                if status == "skipped" and job_info.get("duplicate_info"):
                    st.markdown("---")
                    st.markdown("### Duplicate Information")
                    dup_info = job_info["duplicate_info"]
                    st.markdown(f"**Similarity:** {dup_info.get('highest_score', 0)}%")

                    # Show similar document information if available
                    if (
                        "similar_documents" in dup_info
                        and dup_info["similar_documents"]
                    ):
                        st.markdown("**Similar documents:**")
                        for doc in dup_info["similar_documents"]:
                            st.markdown(
                                f"- {doc.get('document_id', 'Unknown')}: {doc.get('similarity_score', 0)}% ({doc.get('match_type', 'unknown')})"
                            )

                # Only show extracted data for completed jobs
                if status == "completed" and job_info.get("result"):
                    result = job_info["result"]

                    st.markdown("---")
                    st.markdown("### Extracted Data")

                    # Vendor information
                    st.markdown(f"**Vendor:** {result.get('vendor_name', 'N/A')}")
                    st.markdown(
                        f"**Document Type(s):** {', '.join(t for t in result.get('document_type', 'N/A'))}"
                    )
                    st.markdown(
                        f"**Document Number:** {result.get('document_number', 'N/A')}"
                    )
                    st.markdown(f"**Issue Date:** {result.get('issue_date', 'N/A')}")
                    st.markdown("---")

                    # Date information
                    st.markdown(f"**Due Date:** {result.get('due_date', 'N/A')}")
                    st.markdown(f"**Paid Date:** {result.get('paid_date', 'N/A')}")
                    st.markdown(
                        f"**Service From:** {result.get('service_from', 'N/A')}"
                    )
                    st.markdown(f"**Service To:** {result.get('service_to', 'N/A')}")
                    st.markdown("---")

                    # Currency
                    st.markdown(f"**Currency:** {result.get('currency', 'N/A')}")

                    # Amount information
                    st.markdown(f"**Net Amount:** {result.get('net_amount', 'N/A')}")
                    st.markdown(f"**VAT Amount:** {result.get('vat_amount', 'N/A')}")
                    st.markdown(
                        f"**Gross Amount:** {result.get('gross_amount', 'N/A')}"
                    )
                    st.markdown("---")

                    # Processing metrics
                    st.markdown("### Processing Metrics")

                    if (
                        "processing_time" in result
                        and result["processing_time"] is not None
                    ):
                        st.markdown(
                            f"**Processing Time:** {result['processing_time']} ms"
                        )

                    if "total_tokens" in result and result["total_tokens"] is not None:
                        st.markdown(
                            f"**Token Usage:** {result['total_tokens']:,} tokens"
                        )

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
                                "original_filename": result.get(
                                    "original_filename", ""
                                ),
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
                                "completion_tokens": result.get(
                                    "completion_tokens", ""
                                ),
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

with tab2:
    # Duplicate Detection Statistics tab (renamed to clarify purpose)
    with st.container():
        st.header("Duplicate Detection System")

        # Fetch duplicate detection stats
        try:
            stats_response = requests.get(f"{API_URL}/api/duplicate-stats")
            if stats_response.status_code == 200:
                stats_data = stats_response.json()

                # Display stats in a dashboard
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Documents Indexed", stats_data.get("num_documents", 0))

                with col2:
                    model_name = stats_data.get("model_info", {}).get("name", "Unknown")
                    st.metric("Embedding Model", model_name)

                with col3:
                    vector_dim = stats_data.get("model_info", {}).get(
                        "vector_dimension", 0
                    )
                    st.metric("Vector Dimension", vector_dim)

                # Similarity thresholds
                st.subheader("Similarity Thresholds")

                threshold_info = stats_data.get("threshold_info", {})
                col1, col2 = st.columns(2)

                with col1:
                    identical_threshold = threshold_info.get("identical_threshold", 95)
                    st.write(f"Duplicate Threshold: {identical_threshold}%")
                    st.progress(identical_threshold / 100)

                with col2:
                    related_threshold = threshold_info.get("related_threshold", 70)
                    st.write(f"Related Document Threshold: {related_threshold}%")
                    st.progress(related_threshold / 100)

                # How it works
                st.subheader("How Duplicate Detection Works")
                st.write(
                    """
                The duplicate detection system uses vector embeddings to identify similar invoices:
                
                1. Each document is converted to a vector representation using a language model
                2. New documents are compared against previously processed documents
                3. Similarity scores (0-100) indicate how closely documents match:
                   - 95-100: Duplicate documents (likely the same invoice)
                   - 70-94: Related documents (e.g., invoice and receipt for same transaction)
                   - 0-69: Distinct documents
                
                This helps prevent duplicate payments and organizes related transaction documents.
                """
                )

                # Refresh button for skipped jobs
                dup_refresh_col1, dup_refresh_col2 = st.columns([1, 9])

                with dup_refresh_col1:
                    if st.button("🔄 Refresh", key="refresh_duplicates"):
                        # Set refreshing flag to prevent uploading files during refresh
                        st.session_state.refreshing = True

                        # Update job statuses
                        update_job_statuses()
                        # Update refresh time
                        st.session_state.dup_last_refresh = datetime.now().strftime(
                            "%H:%M:%S"
                        )

                        # Clear the refreshing flag
                        st.session_state.refreshing = False

                        # Notify user
                        st.success("Refresh complete!")
                        # Brief pause for notification
                        time.sleep(0.5)

                with dup_refresh_col2:
                    # Show last refresh time
                    if "dup_last_refresh" not in st.session_state:
                        st.session_state.dup_last_refresh = datetime.now().strftime(
                            "%H:%M:%S"
                        )

                    st.write(f"Last refreshed: {st.session_state.dup_last_refresh}")

                # Display duplicates status
                st.subheader("Detected Duplicates")

                # Fetch all jobs to ensure we have latest status
                try:
                    jobs_response = requests.get(f"{API_URL}/api/jobs")
                    if jobs_response.status_code == 200:
                        all_jobs = jobs_response.json()
                        # Update session state with latest job info, especially skipped jobs
                        for job_id, job_data in all_jobs.items():
                            if job_data.get("status") == "skipped":
                                # For skipped jobs, make sure we have them in the session state
                                if job_id not in st.session_state.jobs:
                                    st.session_state.jobs[job_id] = job_data
                                else:
                                    # Update status of existing job if it's skipped
                                    st.session_state.jobs[job_id].update(job_data)
                except Exception as e:
                    st.error(f"Error fetching latest jobs: {str(e)}")

                # Find all jobs with skipped status (duplicates)
                skipped_jobs = {
                    job_id: job_info
                    for job_id, job_info in st.session_state.jobs.items()
                    if job_info.get("status") == "skipped"
                }

                if skipped_jobs:
                    # Display list of skipped jobs
                    skipped_data = []
                    for job_id, job_info in skipped_jobs.items():
                        similarity_score = None

                        # Try to get similarity score from various places
                        if "similarity_score" in job_info:
                            similarity_score = job_info["similarity_score"]
                        elif (
                            "duplicate_info" in job_info
                            and isinstance(job_info["duplicate_info"], dict)
                            and "highest_score" in job_info["duplicate_info"]
                        ):
                            similarity_score = job_info["duplicate_info"][
                                "highest_score"
                            ]

                        # Format similarity score
                        similarity = "N/A"
                        if similarity_score is not None:
                            similarity = f"{similarity_score}%"

                        skipped_data.append(
                            {
                                "ID": job_id[:8] + "...",
                                "File": job_info.get(
                                    "original_filename", job_info.get("file", "Unknown")
                                ),
                                "Uploaded": job_info.get("time", ""),
                                "Similarity": similarity,
                            }
                        )

                    # Sort by upload time (newest first)
                    skipped_data.sort(key=lambda x: x.get("Uploaded", ""), reverse=True)

                    # Display as dataframe
                    if skipped_data:
                        skipped_df = pd.DataFrame(skipped_data)
                        st.dataframe(skipped_df, use_container_width=True)

                    # Show thumbnails for skipped files
                    st.subheader("Duplicate Invoice Thumbnails")

                    # Get all original documents from completed jobs
                    original_jobs = {
                        job_id: job_info
                        for job_id, job_info in st.session_state.jobs.items()
                        if job_info.get("status") == "completed"
                    }

                    # Get all skipped/duplicate jobs
                    dup_jobs = {
                        job_id: job_info
                        for job_id, job_info in st.session_state.jobs.items()
                        if job_info.get("status") == "skipped"
                    }

                    # Map originals to their duplicates based on duplicate_info
                    originals_with_dups = {}

                    # First identify all duplicates and which original they match
                    for dup_id, dup_info in dup_jobs.items():
                        if "duplicate_info" in dup_info and isinstance(
                            dup_info["duplicate_info"], dict
                        ):
                            # Get similar documents from duplicate info
                            similar_docs = dup_info["duplicate_info"].get(
                                "similar_documents", []
                            )
                            for doc in similar_docs:
                                original_id = doc.get("document_id")
                                if original_id:
                                    # Found an original this duplicate matches
                                    if original_id not in originals_with_dups:
                                        originals_with_dups[original_id] = []
                                    # Add this duplicate to the list for this original
                                    originals_with_dups[original_id].append(dup_id)

                    # Add any originals that don't have duplicates
                    for orig_id in original_jobs:
                        if orig_id not in originals_with_dups:
                            originals_with_dups[orig_id] = []

                    # Now display each original and its duplicates in a row
                    for orig_id, dup_ids in originals_with_dups.items():
                        # Create a row with original on left, duplicates on right
                        st.write("---")

                        # Get info for the original
                        if orig_id in st.session_state.jobs:
                            orig_info = st.session_state.jobs[orig_id]
                            orig_filename = orig_info.get(
                                "original_filename", orig_info.get("file", "Unknown")
                            )

                            # Create a row with columns
                            num_dups = len(dup_ids)
                            if num_dups > 0:
                                cols = st.columns([1] + [1] * min(num_dups, 3))
                            else:
                                cols = st.columns(
                                    [1, 3]
                                )  # Just the original with empty space

                            # Display original in first column
                            with cols[0]:
                                st.markdown(f"**Original:** {orig_filename}")
                                try:
                                    if (
                                        "thumbnail" in orig_info
                                        and orig_info["thumbnail"]
                                    ):
                                        st.image(
                                            orig_info["thumbnail"],
                                            use_container_width=True,
                                        )
                                    elif (
                                        "thumbnail_path" in orig_info
                                        and orig_info["thumbnail_path"]
                                    ):
                                        thumbnail_url = f"{API_URL}/thumbnails/{os.path.basename(orig_info['thumbnail_path'])}"
                                        st.image(
                                            thumbnail_url, use_container_width=True
                                        )
                                    else:
                                        st.write("No thumbnail available")
                                except Exception as e:
                                    st.error(f"Error displaying thumbnail: {str(e)}")

                            # Display duplicates in remaining columns
                            for i, dup_id in enumerate(
                                dup_ids[:3]
                            ):  # Limit to 3 duplicates per row
                                if i + 1 < len(cols):  # Check if column exists
                                    with cols[i + 1]:
                                        dup_info = st.session_state.jobs.get(dup_id, {})
                                        dup_filename = dup_info.get(
                                            "original_filename",
                                            dup_info.get("file", "Unknown"),
                                        )
                                        similarity = dup_info.get(
                                            "similarity_score", ""
                                        )
                                        if similarity:
                                            st.markdown(
                                                f"**Duplicate ({similarity}%):** {dup_filename}"
                                            )
                                        else:
                                            st.markdown(
                                                f"**Duplicate:** {dup_filename}"
                                            )

                                        try:
                                            if (
                                                "thumbnail" in dup_info
                                                and dup_info["thumbnail"]
                                            ):
                                                st.image(
                                                    dup_info["thumbnail"],
                                                    use_container_width=True,
                                                )
                                            elif (
                                                "thumbnail_path" in dup_info
                                                and dup_info["thumbnail_path"]
                                            ):
                                                thumbnail_url = f"{API_URL}/thumbnails/{os.path.basename(dup_info['thumbnail_path'])}"
                                                st.image(
                                                    thumbnail_url,
                                                    use_container_width=True,
                                                )
                                            else:
                                                st.write("No thumbnail available")
                                        except Exception as e:
                                            st.error(
                                                f"Error displaying thumbnail: {str(e)}"
                                            )

                            # If more than 3 duplicates, show a message
                            if len(dup_ids) > 3:
                                st.info(
                                    f"{len(dup_ids) - 3} more duplicate(s) not shown"
                                )
                        else:
                            st.warning(
                                f"Original document {orig_id} not found in session state"
                            )
                else:
                    st.info("No duplicate invoices have been detected yet.")
            else:
                st.error("Unable to fetch duplicate detection statistics")
        except Exception as e:
            st.error(f"Error connecting to duplicate detection API: {str(e)}")
