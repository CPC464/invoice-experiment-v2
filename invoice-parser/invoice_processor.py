import os
import json
import base64
import io
import time
import logging
import textwrap
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF - the actual import name is 'fitz'

# Import LangChain components
from langchain_openai import ChatOpenAI  # Updated import for OpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

# Import utility functions
from preprocess_utils import (
    encode_image,
    pdf_to_images,
    generate_thumbnail,
)
from analytics_utils import (
    calculate_cost,
    extract_token_usage,
)
from logging_utils import log_model_response, PrettyJSONFormatter, setup_logger

# Import prompts
from prompts import prompt_a

# Import duplicate detector
from duplicate_detector import get_duplicate_detector

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Set prompts
SYSTEM_PROMPT = prompt_a["system_prompt"]
HUMAN_PROMPT = prompt_a["human_prompt"]

# Constants for LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Default to GPT-4.1 mini
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

# Check for API keys
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Determine the pricing based on the selected model
PROMPT_PRICE = None
COMPLETION_PRICE = None
if LLM_PROVIDER == "openai":
    if OPENAI_MODEL == "gpt-4o":
        PROMPT_PRICE = float(os.getenv("GPT4O_PROMPT_PRICE"))
        COMPLETION_PRICE = float(os.getenv("GPT4O_COMPLETION_PRICE"))
    elif OPENAI_MODEL == "gpt-4.1-mini":
        PROMPT_PRICE = float(os.getenv("GPT41MINI_PROMPT_PRICE"))
        COMPLETION_PRICE = float(os.getenv("GPT41MINI_COMPLETION_PRICE"))
    # Add other OpenAI models here as needed
elif LLM_PROVIDER == "anthropic":
    PROMPT_PRICE = float(os.getenv("ANTHROPIC_PROMPT_PRICE"))
    COMPLETION_PRICE = float(os.getenv("ANTHROPIC_COMPLETION_PRICE"))
    # Add specific Anthropic model pricing if needed

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up model response logger using the setup_logger function
MODEL_RESPONSE_LOGGER = setup_logger("model_responses", "logs/model_responses.log")

# List of required fields to check for null values
required_fields = [
    "vendor_name",
    "document_number",
    "issue_date",
    "due_date",
    "paid_date",
    "service_from",
    "service_to",
    "currency",
    "net_amount",
    "vat_amount",
    "gross_amount",
    "document_type",
]


def get_vision_model():
    """
    Initialize and return the appropriate vision-capable LLM based on configuration
    """
    if LLM_PROVIDER == "openai":
        # Initialize OpenAI vision model
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            max_tokens=4096,
            temperature=0.1,  # Lower temperature for more factual responses
            verbose=True,  # Enable verbose output for debugging
        )
    elif LLM_PROVIDER == "anthropic":
        # Import Anthropic integration if needed
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.1,
            max_tokens=4096,
            verbose=True,  # Enable verbose output for debugging
            return_response_metadata=True,  # For token usage tracking
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def call_llm_with_invoice(file_path: str, tenant: str) -> Dict[str, Any]:
    """
    Process an invoice file (PDF or image) and get the LLM response

    Args:
        file_path: Path to the invoice file
        tenant: Name of the tenant/company that's using this service

    Returns:
        Dictionary containing LLM response and metadata
    """

    # Get the file extension
    file_extension = file_path.split(".")[-1].lower()

    # Initialize vision model
    vision_model = get_vision_model()

    # Insert variables into the system prompt
    try:
        system_prompt = SYSTEM_PROMPT.format(tenant=tenant)
    except Exception as e:
        # Fallback to unformatted prompt if formatting fails
        system_prompt = SYSTEM_PROMPT

    if file_extension == "pdf":
        # Handle PDF by converting to images
        images = pdf_to_images(file_path)

        # For now, just process the first page
        # TODO: Implement logic for combining results from all pages
        image_data = images[0]

        # Convert bytes to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    else:
        # Handle image formats directly
        image_base64 = encode_image(file_path)

    # Create messages for the LLM
    prompt_messages = []

    if LLM_PROVIDER == "openai":
        # Add OpenAI-specific messages
        prompt_messages.append(SystemMessage(content=system_prompt))
        prompt_messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": HUMAN_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            )
        )
    elif LLM_PROVIDER == "anthropic":
        # Add Anthropic-specific messages
        prompt_messages.append(SystemMessage(content=system_prompt))
        prompt_messages.append(
            HumanMessage(
                content=f"{HUMAN_PROMPT}\n\n<image data:image/jpeg;base64,{image_base64}>"
            )
        )

    # Process with the vision model
    response = vision_model.invoke(prompt_messages)

    # Log the model response with raw data
    log_model_response(
        response,
        MODEL_RESPONSE_LOGGER,
        LLM_PROVIDER,
        OPENAI_MODEL if LLM_PROVIDER == "openai" else ANTHROPIC_MODEL,
        metadata={
            "file": os.path.basename(file_path),
            "file_type": file_extension,
            "tenant": tenant,
        },
    )

    # Return the response and file extension
    return {
        "response": response,
        "file_extension": file_extension,
    }


def parse_llm_response(
    llm_result: Dict[str, Any], file_path: str, tenant: str, processing_time: int
) -> Dict[str, Any]:
    """
    Parse the LLM response and format the final result

    Args:
        llm_result: Dictionary containing LLM response
        file_path: Path to the original invoice file
        tenant: Name of the tenant/company
        processing_time: Processing time in milliseconds

    Returns:
        Dictionary containing structured invoice data
    """
    response = llm_result["response"]

    # Extract token usage information
    token_info = extract_token_usage(response)
    prompt_tokens = token_info["prompt_tokens"]
    completion_tokens = token_info["completion_tokens"]
    total_tokens = token_info["total_tokens"]

    try:
        # Different models might format their responses differently
        response_text = response.content

        # Handle cases where model wraps JSON in code blocks
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        else:
            json_text = response_text.strip()

        # Parse the JSON
        result = json.loads(json_text)

        # Validate required fields and count completed (non-null) fields
        completed_fields_count = 0

        for field in required_fields:
            if field not in result:
                result[field] = None
            else:
                # Count non-null fields
                if result[field] is not None:
                    completed_fields_count += 1

        # Add completed fields count to the result
        result["completed_fields"] = completed_fields_count

        # Add metadata
        result["processed_at"] = datetime.now().isoformat()
        result["filename"] = os.path.basename(file_path)
        result["tenant"] = tenant

        # Add performance and token usage metrics
        result["processing_time"] = processing_time
        result["prompt_tokens"] = prompt_tokens
        result["completion_tokens"] = completion_tokens
        result["total_tokens"] = total_tokens

        # Calculate cost (if possible)
        total_cost = calculate_cost(
            prompt_tokens, completion_tokens, PROMPT_PRICE, COMPLETION_PRICE
        )
        result["total_cost"] = total_cost

        return result

    except json.JSONDecodeError as e:
        # If we can't parse the JSON, return an error
        print(f"JSON decode error: {str(e)}")
        return {
            "error": f"Failed to parse JSON from model response: {str(e)}",
            "raw_response": response.content,
            "processed_at": datetime.now().isoformat(),
            "filename": os.path.basename(file_path),
            "tenant": tenant,
            "processing_time": processing_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": calculate_cost(
                prompt_tokens, completion_tokens, PROMPT_PRICE, COMPLETION_PRICE
            ),
            "completed_fields": 0,  # No fields completed in case of error
        }


def check_duplicate_invoice(file_path: str) -> Dict[str, Any]:
    """
    Check if the given invoice is a duplicate or related to existing invoices

    Args:
        file_path: Path to the invoice file

    Returns:
        Dictionary with duplicate detection results
    """
    try:
        # Get the duplicate detector instance
        duplicate_detector = get_duplicate_detector()

        # Check for duplicates
        duplicate_check = duplicate_detector.check_for_duplicates(file_path)

        return duplicate_check
    except Exception as e:
        # Log the error but continue processing
        logging.error(f"Error checking for duplicate invoice: {str(e)}")
        return {
            "is_duplicate": False,
            "is_related": False,
            "similar_documents": [],
            "highest_score": 0,
            "error": str(e),
        }


def process_invoice(
    file_path: str,
    tenant: str = "Crispa Technologies ApS",
    check_duplicates: bool = True,
    auto_reject_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Process an invoice file (PDF or image) and extract information

    Args:
        file_path: Path to the invoice file
        tenant: Name of the tenant/company that's using this service,
                used to identify which entity should not be considered the vendor
        check_duplicates: Whether to check for duplicate invoices before processing
        auto_reject_duplicates: Whether to automatically reject exact duplicates

    Returns:
        Dictionary containing extracted invoice data or error information
    """
    # First, check for duplicates if enabled
    duplicate_info = None
    if check_duplicates:
        duplicate_info = check_duplicate_invoice(file_path)

        # If it's a duplicate and auto-reject is enabled, return without processing
        if duplicate_info.get("is_duplicate", False) and auto_reject_duplicates:
            return {
                "error": "Duplicate invoice detected",
                "processed_at": datetime.now().isoformat(),
                "filename": os.path.basename(file_path),
                "tenant": tenant,
                "processing_time": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "completed_fields": 0,
                "is_duplicate": True,
                "duplicate_info": duplicate_info,
            }

    try:
        # Start timing the processing
        start_time = time.time()

        # Step 1: Call LLM with invoice
        llm_result = call_llm_with_invoice(file_path, tenant)

        # Calculate processing time
        processing_time = int(
            (time.time() - start_time) * 1000
        )  # Convert to milliseconds

        # Step 2: Parse the LLM response
        result = parse_llm_response(llm_result, file_path, tenant, processing_time)

        # Add duplicate detection information if available
        if duplicate_info:
            result["duplicate_check"] = duplicate_info
            result["is_duplicate"] = duplicate_info.get("is_duplicate", False)
            result["is_related"] = duplicate_info.get("is_related", False)

        # Add document to the index after successful processing
        if check_duplicates and not duplicate_info.get("is_duplicate", False):
            try:
                # Generate a document ID from the file path and timestamp
                document_id = f"{os.path.basename(file_path)}_{int(time.time())}"

                # Add document to the index
                duplicate_detector = get_duplicate_detector()
                duplicate_detector.add_document(
                    file_path=file_path,
                    document_id=document_id,
                    metadata={"tenant": tenant},
                    extracted_data=result.get("extracted_data"),
                )
            except Exception as e:
                # Log the error but continue
                logging.error(f"Error adding document to duplicate index: {str(e)}")
                result["duplicate_index_error"] = str(e)

        return result

    except Exception as e:
        # Handle any other exceptions
        print(f"Error processing invoice: {str(e)}")
        return {
            "error": f"Error processing invoice: {str(e)}",
            "processed_at": datetime.now().isoformat(),
            "filename": os.path.basename(file_path),
            "tenant": tenant,
            "processing_time": -1,  # Indicate error with -1
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": None,  # Set to None for consistency
            "completed_fields": 0,  # No fields completed in case of error
        }
