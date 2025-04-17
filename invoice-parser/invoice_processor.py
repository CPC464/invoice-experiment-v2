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

# Import prompts
from prompts import prompt_a, prompt_b


# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Set prompts - Define these globally as they were before
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

# Set up model response logger
MODEL_RESPONSE_LOGGER = logging.getLogger("model_responses")
MODEL_RESPONSE_LOGGER.setLevel(logging.INFO)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create a file handler for model responses
model_responses_handler = logging.FileHandler("logs/model_responses.log")
model_responses_handler.setLevel(logging.INFO)


# Create a formatter for pretty-printed JSON
class PrettyJSONFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, indent=2, sort_keys=True)
        return super().format(record)


model_responses_handler.setFormatter(PrettyJSONFormatter())
MODEL_RESPONSE_LOGGER.addHandler(model_responses_handler)


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


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pdf_to_images(pdf_path: str) -> List[bytes]:
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of image byte data for each page
    """
    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    images = []

    # Iterate through pages
    for page_num in range(pdf_document.page_count):
        # Get page and render at higher resolution
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        img_bytes = pix.pil_tobytes(format="JPEG")
        images.append(img_bytes)

    return images


def calculate_cost(prompt_tokens, completion_tokens):
    """
    Calculate the cost in cents based on token usage and provider

    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion

    Returns:
        Cost in cents or None if pricing info not available
    """
    # Check if pricing info is available
    if PROMPT_PRICE is None or COMPLETION_PRICE is None:
        return None

    # Calculate costs based on established pricing
    prompt_cost = prompt_tokens * (PROMPT_PRICE / 1_000_000)
    completion_cost = completion_tokens * (COMPLETION_PRICE / 1_000_000)

    # Return total cost in cents
    return prompt_cost + completion_cost


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


def extract_token_usage(response) -> Dict[str, int]:
    """
    Extract token usage information from the LLM response

    Args:
        response: The LLM response object

    Returns:
        Dictionary with token counts
    """
    # Initialize token counts
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    token_usage = None

    # Extract token usage from the response
    if hasattr(response, "usage") and response.usage:
        token_usage = response.usage
    elif hasattr(response, "response_metadata") and response.response_metadata:
        if "usage" in response.response_metadata:
            token_usage = response.response_metadata["usage"]
        elif "token_usage" in response.response_metadata:
            token_usage = response.response_metadata["token_usage"]

    # Extract token counts from token_usage
    if token_usage:
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        # Calculate total if not provided
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
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

        # List of required fields to check for null values
        required_fields = [
            "vendor_name",
            "due_date",
            "paid_date",
            "service_from",
            "service_to",
            "currency",
            "net_amount",
            "vat_amount",
            "gross_amount",
        ]

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
        total_cost = calculate_cost(prompt_tokens, completion_tokens)
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
            "total_cost": calculate_cost(prompt_tokens, completion_tokens),
            "completed_fields": 0,  # No fields completed in case of error
        }


def process_invoice(
    file_path: str, tenant: str = "Crispa Technologies ApS"
) -> Dict[str, Any]:
    """
    Process an invoice file (PDF or image) and extract information

    Args:
        file_path: Path to the invoice file
        tenant: Name of the tenant/company that's using this service,
                used to identify which entity should not be considered the vendor

    Returns:
        Dictionary containing extracted invoice data or error information
    """
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


def generate_thumbnail(
    file_path: str, thumbnail_path: str, max_size: tuple = (1200, 1600)
) -> str:
    """
    Generate a thumbnail for an invoice file (PDF or image)

    Args:
        file_path: Path to the invoice file
        thumbnail_path: Path where to save the thumbnail
        max_size: Maximum dimensions for the thumbnail (width, height)

    Returns:
        Path to the generated thumbnail
    """
    try:
        # Get the file extension
        file_extension = file_path.split(".")[-1].lower()

        if file_extension == "pdf":
            # Handle PDF by converting first page to image
            pdf_document = fitz.open(file_path)

            # Get first page
            if pdf_document.page_count > 0:
                page = pdf_document.load_page(0)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        2.0, 2.0
                    )  # Higher resolution for better readability
                )
                img_data = pix.pil_tobytes(format="JPEG")

                # Open as PIL Image
                img = Image.open(io.BytesIO(img_data))
            else:
                # Empty PDF, create blank image
                img = Image.new("RGB", (100, 100), color="white")

        else:
            # Handle image formats directly
            img = Image.open(file_path)

        # Resize while maintaining aspect ratio
        img.thumbnail(max_size)

        # Save thumbnail with higher quality
        img.save(thumbnail_path, "JPEG", quality=95)

        return thumbnail_path

    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
        # Create a blank thumbnail in case of error
        try:
            blank = Image.new("RGB", (100, 100), color="#eeeeee")
            blank.save(thumbnail_path, "JPEG")
            return thumbnail_path
        except:
            return ""


def log_model_response(response, metadata=None):
    """
    Log a model response with timestamp and metadata

    Args:
        response: The model response content
        metadata: Optional dictionary with additional information
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "provider": LLM_PROVIDER,
        "model": OPENAI_MODEL if LLM_PROVIDER == "openai" else ANTHROPIC_MODEL,
    }

    # Add metadata if provided
    if metadata:
        log_entry.update(metadata)

    # Add response content
    if hasattr(response, "content"):
        log_entry["response"] = response.content
    else:
        log_entry["response"] = str(response)

    # Add raw usage data if available - directly from the response without processing
    if hasattr(response, "usage") and response.usage:
        log_entry["token_usage"] = response.usage
    elif hasattr(response, "response_metadata") and response.response_metadata:
        # For LangChain
        if "usage" in response.response_metadata:
            log_entry["token_usage"] = response.response_metadata["usage"]
        elif "token_usage" in response.response_metadata:
            log_entry["token_usage"] = response.response_metadata["token_usage"]

    # Log the pretty-printed entry
    MODEL_RESPONSE_LOGGER.info(log_entry)
