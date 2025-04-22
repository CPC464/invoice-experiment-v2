import json
import logging
from datetime import datetime


class PrettyJSONFormatter(logging.Formatter):
    """Formatter for logging pretty-printed JSON"""

    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, indent=2, sort_keys=True)
        return super().format(record)


def log_model_response(
    response, model_response_logger, llm_provider, model_name, metadata=None
):
    """
    Log a model response with timestamp and metadata

    Args:
        response: The model response content
        model_response_logger: Logger instance to use
        llm_provider: Provider name (e.g., 'openai', 'anthropic')
        model_name: Name of the model being used
        metadata: Optional dictionary with additional information
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "provider": llm_provider,
        "model": model_name,
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
    model_response_logger.info(log_entry)


def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    Set up a logger with file handler and formatter

    Args:
        logger_name: Name of the logger
        log_file: Path to the log file
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Use PrettyJSONFormatter for the handler
    file_handler.setFormatter(PrettyJSONFormatter())

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger
