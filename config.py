"""
Configuration management for the Data Analyst Agent.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()  

def get_google_api_key() -> str:
    """Get Google API key from environment."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return api_key

def get_google_model() -> str:
    """Get Google model name."""
    return os.getenv('GOOGLE_MODEL', 'gemini-2.5-flash')

def get_max_file_size() -> int:
    """Get maximum file size in bytes."""
    return int(os.getenv('MAX_FILE_SIZE_MB', '50')) * 1024 * 1024

def get_response_timeout() -> int:
    """Get response timeout in seconds."""
    return int(os.getenv('RESPONSE_TIMEOUT_SECONDS', '180'))

def get_debug_mode() -> bool:
    """Get debug mode setting."""
    return os.getenv('DEBUG', 'false').lower() == 'true'

def use_batch_processing() -> bool:
    """Whether to use batch processing for multiple questions."""
    return os.getenv('BATCH_PROCESSING', 'true').lower() == 'true'

def validate_config():
    """Validate configuration settings."""
    try:
        get_google_api_key()
        return True
    except ValueError as e:
        print(f"Configuration error: {e}")
        return False 