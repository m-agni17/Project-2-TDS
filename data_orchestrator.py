"""
Main orchestration module for the Data Analyst Agent.
Coordinates question parsing, data scraping, analysis, and visualization.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import json
import traceback
from groq import Groq

# Import our modules
from question_parser import process_question_file, create_groq_client
from data_scraper import scrape_data_from_urls
from data_analyzer import analyze_scraped_content

class DataAnalysisError(Exception):
    """Custom exception for data analysis errors."""
    pass

def process_data_analysis_request(question_content: str, 
                                additional_files: Dict[str, Any] = None) -> List[Any]:
    """
    Main function to process a complete data analysis request.
    
    Args:
        question_content: Content from questions.txt file
        additional_files: Dictionary of additional files {filename: content/data}
        
    Returns:
        List of answers in the requested format
    """
    try:
        # Step 1: Parse questions using LLM
        parsed_questions = process_question_file(question_content)
        
        # Step 2: Scrape data from identified URLs
        scraped_data = scrape_data_from_urls(parsed_questions["urls"])
        
        # Step 3: Incorporate additional files into the analysis
        combined_data = incorporate_additional_files(scraped_data, additional_files)
        
        # Step 4: Validate we have usable content
        if not validate_scraped_content(combined_data):
            raise DataAnalysisError("No usable content found from provided sources")
        
        # Step 5: Analyze all content with LLM
        client = create_groq_client()
        answers = analyze_scraped_content(
            combined_data,
            parsed_questions["questions"], 
            client
        )
        
        # Step 6: Validate response format
        if not validate_response_format(answers, len(parsed_questions["questions"])):
            print("Warning: Response format validation failed")
            # Still return the answers, but log the issue
        
        return answers
        
    except Exception as e:
        # Log error details for debugging
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"Analysis failed: {error_details}")
        
        # Return error response in expected format
        if "array" in question_content.lower():
            return [f"Error: {str(e)}"]
        else:
            return {"error": str(e)}

def incorporate_additional_files(scraped_data: Dict[str, Any],
                               additional_files: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Incorporate additional uploaded files into the scraped data for analysis.
    
    Args:
        scraped_data: Dictionary of scraped content from URLs
        additional_files: Additional files provided in the request
        
    Returns:
        Combined data dictionary
    """
    combined_data = scraped_data.copy()
    
    if additional_files:
        for filename, file_content in additional_files.items():
            try:
                if filename.lower().endswith('.csv'):
                    # Convert CSV content to text for LLM analysis
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                    
                    combined_data[f"uploaded_file_{filename}"] = {
                        'title': f"Uploaded CSV: {filename}",
                        'text_content': file_content[:5000],  # First 5000 chars
                        'file_type': 'csv',
                        'tables': [],  # Could parse CSV to DataFrame here
                        'links': [],
                        'images': [],
                        'lists': [],
                        'metadata': {'filename': filename}
                    }
                
                elif filename.lower().endswith(('.txt', '.md')):
                    # Handle text files
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                    
                    combined_data[f"uploaded_file_{filename}"] = {
                        'title': f"Uploaded Text: {filename}",
                        'text_content': file_content,
                        'file_type': 'text',
                        'tables': [],
                        'links': [],
                        'images': [],
                        'lists': [],
                        'metadata': {'filename': filename}
                    }
                
                else:
                    # Handle other file types as binary data info
                    combined_data[f"uploaded_file_{filename}"] = {
                        'title': f"Uploaded File: {filename}",
                        'text_content': f"Binary file: {filename}, Size: {len(file_content)} bytes",
                        'file_type': 'binary',
                        'tables': [],
                        'links': [],
                        'images': [],
                        'lists': [],
                        'metadata': {'filename': filename, 'size': len(file_content)}
                    }
                    
            except Exception as e:
                print(f"Failed to process file {filename}: {e}")
                continue
    
    return combined_data

def format_response_for_output(answers: List[Any], 
                             output_format: str) -> Union[List[Any], Dict[str, Any]]:
    """
    Format the final response according to the expected output format.
    
    Args:
        answers: List of answers
        output_format: Expected format ("array" or "object")
        
    Returns:
        Formatted response
    """
    if output_format == "array":
        return answers
    elif output_format == "object":
        # Convert to object format
        result = {}
        # This would need to be customized based on specific question patterns
        # For now, return as numbered questions
        for i, answer in enumerate(answers):
            result[f"question_{i+1}"] = answer
        return result
    else:
        return answers  # Default to array

def validate_scraped_content(scraped_data: Dict[str, Any]) -> bool:
    """
    Validate that we have usable content from scraping.
    
    Args:
        scraped_data: Dictionary of scraped content
        
    Returns:
        True if we have usable content, False otherwise
    """
    if not scraped_data:
        return False
    
    # Check if any URL returned usable content
    for url, content in scraped_data.items():
        if isinstance(content, dict):
            if content.get('text_content') or content.get('tables') or content.get('lists'):
                return True
        elif content is not None:
            return True
    
    return False

def validate_response_format(response: List[Any], expected_count: int = None) -> bool:
    """
    Validate that the response meets the expected format requirements.
    
    Args:
        response: Response to validate
        expected_count: Expected number of answers (optional)
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, list):
        return False
    
    if len(response) == 0:
        return False
    
    # Check if we have the expected number of answers
    if expected_count is not None and len(response) != expected_count:
        print(f"Response length mismatch: expected {expected_count}, got {len(response)}")
        return False
    
    # Check for visualization data URI if present
    if len(response) > 3:
        last_item = response[-1]
        if isinstance(last_item, str) and last_item.startswith("data:image/"):
            # Check size constraint (100KB = ~133,333 base64 chars)
            if len(last_item) > 140_000:  # Give some buffer
                print(f"Visualization data URI too large: {len(last_item)} characters")
                return False
    
    return True 