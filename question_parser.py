"""
Question parsing functionality using Google Gemini to extract questions and data sources.
Simplified to use Gemini directly for better performance.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import json
from google import genai
from google.genai import types
from config import get_google_api_key, get_google_model

def create_gemini_client():
    """Create and return a Gemini client instance."""
    return genai.Client(api_key=get_google_api_key())

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex patterns."""
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates

def process_question_file(question_content: str) -> Dict[str, Any]:
    """
    Process question file content and extract structured information.
    
    Args:
        question_content: Raw content from questions.txt file
        
    Returns:
        Dictionary containing parsed questions, URLs, and metadata
    """
    client = create_gemini_client()
    
    try:
        # Try Gemini parsing first
        parsed_questions = parse_questions_with_gemini(question_content, client)
        
        # Validate the parsed result
        if validate_parsed_questions(parsed_questions):
            return parsed_questions
        else:
            print("Gemini parsing validation failed, using fallback")
            return fallback_question_parsing(question_content)
    
    except Exception as e:
        print(f"Gemini parsing failed: {e}")
        return fallback_question_parsing(question_content)

def parse_questions_with_gemini(text: str, client) -> Dict[str, Any]:
    """
    Use Gemini to parse questions and extract structured information.
    
    Args:
        text: Raw question text
        client: Gemini client instance
    
    Returns:
        Dictionary containing parsed questions, URLs, and expected format
    """
    system_prompt = """
You are a question parser for a data analysis system. Your job is to:
1. Extract all questions from the input text
2. Identify any URLs mentioned for data sources
3. Determine the expected output format
4. Classify the type of analysis needed
5. If the question asks for a visualization, determine the type of visualization needed
6. If the question asks us to do a thing then it is a question like "Draw a scatterplot of Rank and Peak along with a dotted red regression line through it."


Return your response as a JSON object with the following structure:
{
    "questions": ["question1", "question2", ...],
    "urls": ["url1", "url2", ...],
    "output_format": "array" or "object",
    "analysis_types": ["scraping", "statistical", "visualization", "numerical"],
    "visualization_requirements": {
        "needed": true/false,
        "type": "scatterplot/histogram/etc",
        "encoding": "base64",
        "format": "png/webp/etc"
    }
}

Parse this text:

""" + text

    try:
        
        response = client.models.generate_content(
            model=get_google_model(),
            contents=system_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking
            )
        )
        
        response_text = response.text
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx + 1]
            parsed_response = json.loads(json_str)
        else:
            raise json.JSONDecodeError("No JSON found in response", response_text, 0)
        
        # Add direct URL extraction as backup
        direct_urls = extract_urls_from_text(text)
        if direct_urls:
            if "urls" not in parsed_response:
                parsed_response["urls"] = []
            parsed_response["urls"].extend(direct_urls)
            parsed_response["urls"] = list(set(parsed_response["urls"]))  # Remove duplicates
            
        return parsed_response
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Gemini parsing failed: {e}")
        # Fallback to basic parsing if Gemini fails
        return fallback_question_parsing(text)

def fallback_question_parsing(text: str) -> Dict[str, Any]:
    """
    Fallback question parsing when Gemini fails.
    Uses simple heuristics to extract questions.
    """
    lines = text.strip().split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        # Look for numbered questions or lines ending with '?'
        if (re.match(r'^\d+\.', line) or 
            line.endswith('?') or 
            'what' in line.lower() or 
            'how' in line.lower() or 
            'which' in line.lower() or
            'where' in line.lower() or
            'when' in line.lower() or
            'who' in line.lower() or
            'why' in line.lower() or
            'analyze' in line.lower() or
            'find' in line.lower() or
            'calculate' in line.lower() or
            'count' in line.lower()):
            questions.append(line)
    
    urls = extract_urls_from_text(text)
    
    # Determine if visualization is needed
    needs_viz = any(keyword in text.lower() for keyword in 
                   ['plot', 'chart', 'graph', 'scatterplot', 'scatter plot', 
                    'visualization', 'visualize', 'draw', 'histogram', 
                    'bar chart', 'line chart', 'base64', 'base-64'])
    
    # Determine visualization type
    viz_type = "unknown"
    if "scatterplot" in text.lower() or "scatter plot" in text.lower():
        viz_type = "scatterplot"
    elif "histogram" in text.lower():
        viz_type = "histogram"
    elif "bar chart" in text.lower():
        viz_type = "bar"
    elif "line chart" in text.lower():
        viz_type = "line"
    
    return {
        "questions": questions,
        "urls": urls,
        "output_format": "array" if "JSON array" in text else "object",
        "analysis_types": ["scraping", "analysis", "numerical"],
        "visualization_requirements": {
            "needed": needs_viz,
            "type": viz_type,
            "encoding": "base64",
            "format": "png"
        }
    }

def validate_parsed_questions(parsed_data: Dict[str, Any]) -> bool:
    """
    Validate that parsed question data is complete and valid.
    
    Args:
        parsed_data: Dictionary containing parsed question information
        
    Returns:
        True if data is valid, False otherwise
    """
    required_keys = ["questions", "urls", "output_format"]
    
    return (
        isinstance(parsed_data, dict) and
        all(key in parsed_data for key in required_keys) and
        isinstance(parsed_data["questions"], list) and
        isinstance(parsed_data["urls"], list) and
        len(parsed_data["questions"]) > 0
    )

# Keep old function name for backward compatibility
def create_groq_client():
    """Create Gemini client (backward compatibility)."""
    return create_gemini_client() 