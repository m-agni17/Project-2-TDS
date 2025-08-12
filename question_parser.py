import re
import logging
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse
import asyncio

logger = logging.getLogger(__name__)

class QuestionParser:
    """
    Parses questions.txt files to extract questions, URLs, and output format requirements.
    Uses LLM-based parsing for better accuracy and understanding of natural language.
    """
    
    def __init__(self):
        # Only keep URL pattern for basic URL extraction as fallback
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    async def parse_questions(self, content: str) -> Tuple[List[str], List[str], str]:
        """
        Parse questions.txt content and extract questions, URLs, and output format using LLM.
        
        Args:
            content: Raw content of questions.txt file
            
        Returns:
            Tuple of (questions_list, urls_list, output_format)
        """
        try:
            logger.info("Parsing questions.txt content using LLM")
            
            # Clean the content
            content = content.strip()
            
            # Extract everything using LLM for better accuracy
            questions = await self._extract_questions_with_llm(content)
            urls = await self._extract_urls_with_llm(content)
            output_format = await self._detect_output_format_with_llm(content)
            
            # If LLM fails for questions, log error but continue with other data
            if not questions:
                logger.warning("LLM question extraction returned no results")
            
            logger.info(f"Extracted {len(questions)} questions, {len(urls)} URLs, format: {output_format}")
            
            return questions, urls, output_format
            
        except Exception as e:
            logger.error(f"Error parsing questions: {str(e)}")
            return [], [], "json_array"  # Default fallback

    async def _extract_questions_with_llm(self, content: str) -> List[str]:
        """Extract questions using LLM for better accuracy and natural language understanding."""
        try:
            # Import here to avoid circular imports
            from llm_client import LLMClient
            
            llm_client = LLMClient()
            
            prompt = f"""You are an expert text parser specialized in extracting analysis questions from text documents.

TEXT TO PARSE:
{content}

TASK:
Extract all individual questions, analysis requests, or tasks that require data processing or analysis. Return ONLY the questions/tasks, one per line.

RULES:
1. Extract numbered questions (1., 2., 3., etc.) and remove the numbering
2. Extract lettered questions (a., b., c., etc.) and remove the lettering  
3. Extract questions that start with question words (How, What, Which, etc.)
4. Extract requests for analysis, plots, calculations, or visualizations
5. Extract imperative statements that request actions (Calculate..., Find..., Determine...)
6. Remove all numbering/lettering prefixes from your output
7. Ignore URLs, format instructions, and metadata
8. Keep each question exactly as written (don't paraphrase or summarize)
9. Each question should be on its own line
10. Don't include explanatory text, just the questions

EXAMPLES OF WHAT TO EXTRACT:
- "How many movies earned over $2 billion worldwide?"
- "What is the correlation between budget and revenue?"
- "Calculate the average rating for each genre"
- "Plot a scatterplot showing the relationship between variables"
- "Find the earliest and latest release years"
- "Which director has the highest average box office?"

EXAMPLES OF WHAT TO IGNORE:
- URLs (http://...)
- Format instructions ("Answer in JSON format")
- Data source references ("Using the dataset from...")
- General instructions ("Scrape the data")

Extract the questions now:"""
            
            try:
                response = await llm_client._query_llm(prompt)
                
                # Parse the response more intelligently
                questions = []
                lines = response.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip common LLM response artifacts
                    skip_patterns = [
                        'extract the questions',
                        'here are the',
                        'the questions are',
                        'analysis requests',
                        'questions:',
                        'output:',
                        'answer:',
                        'format:',
                        'instructions:',
                        'examples:',
                        'based on the text',
                        'from the given text'
                    ]
                    
                    if any(pattern in line.lower() for pattern in skip_patterns):
                        continue
                    
                    # Remove any remaining numbering/lettering artifacts
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove "1. " or "1) "
                    line = re.sub(r'^[a-zA-Z][\.\)]\s*', '', line)  # Remove "a. " or "a) "
                    line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
                    line = re.sub(r'^\s*[-]\s*', '', line)  # Remove dashes
                    
                    # Final cleanup
                    line = line.strip()
                    
                    # Basic validation - should be a reasonable length and look like a question/task
                    if len(line) > 5 and self._is_valid_question_or_task(line):
                        questions.append(line)
                
                if questions:
                    logger.info(f"LLM successfully extracted {len(questions)} questions")
                    return questions
                else:
                    logger.warning("LLM returned response but no valid questions found")
                    return []
                
            except Exception as e:
                logger.error(f"LLM question extraction failed: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in LLM question extraction: {str(e)}")
            return []

    def _is_valid_question_or_task(self, text: str) -> bool:
        """Simple validation to check if text looks like a valid question or analysis task."""
        text_lower = text.lower()
        
        # Skip obvious non-questions
        if any(skip in text_lower for skip in [
            'http://', 'https://', 'www.', '.com', '.org',
            'json format', 'csv format', 'answer in',
            'return as', 'format:', 'using the dataset'
        ]):
            return False
        
        # Must have some substance
        if len(text.split()) < 3:
            return False
            
        # Look for question/task indicators
        indicators = [
            # Question words
            'how', 'what', 'where', 'when', 'why', 'which', 'who',
            # Question auxiliaries  
            'is', 'are', 'was', 'were', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'should',
            # Action words
            'calculate', 'find', 'determine', 'identify', 'analyze',
            'compare', 'count', 'plot', 'draw', 'show', 'display',
            'create', 'generate', 'list', 'extract', 'compute',
            # Analysis terms
            'correlation', 'regression', 'average', 'mean', 'median',
            'distribution', 'relationship', 'trend', 'pattern'
        ]
        
        # Check if it starts with an indicator or ends with ?
        first_word = text_lower.split()[0] if text_lower.split() else ''
        
        return (
            first_word in indicators or 
            text.endswith('?') or
            any(indicator in text_lower for indicator in indicators)
        )
    
    async def _extract_urls_with_llm(self, content: str) -> List[str]:
        """Extract URLs using LLM for better accuracy."""
        try:
            # Import here to avoid circular imports
            from llm_client import LLMClient
            
            llm_client = LLMClient()
            
            prompt = f"""Extract all URLs from the following text. Return ONLY the URLs, one per line.

TEXT:
{content}

INSTRUCTIONS:
- Find all HTTP and HTTPS URLs
- Return each URL on a separate line
- Do not include any explanations or additional text
- If no URLs found, return empty response

URLs:"""
            
            try:
                response = await llm_client._query_llm(prompt)
                
                urls = []
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if line.startswith(('http://', 'https://')):
                        # Validate URL format
                        try:
                            parsed = urlparse(line)
                            if parsed.scheme and parsed.netloc:
                                urls.append(line)
                        except:
                            continue
                
                logger.info(f"LLM extracted {len(urls)} URLs")
                return urls
                
            except Exception as e:
                logger.error(f"LLM URL extraction failed: {str(e)}")
                # Fallback to regex for URLs as they have a clear pattern
                return self._extract_urls_fallback(content)
                
        except Exception as e:
            logger.error(f"Error in LLM URL extraction: {str(e)}")
            return self._extract_urls_fallback(content)
    
    def _extract_urls_fallback(self, content: str) -> List[str]:
        """Fallback URL extraction using regex pattern."""
        try:
            urls = re.findall(self.url_pattern, content)
            
            # Clean and validate URLs
            cleaned_urls = []
            for url in urls:
                try:
                    parsed = urlparse(url)
                    if parsed.scheme and parsed.netloc:
                        cleaned_urls.append(url)
                except:
                    continue
            
            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(cleaned_urls))
            
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error in fallback URL extraction: {str(e)}")
            return []
    
    async def _detect_output_format_with_llm(self, content: str) -> str:
        """Detect output format using LLM."""
        try:
            # Import here to avoid circular imports
            from llm_client import LLMClient
            
            llm_client = LLMClient()
            
            prompt = f"""Analyze the following text and determine what output format is requested for the answers.

TEXT:
{content}

TASK: Determine the requested output format from these options:
- json_array (if text mentions "JSON array format", "array format", "list format", or similar)
- json_object (if text mentions "JSON object format", "object format", "dictionary format", or similar)

Look for phrases like:
- "answer in JSON array format"
- "return as JSON object"  
- "format the response as an array"
- "provide results in object format"

Return ONLY one of these two options: json_array OR json_object

If no specific format is mentioned or if unclear, default to: json_array

Format:"""
            
            try:
                response = await llm_client._query_llm(prompt)
                
                response = response.strip().lower()
                
                if 'json_object' in response or 'object' in response:
                    return 'json_object'
                else:
                    return 'json_array'  # Default
                
            except Exception as e:
                logger.error(f"LLM format detection failed: {str(e)}")
                return 'json_array'  # Default fallback
                
        except Exception as e:
            logger.error(f"Error in LLM format detection: {str(e)}")
            return 'json_array'  # Default fallback
    
    def extract_data_sources(self, content: str) -> Dict[str, Any]:
        """
        Extract data source information from questions.txt.
        
        Args:
            content: Raw content of questions.txt
            
        Returns:
            Dictionary with data source information
        """
        try:
            sources = {
                'urls': [],
                's3_paths': [],
                'database_connections': [],
                'file_references': []
            }
            
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Extract URLs
                if 'http://' in line or 'https://' in line:
                    urls = re.findall(self.url_pattern, line)
                    sources['urls'].extend(urls)
                
                # Extract S3 paths - improved parsing for SQL context
                if 's3://' in line:
                    # Handle S3 URLs that might be embedded in SQL queries
                    # Pattern: s3://bucket/path?query_params followed by optional SQL characters
                    s3_pattern = r"s3://[^'\s\)]+(?:\?[^'\s\)]+)?"
                    s3_matches = re.findall(s3_pattern, line)
                    
                    for match in s3_matches:
                        # Clean up the URL - remove any trailing SQL characters
                        clean_url = match.rstrip("');")
                        
                        # Parse S3 path and extract region if present in query params
                        s3_info = {'path': clean_url}
                        
                        # Check for s3_region in the URL query parameters
                        if '?s3_region=' in clean_url:
                            url_parts = clean_url.split('?')
                            base_url = url_parts[0]
                            query_params = url_parts[1] if len(url_parts) > 1 else ''
                            
                            # Extract region from query parameters
                            region_match = re.search(r's3_region=([^&]+)', query_params)
                            if region_match:
                                s3_info['region'] = region_match.group(1)
                                # Use the base URL without query parameters for DuckDB
                                s3_info['path'] = base_url
                        
                        # Also check for s3_region mentioned separately in the line
                        if 'region' not in s3_info:
                            region_match = re.search(r's3_region=([^&\s]+)', line)
                            if region_match:
                                s3_info['region'] = region_match.group(1)
                        
                        sources['s3_paths'].append(s3_info)
                
                # Extract database connection info (basic patterns)
                if any(keyword in line.lower() for keyword in ['postgres://', 'mysql://', 'sqlite://']):
                    sources['database_connections'].append({'connection_string': line})
            
            return sources
            
        except Exception as e:
            logger.error(f"Error extracting data sources: {str(e)}")
            return {'urls': [], 's3_paths': [], 'database_connections': [], 'file_references': []} 