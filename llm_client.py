import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import json
import re
from groq import Groq
import asyncio
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


logger = logging.getLogger(__name__)

def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for async functions to retry on failure."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator

class LLMClient:
    """
    Handles LLM interactions using Groq for data analysis tasks.
    Follows the Single Responsibility Principle by focusing on LLM communication and prompt engineering.
    """
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable must be set")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        self.max_tokens = 6000
        
        # Data analysis prompt templates
        self.system_prompt = """You are a senior data analyst with extensive experience in statistical analysis, data visualization, and business intelligence. You excel at:

1. Analyzing complex datasets and extracting meaningful insights
2. Performing statistical calculations with high accuracy
3. Understanding data relationships and correlations
4. Providing precise, data-driven answers
5. Working with various data formats (CSV, JSON, Parquet, etc.)

When analyzing data:
- Always base your answers strictly on the provided dataset
- Show your reasoning and calculations where relevant
- Be precise with numerical results
- If you can't determine an answer from the data, clearly state this
- Format responses exactly as requested (JSON array, JSON object, etc.)
- For visualizations, provide clear descriptions of what should be plotted

You must be deterministic - given the same data and question, always provide the same answer."""
        
        # LLM I/O logging configuration
        self.llm_log_dir = os.getenv("LLM_LOG_DIR", os.path.join(os.getcwd(), "logs"))
        self.llm_input_log_path = os.path.join(self.llm_log_dir, "llm_input.txt")
        self.llm_response_log_path = os.path.join(self.llm_log_dir, "llm_response.txt")
        try:
            os.makedirs(self.llm_log_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create log directory '{self.llm_log_dir}': {e}")
    
    async def analyze_question(self, question: str, datasets: Dict[str, pd.DataFrame]) -> Any:
        """
        Analyze a question against the provided datasets.
        
        Args:
            question: The analysis question to answer
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            Analysis result (could be string, number, list, etc.)
        """
        try:
            logger.info(f"Analyzing question: {question[:100]}...")
            
            # Filter and prepare relevant data based on question
            relevant_data = await self._filter_relevant_data(question, datasets)
            
            # Prepare focused data context
            data_context = await self._prepare_focused_data_context(relevant_data)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(question, data_context)
            # Get response from LLM
            response = await self._query_llm(prompt)
            
            # Parse and return response
            return await self._parse_analysis_response(response, question)
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            return f"Error analyzing question: {str(e)}"
    
    async def _filter_relevant_data(self, question: str, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter datasets to include only relevant data based on the question."""
        try:
            filtered_datasets = {}
            
            for dataset_name, df in datasets.items():
                # Check if this is comprehensive web scraped data
                if 'content_type' in df.columns and 'scraped_' in dataset_name:
                    filtered_df = await self._filter_scraped_data(question, df)
                    if not filtered_df.empty:
                        filtered_datasets[dataset_name] = filtered_df
                else:
                    # For regular datasets, include as-is but potentially sample if too large
                    if len(df) > 1000:
                        # Sample the data but keep it representative
                        filtered_datasets[dataset_name] = df.sample(n=1000, random_state=42)
                    else:
                        filtered_datasets[dataset_name] = df
            
            return filtered_datasets
            
        except Exception as e:
            logger.error(f"Error filtering relevant data: {str(e)}")
            return datasets  # Return original if filtering fails
    
    async def _filter_scraped_data(self, question: str, df: pd.DataFrame) -> pd.DataFrame:
        """Filter scraped web data to include only relevant content."""
        try:
            question_lower = question.lower()
            relevant_rows = []
            
            # Always include table data as it's usually the most structured and relevant
            table_rows = df[df['content_type'] == 'table_row'].copy()
            if not table_rows.empty:
                relevant_rows.append(table_rows)
            
            # Include headings for context
            headings = df[df['content_type'] == 'heading'].copy()
            if not headings.empty:
                relevant_rows.append(headings)
            
            # Include relevant paragraphs based on keywords in question
            paragraphs = df[df['content_type'] == 'paragraph'].copy()
            if not paragraphs.empty:
                # Filter paragraphs that might be relevant to the question
                keywords = self._extract_keywords_from_question(question)
                if keywords:
                    relevant_paragraphs = paragraphs[
                        paragraphs['content'].str.contains('|'.join(keywords), case=False, na=False)
                    ]
                    if not relevant_paragraphs.empty:
                        relevant_rows.append(relevant_paragraphs)
                else:
                    # If no specific keywords, take first few paragraphs
                    relevant_rows.append(paragraphs.head(3))
            
            # Include relevant list items
            list_items = df[df['content_type'] == 'list_item'].copy()
            if not list_items.empty and len(list_items) < 100:  # Only if not too many
                relevant_rows.append(list_items)
            
            if relevant_rows:
                filtered_df = pd.concat(relevant_rows, ignore_index=True)
                logger.info(f"Filtered scraped data from {len(df)} to {len(filtered_df)} rows")
                return filtered_df
            else:
                return df.head(100)  # Fallback to first 100 rows
                
        except Exception as e:
            logger.error(f"Error filtering scraped data: {str(e)}")
            return df.head(100)  # Safe fallback
    
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract relevant keywords from the question for filtering."""
        # Common question words to ignore
        stop_words = {'how', 'what', 'where', 'when', 'why', 'which', 'who', 'is', 'are', 'was', 'were', 
                     'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'the', 'a', 'an',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                     'many', 'much', 'some', 'any', 'all', 'most', 'least', 'between', 'among'}
        
        # Extract meaningful words
        words = question.lower().split()
        keywords = [word.strip('.,!?:;()[]{}') for word in words 
                   if len(word) > 3 and word.lower() not in stop_words]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    async def _prepare_focused_data_context(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Prepare a focused data context with only essential information."""
        try:
            context_parts = []
            
            for dataset_name, df in datasets.items():
                context_parts.append(f"\n=== Dataset: {dataset_name} ===")
                
                # Check if this is comprehensive web scraped data
                if 'content_type' in df.columns and 'scraped_' in dataset_name:
                    # Show page title
                    if 'page_title' in df.columns:
                        page_title = df['page_title'].iloc[0] if not df.empty else 'Unknown'
                        context_parts.append(f"Source: {page_title}")
                    
                    # Show table data in a clean, structured format (VERY LIMITED)
                    table_rows = df[df['content_type'] == 'table_row']
                    if not table_rows.empty:
                        context_parts.append(f"\nData Table:")
                        
                        # Only show the first table and first 5 rows
                        first_table_data = table_rows[table_rows['table_index'] == table_rows['table_index'].iloc[0]]
                        for _, row in first_table_data.head(5).iterrows():
                            row_data = row.get('row_data', '')
                            if row_data and len(row_data) < 200:  # Limit row length
                                context_parts.append(f"  {row_data}")
                
                else:
                    # Regular dataset - show very limited sample
                    context_parts.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                    context_parts.append(f"Columns: {', '.join(df.columns.tolist()[:5])}")  # Only first 5 columns
                    
                    # Show only 3 sample rows
                    context_parts.append("Sample data:")
                    for i, (_, row) in enumerate(df.head(3).iterrows()):
                        row_str = " | ".join([f"{col}: {str(val)[:30]}" for col, val in list(row.items())[:5]])  # Limit columns and text length
                        context_parts.append(f"  Row {i+1}: {row_str}")
                
                context_parts.append("")  # Empty line separator
            
            result = "\n".join(context_parts)
            
            # Final safety check - if still too long, truncate
            if len(result) > 8000:  # Conservative limit
                result = result[:8000] + "...\n[Data truncated due to size limits]"
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing focused data context: {str(e)}")
            return f"Error preparing focused data context: {str(e)}"
    
    def _create_analysis_prompt(self, question: str, data_context: str) -> str:
        """Create a comprehensive prompt for data analysis."""
        prompt = f"""You are a senior data analyst. Based on the provided datasets, answer this question with precision and accuracy:

QUESTION: {question}

DATA CONTEXT:
{data_context}

INSTRUCTIONS:
1. Analyze the data carefully to answer the question based on the actual data provided
2. Return ONLY the direct answer - no explanations, reasoning, or extra text
3. For numerical questions: return just the number (e.g., "42" not "The answer is 42")
4. For text questions: return just the answer (e.g., "ProductX" not "The product is ProductX")
5. For correlation questions: return just the correlation value (e.g., "0.485")
6. For date questions: return in YYYY-MM-DD format when possible (e.g., "2024-01-15")
7. If you cannot determine the answer from the available data, return "Unable to determine from available data"
8. Be precise with numerical results - use appropriate decimal places
9. Return ONLY what is asked for - nothing more

ANALYSIS GUIDELINES:
- For counting questions: Count the actual items/records/entities in the data
- For "highest/maximum" questions: Find the actual maximum value or entity with highest value
- For "lowest/minimum" questions: Find the actual minimum value or entity with lowest value
- For average/mean questions: Calculate the arithmetic mean of the relevant values
- For median questions: Find the middle value when data is sorted
- For correlation questions: Calculate Pearson correlation coefficient between specified variables
- For sum/total questions: Add up all relevant values
- For "which" questions seeking entities: Return the actual name/identifier from the data
- For date-related questions: Use the actual dates from the data in appropriate format

Answer:"""

        return prompt
    
    async def _append_log(self, path: str, title: str, content: str) -> None:
        """Append a time-stamped entry to a log file asynchronously."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"\n----- {title} @ {timestamp} -----\n{content}\n"
        try:
            # Use thread to avoid blocking event loop
            await asyncio.to_thread(self._sync_append, path, entry)
        except Exception as e:
            logger.warning(f"Failed to write LLM log to {path}: {e}")
    
    def _sync_append(self, path: str, text: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text)
    
    @async_retry(max_attempts=3, delay=1.0)
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM with retry logic."""
        try:
            # Log the prompt
            await self._append_log(self.llm_input_log_path, "LLM INPUT", prompt)
            
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def sync_query():
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0.1  # Low temperature for consistency
                )
                return chat_completion.choices[0].message.content
            
            response = await loop.run_in_executor(None, sync_query)
            
            # Log the response
            await self._append_log(self.llm_response_log_path, "LLM RESPONSE", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            raise
    
    async def _parse_analysis_response(self, response: str, question: str) -> Any:
        """Parse the LLM response and return appropriate data type."""
        try:
            # Clean the response - remove extra whitespace and common prefixes
            response = response.strip()
            
            # Remove common response prefixes that the LLM might add
            prefixes_to_remove = [
                "Answer:", "The answer is", "Result:", "Based on the data,", 
                "According to the data,", "From the analysis,", "The result is",
                "Looking at the data,", "After analyzing,"
            ]
            
            for prefix in prefixes_to_remove:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].strip()
                    break
            
            # Remove trailing periods or other punctuation for clean answers
            response = response.rstrip('.,!?:;')
            
            # Try to determine if response should be parsed as JSON
            if (response.startswith('[') and response.endswith(']')) or \
               (response.startswith('{') and response.endswith('}')):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as string
                    return response
            
            # Handle numeric responses
            # Check for pure numbers (including decimals and negatives)
            if re.match(r'^-?\d+\.?\d*$', response):
                if '.' in response:
                    return float(response)
                else:
                    return int(response)
            
            # Handle correlation values (might have leading/trailing text)
            correlation_match = re.search(r'-?\d+\.\d+', response)
            if correlation_match and ('correlation' in question.lower() or 'corr' in question.lower()):
                return float(correlation_match.group())
            
            # Handle percentage values
            if response.endswith('%'):
                try:
                    return float(response[:-1])
                except ValueError:
                    pass
            
            # Handle currency values (remove $ and commas)
            if '$' in response:
                clean_value = response.replace('$', '').replace(',', '').strip()
                try:
                    if '.' in clean_value:
                        return float(clean_value)
                    else:
                        return int(clean_value)
                except ValueError:
                    pass
            
            # Return as string if no other type matches
            return response
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return response  # Return original response if parsing fails

    async def analyze_questions_batch(self, questions: List[str], datasets: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Analyze multiple questions in a single LLM call for better efficiency.
        
        Args:
            questions: List of analysis questions to answer
            datasets: Dictionary of dataset name to DataFrame
            
        Returns:
            List of analysis results corresponding to each question
        """
        try:
            logger.info(f"Analyzing {len(questions)} questions in batch mode")
            
            # Filter and prepare relevant data based on all questions
            relevant_data = await self._filter_relevant_data_for_batch(questions, datasets)
            
            # Prepare focused data context
            data_context = await self._prepare_focused_data_context(relevant_data)
            
            # Create batch analysis prompt
            prompt = self._create_batch_analysis_prompt(questions, data_context)
            
            # Get response from LLM
            response = await self._query_llm(prompt)
            
            # Parse batch response and return results
            return await self._parse_batch_analysis_response(response, questions)
            
        except Exception as e:
            logger.error(f"Error in batch question analysis: {str(e)}")
            # Fallback to individual processing if batch fails
            logger.info("Falling back to individual question processing")
            results = []
            for question in questions:
                try:
                    result = await self.analyze_question(question, datasets)
                    results.append(result)
                except Exception as individual_error:
                    logger.error(f"Error processing individual question '{question}': {str(individual_error)}")
                    results.append(f"Error processing question: {str(individual_error)}")
            return results

    async def _filter_relevant_data_for_batch(self, questions: List[str], datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Filter datasets to include relevant data based on all questions in the batch."""
        try:
            filtered_datasets = {}
            
            # Combine all questions to understand overall data requirements
            combined_questions = " ".join(questions)
            
            for dataset_name, df in datasets.items():
                # Check if this is comprehensive web scraped data
                if 'content_type' in df.columns and 'scraped_' in dataset_name:
                    filtered_df = await self._filter_scraped_data(combined_questions, df)
                    if not filtered_df.empty:
                        filtered_datasets[dataset_name] = filtered_df
                else:
                    # For regular datasets, include as-is but potentially sample if too large
                    # For batch processing, we can be more generous with data size since it's one call
                    if len(df) > 2000:  # Increased limit for batch processing
                        # Sample the data but keep it representative
                        filtered_datasets[dataset_name] = df.sample(n=2000, random_state=42)
                    else:
                        filtered_datasets[dataset_name] = df
            
            return filtered_datasets
            
        except Exception as e:
            logger.error(f"Error filtering relevant data for batch: {str(e)}")
            return datasets  # Return original if filtering fails

    def _create_batch_analysis_prompt(self, questions: List[str], data_context: str) -> str:
        """Create a comprehensive prompt for batch data analysis."""
        
        # Number the questions for clear mapping
        numbered_questions = []
        for i, question in enumerate(questions, 1):
            numbered_questions.append(f"{i}. {question}")
        
        questions_text = "\n".join(numbered_questions)
        
        prompt = f"""You are a senior data analyst. Based on the provided datasets, answer ALL questions with precision and accuracy.

QUESTIONS TO ANSWER:
{questions_text}

DATA CONTEXT:
{data_context}

INSTRUCTIONS:
1. Answer ALL questions in the exact order they are listed (1, 2, 3, etc.)
2. For each question, return ONLY the direct answer - no explanations, reasoning, or extra text
3. Format your response as a JSON array where each element corresponds to the answer for that question number
4. For numerical questions: return just the number (e.g., 42 not "The answer is 42")
5. For text questions: return just the answer (e.g., "ProductX" not "The product is ProductX")
6. For correlation questions: return just the correlation value as a number (e.g., 0.485)
7. For date questions: return in YYYY-MM-DD format when possible (e.g., "2024-01-15")
8. Analyze the data carefully and compute metrics precisely based on the actual data provided
9. If you cannot determine an answer from the available data, return "Unable to determine from available data"
10. Be precise with numerical results - use appropriate decimal places but avoid excessive precision
11. Return ONLY what is asked for each question - nothing more

ANALYSIS GUIDELINES:
- For counting questions: Count the actual items/records/entities in the data
- For "highest/maximum" questions: Find the actual maximum value or entity with highest value
- For "lowest/minimum" questions: Find the actual minimum value or entity with lowest value
- For average/mean questions: Calculate the arithmetic mean of the relevant values
- For median questions: Find the middle value when data is sorted
- For correlation questions: Calculate Pearson correlation coefficient between specified variables
- For sum/total questions: Add up all relevant values
- For "which" questions seeking entities: Return the actual name/identifier from the data
- For date-related questions: Use the actual dates from the data in appropriate format
- For percentage/ratio questions: Calculate based on the actual data proportions

CRITICAL: You MUST respond with ONLY a valid JSON array. No other text, no explanations, no markdown formatting.

RESPONSE FORMAT:
Return a valid JSON array with exactly {len(questions)} elements, where:
- Element 1 = answer to question 1
- Element 2 = answer to question 2
- And so on...

Example format: ["answer1", 42, "answer3", 0.485, "2024-01-15"]

JSON Response:"""
        
        return prompt

    async def _parse_batch_analysis_response(self, response: str, questions: List[str]) -> List[Any]:
        """Parse the batch LLM response and return a list of answers."""
        try:
            # Clean the response more thoroughly
            response = response.strip()
            
            # Remove markdown formatting
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Remove common response prefixes
            prefixes_to_remove = [
                "JSON Response:", "Response:", "Answer:", "Result:", "Based on the data:",
                "According to the data:", "From the analysis:", "The results are:",
                "Looking at the data:", "After analyzing:", "Here are the answers:",
                "JSON Array:", "Array:", "The answers are:", "Answers:"
            ]
            
            for prefix in prefixes_to_remove:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].strip()
                    break
            
            # Remove any trailing text after the JSON array
            # Look for the last ] and truncate there
            last_bracket = response.rfind(']')
            if last_bracket != -1:
                # Check if there's significant content after the last bracket
                after_bracket = response[last_bracket + 1:].strip()
                if after_bracket and not after_bracket.startswith(','):  # Allow for trailing commas
                    response = response[:last_bracket + 1]
            
            # Try to parse as JSON array
            try:
                parsed_response = json.loads(response)
                
                # Ensure it's a list
                if not isinstance(parsed_response, list):
                    logger.warning("LLM returned non-array response, converting to array")
                    parsed_response = [parsed_response]
                
                # Ensure we have the right number of answers
                if len(parsed_response) != len(questions):
                    logger.warning(f"Expected {len(questions)} answers but got {len(parsed_response)}")
                    
                    # Pad with error messages if too few
                    while len(parsed_response) < len(questions):
                        parsed_response.append("Unable to determine from available data")
                    
                    # Truncate if too many
                    parsed_response = parsed_response[:len(questions)]
                
                # Post-process each answer
                processed_answers = []
                for i, (answer, question) in enumerate(zip(parsed_response, questions)):
                    processed_answer = await self._parse_individual_answer(answer, question)
                    processed_answers.append(processed_answer)
                
                logger.info(f"Successfully parsed batch response with {len(processed_answers)} answers")
                return processed_answers
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse batch response as JSON: {str(e)}")
                logger.error(f"Raw response: {response[:500]}...")  # Log first 500 chars for debugging
                
                # Try to extract answers from text format as fallback
                return await self._parse_text_batch_response(response, questions)
                
        except Exception as e:
            logger.error(f"Error parsing batch analysis response: {str(e)}")
            
            # Return error messages for all questions
            return ["Unable to determine from available data" for _ in questions]

    async def _parse_individual_answer(self, answer: Any, question: str) -> Any:
        """Parse an individual answer from the batch response."""
        try:
            # If already a non-string type, return as-is
            if not isinstance(answer, str):
                return answer
            
            # Apply the same parsing logic as individual responses
            return await self._parse_analysis_response(str(answer), question)
            
        except Exception as e:
            logger.error(f"Error parsing individual answer: {str(e)}")
            return str(answer)  # Return as string if parsing fails

    async def _parse_text_batch_response(self, response: str, questions: List[str]) -> List[Any]:
        """Fallback parser for non-JSON batch responses."""
        try:
            logger.info("Attempting to parse batch response as structured text")
            
            answers = []
            lines = response.split('\n')
            
            # Try to find numbered answers
            for i in range(1, len(questions) + 1):
                answer_found = False
                
                # Look for patterns like "1. answer", "1) answer", "Answer 1: answer"
                patterns = [
                    rf"^{i}[\.\)]\s*(.+)$",
                    rf"^Answer\s+{i}:\s*(.+)$",
                    rf"^Question\s+{i}:\s*(.+)$",
                    rf"^{i}:\s*(.+)$"
                ]
                
                for line in lines:
                    line = line.strip()
                    for pattern in patterns:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            answer = match.group(1).strip()
                            processed_answer = await self._parse_individual_answer(answer, questions[i-1])
                            answers.append(processed_answer)
                            answer_found = True
                            break
                    if answer_found:
                        break
                
                if not answer_found:
                    answers.append("Unable to determine from available data")
            
            logger.info(f"Parsed {len(answers)} answers from text format")
            return answers
            
        except Exception as e:
            logger.error(f"Error parsing text batch response: {str(e)}")
            return ["Error processing question" for _ in questions] 