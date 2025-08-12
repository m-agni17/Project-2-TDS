import os
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging

from data_processor import DataProcessor
from llm_client import LLMClient
from web_scraper import WebScraper
from question_parser import QuestionParser
from visualization_generator import VisualizationGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API",
    description="AI-powered data analysis API that processes arbitrary datasets and answers analysis questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
data_processor = DataProcessor()
# Lazily initialize LLM client to avoid import-time env var issues
llm_client: Optional[LLMClient] = None
web_scraper = WebScraper()
question_parser = QuestionParser()
viz_generator = VisualizationGenerator()


def get_llm_client() -> LLMClient:
    """Get or create the LLMClient instance lazily."""
    global llm_client
    if llm_client is None:
        try:
            llm_client = LLMClient()
        except ValueError as e:
            logger.error(str(e))
            raise HTTPException(status_code=500, detail="Server LLM configuration missing")
    return llm_client


@app.on_event("startup")
async def startup_check():
    # Log presence (not value) of GROQ_API_KEY for easier ops debugging
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set at startup; LLM features will fail until configured")


@app.post("/api/")
async def analyze_data(request: Request):
    """
    Main API endpoint for data analysis.
    Processes questions.txt and optional data files to provide AI-powered analysis.
    """
    start_time = datetime.now()
    
    try:
        # Get the form data manually to handle dynamic field names
        form = await request.form()
        
        # Extract questions.txt file
        questions_file = form.get("questions.txt")
        if not questions_file:
            raise HTTPException(
                status_code=422, 
                detail="Missing questions.txt file. This file is required and must contain analysis questions."
            )
        
        # Extract all other files
        data_files = []
        for field_name, file_data in form.items():
            if field_name != "questions.txt" and hasattr(file_data, 'read'):
                data_files.append(file_data)
        
        # Read and parse questions
        questions_content = await questions_file.read()
        questions_text = questions_content.decode("utf-8")
        
        logger.info(f"Processing {len(data_files)} additional files with questions")
        
        # Parse questions and extract URLs if any
        parsed_questions, urls, output_format = await question_parser.parse_questions(questions_text)
        
        # Extract data sources information
        data_sources = question_parser.extract_data_sources(questions_text)
        
        # Process data from multiple sources
        all_data = {}
        
        # 1. Process uploaded data files
        for field_name, data_file in [(k, v) for k, v in form.items() if k != "questions.txt" and hasattr(v, 'read')]:
            try:
                file_content = await data_file.read()
                # Use the field name as filename (e.g., "data.csv", "image.png")
                processed_data = await data_processor.process_file(field_name, file_content)
                if processed_data is not None:
                    all_data[field_name] = processed_data
            except Exception as e:
                logger.warning(f"Failed to process file {field_name}: {str(e)}")
        
        # 2. Scrape data from URLs if any
        for url in urls:
            try:
                scraped_data = await web_scraper.scrape_url(url)
                if scraped_data is not None and not scraped_data.empty:
                    all_data[f"scraped_{url}"] = scraped_data
            except Exception as e:
                logger.warning(f"Failed to scrape URL {url}: {str(e)}")
        
        # 3. Process S3 data sources if any
        for s3_info in data_sources.get('s3_paths', []):
            try:
                s3_data = await data_processor.query_database({
                    'type': 's3',
                    'url': s3_info['path'],
                    's3_region': s3_info.get('region', 'us-east-1')
                })
                if s3_data is not None and not s3_data.empty:
                    all_data[f"s3_data_{len(all_data)}"] = s3_data
            except Exception as e:
                logger.warning(f"Failed to process S3 data {s3_info['path']}: {str(e)}")
        
        # Separate visualization and analysis questions for optimal processing
        viz_questions = []
        analysis_questions = []
        question_types = []  # Track which type each question is
        
        for question in parsed_questions:
            if "plot" in question.lower() or "chart" in question.lower() or "graph" in question.lower():
                viz_questions.append(question)
                question_types.append("visualization")
            else:
                analysis_questions.append(question)
                question_types.append("analysis")
        
        # Check timeout before processing
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 170:  # Leave 10 seconds buffer
            raise HTTPException(status_code=408, detail="Processing timeout exceeded")
        
        # Process analysis questions in batch (much more efficient)
        analysis_responses = []
        if analysis_questions:
            try:
                logger.info(f"Processing {len(analysis_questions)} analysis questions in batch")
                analysis_responses = await get_llm_client().analyze_questions_batch(analysis_questions, all_data)
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Fallback to individual processing
                for question in analysis_questions:
                    try:
                        response = await get_llm_client().analyze_question(question, all_data)
                        analysis_responses.append(response)
                    except Exception as individual_error:
                        logger.error(f"Error processing individual question: {str(individual_error)}")
                        analysis_responses.append(f"Error processing question: {str(individual_error)}")
        
        # Process visualization questions individually (they need special handling)
        viz_responses = []
        if viz_questions:
            for question in viz_questions:
                try:
                    # Check timeout for each visualization
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 170:
                        viz_responses.append("Processing timeout exceeded")
                        continue
                    
                    response = await viz_generator.generate_visualization(question, all_data)
                    viz_responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing visualization question: {str(e)}")
                    viz_responses.append(f"Error processing visualization: {str(e)}")
        
        # Merge responses back in original order
        responses = []
        analysis_idx = 0
        viz_idx = 0
        
        for question_type in question_types:
            if question_type == "analysis":
                if analysis_idx < len(analysis_responses):
                    responses.append(analysis_responses[analysis_idx])
                    analysis_idx += 1
                else:
                    responses.append("Error: Missing analysis response")
            else:  # visualization
                if viz_idx < len(viz_responses):
                    responses.append(viz_responses[viz_idx])
                    viz_idx += 1
                else:
                    responses.append("Error: Missing visualization response")
        
        # Format response according to requested format
        if output_format.lower() == "json_array":
            return responses
        elif output_format.lower() == "json_object":
            # Create object mapping questions to responses
            result = {}
            for question, response in zip(parsed_questions, responses):
                result[question] = response
            return result
        else:
            # Default to array format
            return responses
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    ) 