"""
FastAPI Data Analyst Agent - Main Application
Accepts POST requests with question files and optional data attachments.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiofiles
import json
from typing import List, Dict, Any, Optional
import tempfile
import os
import traceback
from contextlib import asynccontextmanager

# Import our modules
from data_orchestrator import process_data_analysis_request, DataAnalysisError
from config import get_max_file_size, get_response_timeout, get_debug_mode

# Global variables for app state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting Data Analyst Agent API...")
    app_state["startup_complete"] = True
    
    yield
    
    # Shutdown
    print("Shutting down Data Analyst Agent API...")

# Create FastAPI application
app = FastAPI(
    title="Data Analyst Agent",
    description="API that uses LLMs to source, prepare, analyze, and visualize data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Data Analyst Agent API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "startup_complete": app_state.get("startup_complete", False),
        "max_file_size_mb": get_max_file_size() // (1024 * 1024),
        "response_timeout_seconds": get_response_timeout()
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main API endpoint for data analysis.
    
    Accepts multipart/form-data with:
    - questions.txt (required): Contains the analysis questions
    - Additional files (optional): CSV, images, or other data files
    
    Returns:
    - JSON array or object with analysis results
    """
    try:
        # Validate request
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Find questions.txt file
        questions_file = None
        additional_files = {}
        
        max_size = get_max_file_size()
        
        for file in files:
            # Check file size
            file_content = await file.read()
            if len(file_content) > max_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} exceeds maximum size of {max_size // (1024*1024)}MB"
                )
            
            # Reset file position
            await file.seek(0)
            
            # Identify file type
            if file.filename and file.filename.lower() == "questions.txt":
                questions_file = file
            else:
                # Store additional files
                if file.filename:
                    additional_files[file.filename] = file_content
        
        if questions_file is None:
            raise HTTPException(
                status_code=400, 
                detail="questions.txt file is required"
            )
        
        # Read questions file content
        questions_content = await questions_file.read()
        if isinstance(questions_content, bytes):
            questions_content = questions_content.decode('utf-8')
        
        # Process the analysis request with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    process_data_analysis_request,
                    questions_content,
                    additional_files
                ),
                timeout=get_response_timeout()
            )
            
            # Return result as JSON
            print(result)
            return JSONResponse(content=result)
            
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408, 
                detail=f"Analysis timed out after {get_response_timeout()} seconds"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error for debugging
        if get_debug_mode():
            print(f"Unexpected error: {e}")
            print(traceback.format_exc())
        
        # Return generic error response
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during analysis: {str(e)}"
        )

@app.post("/api/analyze")
async def analyze_data_alternative(
    questions: UploadFile = File(..., description="Questions text file"),
    data_file: Optional[UploadFile] = File(None, description="Optional data CSV file"),
    image_file: Optional[UploadFile] = File(None, description="Optional image file")
):
    """
    Alternative endpoint with explicit file parameters.
    Useful for testing and documentation.
    """
    files = [questions]
    if data_file:
        files.append(data_file)
    if image_file:
        files.append(image_file)
    
    return await analyze_data(files)

@app.get("/api/example")
async def get_example_request():
    """
    Return an example of how to use the API.
    """
    return {
        "description": "Data Analyst Agent API Usage",
        "curl_example": '''
curl -X POST "http://localhost:8000/api/" \\
  -F "questions.txt=@questions.txt" \\
  -F "data.csv=@data.csv" \\
  -F "image.png=@image.png"
        '''.strip(),
        "questions_example": """
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
        """.strip(),
        "expected_response": [1, "Titanic", 0.485782, "data:image/png;base64,..."]
    }

@app.exception_handler(DataAnalysisError)
async def data_analysis_exception_handler(request, exc: DataAnalysisError):
    """Handle custom data analysis errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "data_analysis_error"}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle value errors from data processing."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "value_error"}
    )

# Error handling for common issues
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    """Middleware for consistent error handling."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        if get_debug_mode():
            print(f"Middleware caught error: {e}")
            print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Data Analyst Agent...")
    print(f"Server will run on: http://0.0.0.0:8000")
    print(f"Max file size: {get_max_file_size() // (1024*1024)}MB")
    print(f"Response timeout: {get_response_timeout()}s")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 