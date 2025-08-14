# Generic Data Analyst Agent

A FastAPI-based data analysis agent that uses LLMs to scrape any website, analyze content, and answer questions about the data. Built with a function-first architecture following SOLID principles adapted for Python.

## Features

- 🤖 **LLM-Powered Analysis**: Uses AI to understand and analyze content from any website
- 🌐 **Universal Web Scraping**: Extracts text, tables, lists, and metadata from any website
- 📊 **Generic Data Analysis**: LLM-driven analysis that adapts to any data type or domain
- 📈 **Smart Visualization**: Automatically generates appropriate charts based on question context
- ⚡ **Fast API**: RESTful API with file upload support and 3-minute response guarantee  
- 🏗️ **Modular Architecture**: Function-first design for maintainability and scalability

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key (get from [Groq Console](https://console.groq.com/))

### Installation

1. **Clone and Setup**
   ```bash
   cd Project-2
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp config.env.template .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Start the Server**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## Test the Generic Capabilities

```bash
# Test with the generic sample (countries population data)
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@sample_questions_generic.txt"

# Test with the original films example  
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@sample_questions.txt"
```

## API Usage

### Main Endpoint

```bash
POST /api/
```

**Request Format**: `multipart/form-data`

- `questions.txt` (required): Text file containing analysis questions
- Additional files (optional): CSV, images, or other data files

### Example Request

```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Example Questions File

```text
Scrape content from this website: https://en.wikipedia.org/wiki/List_of_countries_by_population

Answer the following questions based on the scraped content:

1. What are the top 5 most populous countries according to the data?
2. What is the total population of the United States?
3. How many countries have a population greater than 100 million?
4. What percentage of world population does China represent?
5. Create a visualization showing the relationship between any two numerical columns found in the data. Return as a base-64 encoded data URI under 100,000 bytes.
```

### Example Response

```json
["China, India, United States, Indonesia, Pakistan", "331 million", "14", "18.5%", "data:image/png;base64,iVBORw0KG..."]
```

## Architecture

The system follows a function-first approach with clear separation of concerns:

```
├── main.py                 # FastAPI application and endpoints
├── config.py              # Configuration management
├── question_parser.py     # LLM-based question parsing
├── data_scraper.py        # Web scraping functionality
├── data_analyzer.py       # Statistical analysis and LLM insights
├── visualizer.py          # Chart generation and encoding
└── data_orchestrator.py   # Main coordination logic
```

### Key Principles

- **Function-First**: Pure functions over complex classes
- **Single Responsibility**: Each module has a focused purpose
- **Dependency Injection**: Dependencies passed as parameters
- **Error Handling**: Graceful fallbacks and meaningful errors
- **Scalability**: Modular design for easy extension

## Supported Analysis Types

### Data Sources
- **Any Website**: Universal scraping that adapts to any site structure
- **Tables**: Automatic detection and extraction from any webpage
- **Text Content**: Full text analysis from any source
- **Lists and Metadata**: Structured content extraction
- **CSV Files**: Direct upload and processing
- **Images**: Basic processing and metadata extraction

### Analysis Capabilities  
- **Universal LLM Analysis**: AI-powered analysis that adapts to any domain
- **Question Answering**: Natural language questions about any data
- **Pattern Recognition**: Finds insights in any type of content
- **Data Relationships**: Discovers connections in scraped data
- **Statistical Analysis**: When numerical data is available

### Visualization Features
- **Context-Aware Charts**: Automatically chooses appropriate visualization
- **Dynamic Adaptation**: Adjusts to available data types
- **Size Optimization**: Automatic compression under 100KB
- **Base64 Encoding**: Direct embedding in JSON responses  
- **Multiple Chart Types**: Scatter plots, histograms, bar charts, line charts

## Configuration

Environment variables in `.env`:

```bash
# Required
GROQ_API_KEY=your_api_key_here

# Optional
GROQ_MODEL=mixtral-8x7b-32768    # LLM model to use
HOST=0.0.0.0                     # Server host
PORT=8000                        # Server port
MAX_FILE_SIZE=10485760           # 10MB file size limit
RESPONSE_TIMEOUT=180             # 3 minute timeout
DEBUG=false                      # Debug mode
```

## API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Examples**: http://localhost:8000/api/example

## Error Handling

The API provides structured error responses:

```json
{
  "detail": "Error description",
  "error_type": "data_analysis_error"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (missing files, invalid format)
- `408`: Timeout (analysis took > 3 minutes)
- `413`: File too large
- `422`: Data analysis error
- `500`: Internal server error

## Performance Considerations

- **Response Time**: 3-minute maximum per request
- **File Size**: 10MB limit per file
- **Concurrent Requests**: Handled via async/await
- **Memory Management**: Streaming file processing
- **Visualization Size**: 100KB limit for base64 images

## Development

### Running in Debug Mode

```bash
DEBUG=true python main.py
```

### Adding New Analysis Types

1. **Extend Question Parser**: Add patterns in `question_parser.py`
2. **Add Data Source**: Implement scraper in `data_scraper.py`
3. **Create Analyzer**: Add analysis logic in `data_analyzer.py`
4. **Update Orchestrator**: Wire components in `data_orchestrator.py`

### Testing

The system is designed for easy testing with pure functions:

```python
# Example: Test data analysis
from data_analyzer import count_movies_by_criteria
import pandas as pd

df = pd.DataFrame(...)  # Your test data
result = count_movies_by_criteria(df, min_gross=2000000000, max_year=2000)
assert result == expected_value
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations

- Set `DEBUG=false`
- Configure proper CORS origins
- Add rate limiting
- Set up proper logging
- Use environment-specific configurations
- Consider load balancing for high traffic

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Ensure `.env` file exists with valid API key

2. **"Analysis timed out"**
   - Check internet connection for scraping
   - Consider increasing `RESPONSE_TIMEOUT`

3. **"No Wikipedia tables found"**
   - Verify URL is accessible
   - Check if page structure changed

4. **Visualization too large**
   - Reduce data points or image quality
   - Check base64 output size

### Logging

Enable debug mode for detailed logs:

```bash
DEBUG=true python main.py
```

## Contributing

1. Follow the function-first architecture principles
2. Add comprehensive error handling
3. Include type hints for all functions
4. Keep functions under 50 lines when possible
5. Add docstrings for all public functions

## License

This project is built for educational and evaluation purposes.

---

**Built with ❤️ using FastAPI, Groq, and modern Python practices.** 