# Data Analyst Agent API

A powerful AI-powered data analysis API that uses LLMs to source, prepare, analyze, and visualize arbitrary datasets based on incoming tasks.

## Features

- **Multi-format Data Processing**: Supports CSV, JSON, Parquet, Excel, and database connections
- **Web Scraping**: Automatically scrapes data from URLs mentioned in questions
- **AI-Powered Analysis**: Uses Groq's LLM for intelligent data analysis
- **Visualization Generation**: Creates charts and graphs as Base64 data URIs
- **Flexible Output Formats**: Supports JSON array and JSON object response formats
- **Timeout Management**: 3-minute processing limit with progress tracking
- **Error Handling**: Comprehensive error handling with detailed logging

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Data-Analyst-Agent-API
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

4. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Main Endpoint

**POST** `/api/`

#### Request Format

The API accepts multipart/form-data with the following requirements:

- **Required**: `questions.txt` - Text file containing analysis questions
- **Optional**: Additional data files (CSV, JSON, Parquet, Excel, etc.)

#### Request Body Requirements

1. **questions.txt**: Must contain one or more analysis questions in plain text
2. **Data files**: Any number of structured data files
3. **URLs**: Can be included in questions.txt for automatic scraping

#### Example Request

```bash
curl "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

#### Response Format

The response format is determined by the content of `questions.txt`:

- **JSON Array**: `[result1, result2, result3]`
- **JSON Object**: `{"question1": "result1", "question2": "result2"}`

### Health Check

**GET** `/health`

Returns server health status.

## Usage Examples

### Example 1: Wikipedia Dataset Analysis

**questions.txt**:
```
Scrape the list of highest-grossing films from:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer in JSON array format:
1. How many $2B+ movies were released before 2000?
2. Which is the earliest film that grossed over $1.5B?
3. What's the correlation between Rank and Peak?
4. Draw a scatterplot of Rank vs Peak with a dotted red regression line.
   Return as Base64 PNG data URI under 100,000 bytes.
```

**Expected Response**:
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

### Example 2: S3 Dataset Analysis

**questions.txt**:
```
Using the dataset from:
s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1

Answer in JSON object format:
1. Which high court disposed the most cases from 2019–2022?
2. What's the regression slope of (date_of_registration → decision_date) by year for court=33_10?
3. Plot the year vs delay (days) from question #2 as scatterplot with regression line.
   Return as Base64 data URI under 100,000 bytes.
```

**Expected Response**:
```json
{
  "Which high court disposed the most cases from 2019–2022?": "XYZ High Court",
  "What's the regression slope of (date_of_registration → decision_date) by year for court=33_10?": 1.27,
  "Plot the year vs delay (days) from question #2 as scatterplot with regression line": "data:image/webp;base64,..."
}
```

## Architecture

The API follows SOLID principles with a modular architecture:

### Core Components

1. **main.py**: FastAPI application and main endpoint
2. **data_processor.py**: Handles data ingestion from multiple formats
3. **web_scraper.py**: Scrapes data from URLs
4. **llm_client.py**: Interfaces with Groq LLM for analysis
5. **visualization_generator.py**: Creates charts and converts to Base64
6. **question_parser.py**: Parses questions.txt and extracts metadata

### Data Flow

1. **Request Processing**: Parse uploaded files and questions
2. **Data Ingestion**: Process data files and scrape URLs
3. **Question Analysis**: Extract questions and determine output format
4. **AI Analysis**: Use LLM to analyze data and answer questions
5. **Visualization**: Generate charts for visualization questions
6. **Response Formatting**: Format results according to requested format

## Supported Data Sources

### File Formats
- **CSV**: Comma-separated values with intelligent delimiter detection
- **JSON**: Various JSON structures with automatic normalization
- **Parquet**: High-performance columnar format
- **Excel**: .xlsx and .xls files with multi-sheet support
- **Text**: Plain text files

### Web Sources
- **HTML Tables**: Automatic table extraction from web pages
- **JSON APIs**: Direct API response processing
- **CSV Downloads**: Direct CSV file downloads
- **Structured Data**: JSON-LD and microdata extraction

### Database Connections
- **PostgreSQL**: Full PostgreSQL support
- **MySQL**: MySQL database connections
- **SQLite**: Local SQLite databases
- **S3**: Amazon S3 data with DuckDB integration

## Visualization Features

### Supported Chart Types
- **Scatter Plots**: With optional regression lines
- **Line Charts**: Time series and trend analysis
- **Bar Charts**: Categorical data visualization
- **Correlation Heatmaps**: Statistical relationships
- **Distribution Plots**: Data distribution analysis
- **Box Plots**: Statistical summaries

### Output Formats
- **PNG**: High-quality raster images
- **WebP**: Compressed web-optimized format
- **JPEG**: Compressed format for photographs

All visualizations are automatically optimized to stay under 100KB.

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Required Groq API key for LLM access

### Timeout Settings
- Default processing timeout: 3 minutes
- Individual operation timeout: 30 seconds
- Visualization generation timeout: 60 seconds

## Error Handling

The API includes comprehensive error handling:

- **422 Unprocessable Entity**: Missing questions.txt file
- **408 Request Timeout**: Processing exceeded 3-minute limit
- **500 Internal Server Error**: Unexpected server errors

All errors include detailed messages for debugging.

## Performance Considerations

- **Concurrent Processing**: Multiple datasets processed in parallel
- **Memory Management**: Efficient handling of large datasets
- **Timeout Management**: Prevents long-running requests
- **Resource Cleanup**: Automatic cleanup of temporary resources

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Data Processors
Extend the `DataProcessor` class to support additional file formats.

### Adding New Visualization Types
Extend the `VisualizationGenerator` class to support new chart types.

## Troubleshooting

### Common Issues

1. **GROQ_API_KEY not set**: Ensure the environment variable is properly configured
2. **Timeout errors**: Large datasets may require optimization
3. **File format errors**: Ensure uploaded files are in supported formats
4. **Memory issues**: Large datasets may require server optimization

### Logging

The API uses structured logging. Check logs for detailed error information.

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create a GitHub issue or contact the development team. 