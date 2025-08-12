import pandas as pd
import json
import io
import os
import logging
from typing import Optional, Dict, Any
import duckdb
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles processing of various data formats including CSV, JSON, Parquet, and database connections.
    Follows the Single Responsibility Principle by focusing solely on data ingestion and processing.
    """
    
    def __init__(self):
        self.supported_extensions = {
            '.csv': self._process_csv,
            '.json': self._process_json,
            '.parquet': self._process_parquet,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.txt': self._process_text
        }
        
    async def process_file(self, filename: str, content: bytes) -> Optional[pd.DataFrame]:
        """
        Process a file based on its extension.
        
        Args:
            filename: Name of the file including extension
            content: File content as bytes
            
        Returns:
            Processed data as pandas DataFrame or None if processing fails
        """
        try:
            file_extension = self._get_file_extension(filename)
            
            if file_extension not in self.supported_extensions:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return None
                
            processor_func = self.supported_extensions[file_extension]
            return await processor_func(content, filename)
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            return None
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension in lowercase."""
        return os.path.splitext(filename.lower())[1]
    
    async def _process_csv(self, content: bytes, filename: str) -> pd.DataFrame:
        """Process CSV files with intelligent delimiter detection."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content_str = content.decode(encoding)
                    
                    # Try different delimiters
                    delimiters = [',', ';', '\t', '|']
                    
                    for delimiter in delimiters:
                        try:
                            df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
                            
                            # Check if parsing was successful (more than 1 column)
                            if len(df.columns) > 1:
                                logger.info(f"Successfully parsed CSV with delimiter '{delimiter}' and encoding '{encoding}'")
                                return df
                                
                        except Exception:
                            continue
                            
                except UnicodeDecodeError:
                    continue
            
            # Fallback to default pandas CSV reader
            return pd.read_csv(io.BytesIO(content))
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {filename}: {str(e)}")
            raise
    
    async def _process_json(self, content: bytes, filename: str) -> pd.DataFrame:
        """Process JSON files with various structures."""
        try:
            content_str = content.decode('utf-8')
            data = json.loads(content_str)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Single object or nested structure
                if all(isinstance(v, (list, dict)) for v in data.values()):
                    # Try to normalize nested JSON
                    return pd.json_normalize(data)
                else:
                    # Simple key-value pairs
                    return pd.DataFrame([data])
            else:
                # Simple values
                return pd.DataFrame({'value': [data]})
                
        except Exception as e:
            logger.error(f"Failed to process JSON file {filename}: {str(e)}")
            raise
    
    async def _process_parquet(self, content: bytes, filename: str) -> pd.DataFrame:
        """Process Parquet files."""
        try:
            # Use pyarrow to read parquet from bytes
            table = pq.read_table(io.BytesIO(content))
            return table.to_pandas()
            
        except Exception as e:
            logger.error(f"Failed to process Parquet file {filename}: {str(e)}")
            raise
    
    async def _process_excel(self, content: bytes, filename: str) -> pd.DataFrame:
        """Process Excel files."""
        try:
            # Read all sheets and combine if multiple
            excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
            
            if len(excel_data) == 1:
                # Single sheet
                return list(excel_data.values())[0]
            else:
                # Multiple sheets - combine with sheet name prefix
                combined_data = []
                for sheet_name, df in excel_data.items():
                    df['sheet_name'] = sheet_name
                    combined_data.append(df)
                return pd.concat(combined_data, ignore_index=True)
                
        except Exception as e:
            logger.error(f"Failed to process Excel file {filename}: {str(e)}")
            raise
    
    async def _process_text(self, content: bytes, filename: str) -> pd.DataFrame:
        """Process text files as single column data."""
        try:
            content_str = content.decode('utf-8')
            lines = content_str.strip().split('\n')
            return pd.DataFrame({'text': lines})
            
        except Exception as e:
            logger.error(f"Failed to process text file {filename}: {str(e)}")
            raise
    
    async def query_database(self, connection_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Query database using provided connection information.
        
        Args:
            connection_info: Dictionary containing database connection details
            
        Returns:
            Query results as pandas DataFrame
        """
        try:
            db_type = connection_info.get('type', '').lower()
            
            if db_type == 's3':
                return await self._query_s3_data(connection_info)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise
    
    async def _query_s3_data(self, connection_info: Dict[str, Any]) -> pd.DataFrame:
        """Query S3 data using DuckDB with s3 extension."""
        try:
            # Use DuckDB for S3 queries
            conn = duckdb.connect()
            
            # Install and load required extensions
            conn.execute("INSTALL httpfs;")
            conn.execute("LOAD httpfs;")
            conn.execute("INSTALL parquet;")
            conn.execute("LOAD parquet;")
            
            # Set S3 region if provided
            region = connection_info.get('s3_region', 'us-east-1')
            conn.execute(f"SET s3_region='{region}';")
            
            # Get the S3 path
            s3_path = connection_info.get('url') or connection_info.get('path')
            
            # Execute query - use read_parquet for parquet files
            if connection_info.get('query'):
                # Use custom query if provided
                query = connection_info['query']
            else:
                # Default query using read_parquet function
                query = f"SELECT * FROM read_parquet('{s3_path}')"
            
            logger.info(f"Executing S3 query: {query}")
            result = conn.execute(query).fetchdf()
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"S3 query failed: {str(e)}")
            raise 