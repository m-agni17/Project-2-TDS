import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class WebScraper:
    """
    Handles web scraping operations for extracting data from URLs.
    Follows the Single Responsibility Principle by focusing on web data extraction.
    """
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.headers = {
            'User-Agent': 'Data-Analyst-Agent/1.0 (Data Analysis Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def scrape_url(self, url: str) -> Optional[pd.DataFrame]:
        """
        Scrape data from a URL and return as DataFrame.
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped data as pandas DataFrame or None if scraping fails
        """
        try:
            logger.info(f"Scraping URL: {url}")
            
            session = await self.get_session()
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} error for URL: {url}")
                    return None
                
                content_type = response.headers.get('content-type', '').lower()
                content = await response.read()
                
                # Handle different content types
                if 'application/json' in content_type:
                    return await self._process_json_response(content, url)
                elif 'text/html' in content_type or 'text/xml' in content_type:
                    return await self._process_html_response(content, url)
                else:
                    # Try to detect content type from content
                    return await self._process_html_response(content, url)
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout while scraping URL: {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
    
    async def _process_json_response(self, content: bytes, url: str) -> pd.DataFrame:
        """Process JSON response from API."""
        try:
            content_str = content.decode('utf-8')
            data = json.loads(content_str)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle nested JSON structures
                if 'data' in data and isinstance(data['data'], list):
                    # Common API pattern
                    return pd.DataFrame(data['data'])
                elif 'results' in data and isinstance(data['results'], list):
                    # Another common API pattern
                    return pd.DataFrame(data['results'])
                else:
                    # Try to normalize the entire structure
                    return pd.json_normalize(data)
            else:
                return pd.DataFrame({'value': [data]})
                
        except Exception as e:
            logger.error(f"Failed to process JSON response from {url}: {str(e)}")
            raise
    
    async def _process_html_response(self, content: bytes, url: str) -> pd.DataFrame:
        """Process HTML response and extract comprehensive page content."""
        try:
            content_str = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(content_str, 'html.parser')
            
            # Extract the entire page content as structured data
            page_data = await self._extract_comprehensive_page_data(soup, url)
            
            return page_data
            
        except Exception as e:
            logger.error(f"Failed to process HTML response from {url}: {str(e)}")
            raise
    
    async def _extract_comprehensive_page_data(self, soup: BeautifulSoup, url: str) -> pd.DataFrame:
        """Extract comprehensive data from the entire webpage."""
        try:
            all_content = []
            
            # 1. Extract page metadata
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else 'No title'
            
            # 2. Extract ALL tables with better parsing
            tables = soup.find_all('table')
            logger.info(f"Found {len(tables)} tables on the page")
            
            for i, table in enumerate(tables):
                try:
                    # Get table caption or nearby heading for context
                    table_context = self._get_table_context(table)
                    
                    rows = table.find_all('tr')
                    for row_idx, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        if cells:
                            row_data = []
                            for cell in cells:
                                cell_text = cell.get_text(strip=True)
                                cell_text = re.sub(r'\s+', ' ', cell_text)
                                row_data.append(cell_text)
                            
                            # Create a record for each row
                            if any(cell.strip() for cell in row_data):
                                record = {
                                    'content_type': 'table_row',
                                    'table_index': i,
                                    'table_context': table_context,
                                    'row_index': row_idx,
                                    'row_data': ' | '.join(row_data),
                                    'cell_count': len(row_data),
                                    'source_url': url,
                                    'page_title': title_text
                                }
                                all_content.append(record)
                                
                except Exception as e:
                    logger.warning(f"Failed to extract table {i}: {str(e)}")
                    continue
            
            # 3. Extract structured lists
            lists = soup.find_all(['ul', 'ol', 'dl'])
            for i, lst in enumerate(lists):
                try:
                    if lst.name == 'dl':
                        # Definition list
                        dts = lst.find_all('dt')
                        dds = lst.find_all('dd')
                        for dt, dd in zip(dts, dds):
                            record = {
                                'content_type': 'definition',
                                'list_index': i,
                                'term': dt.get_text(strip=True),
                                'definition': dd.get_text(strip=True),
                                'source_url': url,
                                'page_title': title_text
                            }
                            all_content.append(record)
                    else:
                        # Regular list
                        items = lst.find_all('li')
                        for j, item in enumerate(items):
                            record = {
                                'content_type': 'list_item',
                                'list_index': i,
                                'item_index': j,
                                'content': item.get_text(strip=True),
                                'source_url': url,
                                'page_title': title_text
                            }
                            all_content.append(record)
                except Exception as e:
                    continue
            
            # 4. Extract paragraphs with meaningful content
            paragraphs = soup.find_all('p')
            for i, para in enumerate(paragraphs):
                text = para.get_text(strip=True)
                if text and len(text) > 50:  # Only meaningful paragraphs
                    record = {
                        'content_type': 'paragraph',
                        'paragraph_index': i,
                        'content': text,
                        'source_url': url,
                        'page_title': title_text
                    }
                    all_content.append(record)
            
            # 5. Extract headings for structure
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for i, heading in enumerate(headings):
                text = heading.get_text(strip=True)
                if text:
                    record = {
                        'content_type': 'heading',
                        'heading_level': heading.name,
                        'heading_index': i,
                        'content': text,
                        'source_url': url,
                        'page_title': title_text
                    }
                    all_content.append(record)
            
            if all_content:
                df = pd.DataFrame(all_content)
                logger.info(f"Extracted {len(df)} content items from {url}")
                return df
            else:
                # Fallback: return basic page info
                return pd.DataFrame([{
                    'content_type': 'page_info',
                    'content': f"Page: {title_text}",
                    'source_url': url,
                    'page_title': title_text
                }])
                
        except Exception as e:
            logger.error(f"Error in comprehensive page extraction: {str(e)}")
            return pd.DataFrame([{
                'content_type': 'error',
                'content': f"Failed to extract content: {str(e)}",
                'source_url': url
            }])
    
    def _get_table_context(self, table) -> str:
        """Get context for a table (caption, nearby headings, etc.)."""
        try:
            # Try to find table caption
            caption = table.find('caption')
            if caption:
                return caption.get_text(strip=True)
            
            # Look for nearby headings
            parent = table.parent
            if parent:
                # Look for preceding heading
                for sibling in parent.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    heading_text = sibling.get_text(strip=True)
                    if heading_text:
                        return heading_text
            
            return "Table"
            
        except:
            return "Table"
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # Schedule cleanup for next event loop iteration
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.close_session())
            except:
                pass 