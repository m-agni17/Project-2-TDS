"""
Generic data scraping functionality for extracting content from any web source.
Extracts text, tables, and structured data from websites.
"""

from typing import Dict, List, Any, Optional
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
import json

def create_session() -> requests.Session:
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    return session

def fetch_web_content(url: str, timeout: int = 30) -> str:
    """
    Fetch content from a URL with error handling.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Raw HTML content as string
    """
    session = create_session()
    
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch content from {url}: {str(e)}")

def extract_all_tables(html_content: str, url: str) -> List[pd.DataFrame]:
    """
    Extract all tables from HTML content regardless of website type.
    
    Args:
        html_content: Raw HTML content
        url: Source URL for context
        
    Returns:
        List of pandas DataFrames containing table data
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all tables on the page
    tables = soup.find_all('table')
    
    if not tables:
        return []  # Return empty list instead of raising error
    
    dataframes = []
    
    for i, table in enumerate(tables):
        try:
            # Extract table to DataFrame using pandas
            df = pd.read_html(str(table), header=0)[0]
            
            # Clean column names
            df.columns = [clean_column_name(col) for col in df.columns]
            
            # Basic data cleaning
            df = clean_table_data(df)
            
            if len(df) > 0:  # Only add non-empty tables
                dataframes.append(df)
                
        except Exception as e:
            print(f"Warning: Failed to parse table {i+1}: {str(e)}")
            continue
    
    return dataframes

def clean_column_name(name: str) -> str:
    """Clean and normalize column names."""
    if isinstance(name, tuple):
        # Handle multi-level column names
        name = ' '.join(str(x) for x in name if str(x) != 'nan')
    
    name = str(name).strip()
    # Remove common Wikipedia column artifacts
    name = re.sub(r'\[.*?\]', '', name)  # Remove reference brackets
    name = re.sub(r'\s+', ' ', name)     # Normalize spaces
    return name.strip()

def clean_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean table data from any website.
    
    Args:
        df: Raw DataFrame from any table
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Clean each column
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove common reference markers and formatting
            df[col] = df[col].astype(str).str.replace(r'\[.*?\]', '', regex=True)  # [1], [2], etc.
            df[col] = df[col].str.replace(r'\(.*?\)', '', regex=True)  # (notes)
            
            # Remove extra whitespace and line breaks
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\n+', ' ', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove common HTML entities
            df[col] = df[col].str.replace(r'&nbsp;', ' ', regex=True)
            df[col] = df[col].str.replace(r'&amp;', '&', regex=True)
    
    # Try to convert numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to extract numbers from various formats
            numeric_values = df[col].str.extract(r'([\d,\.]+)', expand=False)
            if numeric_values.notna().any():
                try:
                    # Convert to numeric, handling commas
                    clean_numeric = pd.to_numeric(
                        numeric_values.str.replace(',', ''), 
                        errors='coerce'
                    )
                    if clean_numeric.notna().sum() > len(df) * 0.3:  # If > 30% are numeric
                        df[f'{col}_numeric'] = clean_numeric
                except:
                    pass
    
    return df

def extract_website_content(url: str) -> Dict[str, Any]:
    """
    Extract comprehensive content from any website.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary containing all extracted content
    """
    try:
        html_content = fetch_web_content(url)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract various content types
        content = {
            'url': url,
            'title': extract_page_title(soup),
            'text_content': extract_text_content(soup),
            'tables': extract_all_tables(html_content, url),
            'links': extract_links(soup, url),
            'images': extract_images(soup, url),
            'lists': extract_lists(soup),
            'metadata': extract_metadata(soup)
        }
        
        return content
        
    except Exception as e:
        raise ValueError(f"Failed to scrape content from {url}: {str(e)}")

def extract_page_title(soup: BeautifulSoup) -> str:
    """Extract page title."""
    title_tag = soup.find('title')
    return title_tag.text.strip() if title_tag else "Unknown Title"

def extract_text_content(soup: BeautifulSoup) -> str:
    """Extract main text content, removing scripts and styles."""
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    
    # Limit text size to prevent overwhelming the LLM
    return text

def extract_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Extract all links from the page."""
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.text.strip()
        
        # Convert relative URLs to absolute
        if href.startswith('/'):
            href = urljoin(base_url, href)
        elif href.startswith('#'):
            continue  # Skip anchor links
        
        if text and href.startswith('http'):
            links.append({'text': text, 'url': href})
    
    return links # Limit to first 50 links

def extract_images(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """Extract image information."""
    images = []
    for img in soup.find_all('img', src=True):
        src = img['src']
        alt = img.get('alt', '')
        
        # Convert relative URLs to absolute
        if src.startswith('/'):
            src = urljoin(base_url, src)
        
        if src.startswith('http'):
            images.append({'src': src, 'alt': alt})
    
    return images[:20]  # Limit to first 20 images

def extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract ordered and unordered lists."""
    lists = []
    
    for list_type in ['ul', 'ol']:
        for lst in soup.find_all(list_type):
            items = [li.text.strip() for li in lst.find_all('li')]
            if items:
                lists.append({
                    'type': 'ordered' if list_type == 'ol' else 'unordered',
                    'items': items
                })
    
    return lists  # Limit to first 10 lists

def extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract page metadata."""
    metadata = {}
    
    # Meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        if name and content:
            metadata[name] = content
    
    # Headings
    headings = []
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        headings.append(h.text.strip())
    
    metadata['headings'] = headings  # First 20 headings
    
    return metadata

def scrape_data_from_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Scrape comprehensive data from multiple URLs.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        Dictionary mapping URLs to scraped content
    """
    results = {}
    
    for url in urls:
        try:
            # Use generic content extraction for all websites
            content = extract_website_content(url)
            results[url] = content
            
            print(f"Successfully scraped {url} - Title: {content.get('title', 'Unknown')}")
            
        except Exception as e:
            print(f"Warning: Failed to scrape {url}: {str(e)}")
            results[url] = {
                'url': url,
                'error': str(e),
                'title': 'Failed to load',
                'text_content': f"Error loading content from {url}: {str(e)}",
                'tables': [],
                'links': [],
                'images': [],
                'lists': [],
                'metadata': {}
            }
        
        # Be respectful - add delay between requests
        time.sleep(1)
    
    return results 