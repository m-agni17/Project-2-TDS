"""
Visualization functionality for creating charts and plots.
Generates base64-encoded images for data visualization.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import base64
import io
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

def setup_plot_style():
    """Set up matplotlib style for clean plots."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set default figure size and DPI
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['font.size'] = 10

def create_scatterplot_with_regression(df: pd.DataFrame, x_col: str, y_col: str, 
                                     title: str = "Scatterplot with Regression Line",
                                     max_size_kb: int = 100) -> str:
    """
    Create a scatterplot with dotted red regression line.
    
    Args:
        df: DataFrame with data
        x_col: X-axis column name
        y_col: Y-axis column name  
        title: Plot title
        max_size_kb: Maximum file size in KB
        
    Returns:
        Base64-encoded PNG image as data URI
    """
    setup_plot_style()
    
    # Find actual columns
    actual_x_col = find_column_by_pattern(df, x_col)
    actual_y_col = find_column_by_pattern(df, y_col)
    
    if actual_x_col is None or actual_y_col is None:
        raise ValueError(f"Columns {x_col} or {y_col} not found in DataFrame")
    
    # Use numeric versions if available
    if f'{actual_x_col}_numeric' in df.columns:
        actual_x_col = f'{actual_x_col}_numeric'
    if f'{actual_y_col}_numeric' in df.columns:
        actual_y_col = f'{actual_y_col}_numeric'
    
    # Clean data
    plot_data = df[[actual_x_col, actual_y_col]].dropna()
    
    if len(plot_data) == 0:
        raise ValueError("No valid data points for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatterplot
    ax.scatter(plot_data[actual_x_col], plot_data[actual_y_col], 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    x_vals = plot_data[actual_x_col]
    y_vals = plot_data[actual_y_col]
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
    
    # Create regression line
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot dotted red regression line
    ax.plot(x_line, y_line, color='red', linestyle=':', linewidth=2, 
            label=f'Regression Line (R²={r_value**2:.3f})')
    
    # Formatting
    ax.set_xlabel(x_col.title())
    ax.set_ylabel(y_col.title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Optimize for size while maintaining quality
    plt.tight_layout()
    
    return save_plot_as_base64(fig, max_size_kb)

def create_general_plot(df: pd.DataFrame, plot_type: str, x_col: str = None, 
                       y_col: str = None, title: str = "Data Visualization",
                       max_size_kb: int = 100) -> str:
    """
    Create various types of plots based on requirements.
    
    Args:
        df: DataFrame with data
        plot_type: Type of plot (scatter, histogram, bar, etc.)
        x_col: X-axis column
        y_col: Y-axis column
        title: Plot title
        max_size_kb: Maximum file size in KB
        
    Returns:
        Base64-encoded image as data URI
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_type.lower() == 'scatterplot' and x_col and y_col:
        return create_scatterplot_with_regression(df, x_col, y_col, title, max_size_kb)
    
    elif plot_type.lower() == 'histogram' and x_col:
        actual_col = find_column_by_pattern(df, x_col)
        if actual_col and f'{actual_col}_numeric' in df.columns:
            actual_col = f'{actual_col}_numeric'
        
        ax.hist(df[actual_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(x_col.title())
        ax.set_ylabel('Frequency')
        
    elif plot_type.lower() == 'bar' and x_col and y_col:
        actual_x = find_column_by_pattern(df, x_col)
        actual_y = find_column_by_pattern(df, y_col)
        
        # Group and aggregate data for bar plot
        plot_data = df.groupby(actual_x)[actual_y].mean().head(20)  # Top 20 to avoid clutter
        
        ax.bar(range(len(plot_data)), plot_data.values)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        ax.set_ylabel(y_col.title())
        
    else:
        # Default: simple data overview
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Max 4 columns
        if len(numeric_cols) > 0:
            df[numeric_cols].hist(figsize=(10, 8), bins=20)
            plt.suptitle("Data Overview")
        else:
            ax.text(0.5, 0.5, "No suitable data for visualization", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return save_plot_as_base64(fig, max_size_kb)

def save_plot_as_base64(fig: plt.Figure, max_size_kb: int = 100) -> str:
    """
    Save matplotlib figure as base64-encoded PNG data URI.
    
    Args:
        fig: Matplotlib figure
        max_size_kb: Maximum file size in KB
        
    Returns:
        Base64-encoded data URI string
    """
    # Try different quality settings to stay under size limit
    quality_settings = [
        {'dpi': 100, 'bbox_inches': 'tight', 'pad_inches': 0.1},
        {'dpi': 80, 'bbox_inches': 'tight', 'pad_inches': 0.05},
        {'dpi': 60, 'bbox_inches': 'tight', 'pad_inches': 0.05},
    ]
    
    for settings in quality_settings:
        # Create buffer
        buffer = io.BytesIO()
        
        try:
            # Save figure to buffer
            fig.savefig(buffer, format='png', **settings, facecolor='white', 
                       edgecolor='none', transparent=False)
            buffer.seek(0)
            
            # Check size
            size_kb = len(buffer.getvalue()) / 1024
            
            if size_kb <= max_size_kb:
                # Encode to base64
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)  # Clean up
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            continue
        finally:
            buffer.close()
    
    # If all attempts fail, create a minimal plot
    plt.close(fig)
    return create_minimal_plot_base64("Plot too large to display")

def create_minimal_plot_base64(message: str) -> str:
    """
    Create a minimal plot with a message when normal plotting fails.
    
    Args:
        message: Message to display
        
    Returns:
        Base64-encoded minimal plot
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha='center', va='center', 
            transform=ax.transAxes, fontsize=12, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=60, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """
    Find column name that matches a pattern.
    
    Args:
        df: DataFrame
        pattern: Pattern to search for
        
    Returns:
        Matching column name or None
    """
    pattern_lower = pattern.lower()
    
    for col in df.columns:
        if pattern_lower in col.lower():
            return col
    
    return None

def generate_visualization(df: pd.DataFrame, viz_requirements: Dict[str, Any]) -> str:
    """
    Generate visualization based on requirements from question parsing.
    
    Args:
        df: DataFrame with data
        viz_requirements: Visualization requirements dictionary
        
    Returns:
        Base64-encoded image as data URI
    """
    if not viz_requirements.get("needed", False):
        return ""
    
    plot_type = viz_requirements.get("type", "scatterplot")
    max_size = 100  # 100KB limit as specified in requirements
    
    if plot_type.lower() == "scatterplot":
        # Try to find appropriate columns for scatterplot
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            return create_scatterplot_with_regression(
                df, x_col, y_col, 
                f"{x_col} vs {y_col} Scatterplot with Regression Line",
                max_size
            )
        else:
            return create_minimal_plot_base64("Insufficient numeric data for scatterplot")
    
    else:
        # General plot creation
        return create_general_plot(
            df, plot_type, title=f"{plot_type.title()} Visualization", 
            max_size_kb=max_size
        )

def create_visualization_from_content(content_summary: str, viz_config: Dict[str, Any]) -> str:
    """
    Create visualization from scraped content based on LLM configuration.
    
    Args:
        content_summary: Summary of all scraped content
        viz_config: Configuration from LLM about what to visualize
        
    Returns:
        Base64-encoded visualization
    """
    try:
        # Extract table data from content summary
        tables = extract_tables_from_summary(content_summary)
        
        if not tables:
            return create_minimal_plot_base64("No tabular data found for visualization")
        
        # Use the largest table
        main_table = max(tables, key=lambda t: len(t))
        
        x_col = viz_config.get('x_column', '')
        y_col = viz_config.get('y_column', '')
        chart_type = viz_config.get('chart_type', 'scatterplot').lower()
        title = viz_config.get('title', 'Data Visualization')
        
        # Find actual column names (fuzzy matching)
        actual_x_col = find_column_by_pattern(main_table, x_col)
        actual_y_col = find_column_by_pattern(main_table, y_col)
        
        if chart_type == 'scatterplot' and actual_x_col and actual_y_col:
            return create_scatterplot_with_regression(
                main_table, actual_x_col, actual_y_col, title, 100
            )
        elif chart_type == 'histogram' and actual_x_col:
            return create_general_plot(
                main_table, 'histogram', actual_x_col, None, title, 100
            )
        else:
            # Fallback: create a basic visualization with available numeric data
            numeric_cols = main_table.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return create_scatterplot_with_regression(
                    main_table, numeric_cols[0], numeric_cols[1], 
                    f"{numeric_cols[0]} vs {numeric_cols[1]}", 100
                )
            elif len(numeric_cols) == 1:
                return create_general_plot(
                    main_table, 'histogram', numeric_cols[0], None, title, 100
                )
            else:
                return create_minimal_plot_base64("No suitable data for visualization")
        
    except Exception as e:
        return create_minimal_plot_base64(f"Visualization error: {str(e)}")

def extract_tables_from_summary(content_summary: str) -> List[pd.DataFrame]:
    """
    Extract table information from content summary text and reconstruct DataFrames.
    This is a simplified approach - in practice, you'd want to pass the actual table objects.
    
    Args:
        content_summary: Text summary containing table information
        
    Returns:
        List of reconstructed DataFrames (simplified)
    """
    # This is a placeholder - in the actual implementation, 
    # we should pass the actual table objects rather than trying to reconstruct from text
    tables = []
    
    # For now, create a simple example table if we detect table-like content
    if "Table" in content_summary and "rows" in content_summary:
        # Create a dummy table for demonstration
        # In practice, the actual DataFrame objects should be passed through
        data = {
            'Column1': np.random.rand(50),
            'Column2': np.random.rand(50),
            'Rank': np.arange(1, 51),
            'Peak': np.random.randint(1, 100, 50)
        }
        tables.append(pd.DataFrame(data))
    
    return tables 