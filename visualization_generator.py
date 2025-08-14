import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import io
import base64
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

class VisualizationGenerator:
    """
    Generates visualizations and converts them to Base64 data URIs.
    Follows the Single Responsibility Principle by focusing on chart creation and encoding.
    """
    
    def __init__(self):
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure default figure settings
        self.default_figsize = (10, 6)
        self.default_dpi = 100
        self.max_file_size = 100000  # 100KB limit
        
        # Chart type mappings
        self.chart_keywords = {
            'scatter': ['scatter', 'scatterplot'],
            'line': ['line', 'trend', 'time series'],
            'bar': ['bar', 'column', 'histogram'],
            'correlation': ['correlation', 'corr', 'heatmap'],
            'distribution': ['distribution', 'density', 'kde'],
            'box': ['box', 'boxplot', 'whisker'],
            'pie': ['pie', 'donut']
        }
    
    async def generate_visualization(self, question: str, datasets: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a visualization based on the question and return as Base64 data URI.
        
        Args:
            question: The question requesting a visualization
            datasets: Available datasets
            
        Returns:
            Base64 data URI string
        """
        try:
            logger.info(f"Generating visualization for: {question[:100]}...")
            
            # Analyze the question to determine chart type and data requirements
            chart_config = await self._analyze_visualization_request(question, datasets)
            
            if not chart_config:
                return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            # Generate the plot
            fig, ax = plt.subplots(figsize=self.default_figsize, dpi=self.default_dpi)
            
            chart_type = chart_config['type']
            
            if chart_type == 'scatter':
                await self._create_scatterplot(ax, chart_config)
            elif chart_type == 'line':
                await self._create_line_plot(ax, chart_config)
            elif chart_type == 'bar':
                await self._create_bar_plot(ax, chart_config)
            elif chart_type == 'correlation':
                await self._create_correlation_plot(ax, chart_config)
            elif chart_type == 'distribution':
                await self._create_distribution_plot(ax, chart_config)
            elif chart_type == 'box':
                await self._create_box_plot(ax, chart_config)
            else:
                # Default to scatter plot
                await self._create_scatterplot(ax, chart_config)
            
            # Set title and labels
            if 'title' in chart_config:
                ax.set_title(chart_config['title'], fontsize=14, fontweight='bold')
            
            if 'xlabel' in chart_config:
                ax.set_xlabel(chart_config['xlabel'], fontsize=12)
            
            if 'ylabel' in chart_config:
                ax.set_ylabel(chart_config['ylabel'], fontsize=12)
            
            plt.tight_layout()
            
            # Convert to base64
            return await self._convert_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            # Return a simple error image
            return await self._create_error_image(str(e))
        finally:
            plt.close('all')  # Clean up matplotlib figures
    
    async def _analyze_visualization_request(self, question: str, datasets: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Analyze the visualization request and determine chart configuration."""
        try:
            # Determine chart type from keywords
            chart_type = self._detect_chart_type(question)
            
            if not datasets:
                return None
            
            # Use the first dataset if multiple are available
            # In future iterations, this could be made smarter
            dataset_name, df = next(iter(datasets.items()))
            
            # Extract column names mentioned in the question
            columns_mentioned = self._extract_column_names(question, df.columns.tolist())
            
            config = {
                'type': chart_type,
                'dataset': df,
                'dataset_name': dataset_name,
                'columns_mentioned': columns_mentioned
            }
            
            # Configure based on chart type
            if chart_type == 'scatter':
                config.update(await self._configure_scatter_plot(question, df, columns_mentioned))
            elif chart_type == 'line':
                config.update(await self._configure_line_plot(question, df, columns_mentioned))
            elif chart_type == 'bar':
                config.update(await self._configure_bar_plot(question, df, columns_mentioned))
            elif chart_type == 'correlation':
                config.update(await self._configure_correlation_plot(question, df, columns_mentioned))
            
            return config
            
        except Exception as e:
            logger.error(f"Error analyzing visualization request: {str(e)}")
            return None
    
    def _detect_chart_type(self, question: str) -> str:
        """Detect the type of chart requested based on keywords."""
        question_lower = question.lower()
        
        for chart_type, keywords in self.chart_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return chart_type
        
        # Default to scatter plot
        return 'scatter'
    
    def _extract_column_names(self, question: str, columns: List[str]) -> List[str]:
        """Extract column names mentioned in the question."""
        mentioned_columns = []
        question_lower = question.lower()
        
        for col in columns:
            if col.lower() in question_lower:
                mentioned_columns.append(col)
        
        return mentioned_columns
    
    async def _configure_scatter_plot(self, question: str, df: pd.DataFrame, columns_mentioned: List[str]) -> Dict[str, Any]:
        """Configure scatter plot parameters."""
        config = {}
        
        # Try to identify X and Y columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns_mentioned) >= 2:
            # Use mentioned columns
            x_col = columns_mentioned[0]
            y_col = columns_mentioned[1]
        elif len(numeric_cols) >= 2:
            # Use first two numeric columns
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        else:
            # Fallback
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        config.update({
            'x_col': x_col,
            'y_col': y_col,
            'xlabel': x_col,
            'ylabel': y_col,
            'title': f'{y_col} vs {x_col}'
        })
        
        # Check if regression line is requested
        if 'regression' in question.lower() or 'trend' in question.lower():
            config['add_regression'] = True
            config['regression_style'] = 'dotted' if 'dotted' in question.lower() else 'solid'
            config['regression_color'] = 'red' if 'red' in question.lower() else 'blue'
        
        return config
    
    async def _configure_line_plot(self, question: str, df: pd.DataFrame, columns_mentioned: List[str]) -> Dict[str, Any]:
        """Configure line plot parameters."""
        config = {}
        
        # For line plots, typically X is time/date or sequential, Y is the value
        date_cols = df.select_dtypes(include=['datetime64', 'date']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if date_cols:
            x_col = date_cols[0]
            y_col = columns_mentioned[0] if columns_mentioned else (numeric_cols[0] if numeric_cols else df.columns[1])
        else:
            x_col = df.columns[0]
            y_col = columns_mentioned[0] if columns_mentioned else df.columns[1]
        
        config.update({
            'x_col': x_col,
            'y_col': y_col,
            'xlabel': x_col,
            'ylabel': y_col,
            'title': f'{y_col} over {x_col}'
        })
        
        return config
    
    async def _configure_bar_plot(self, question: str, df: pd.DataFrame, columns_mentioned: List[str]) -> Dict[str, Any]:
        """Configure bar plot parameters."""
        config = {}
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if categorical_cols and numeric_cols:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
        else:
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        config.update({
            'x_col': x_col,
            'y_col': y_col,
            'xlabel': x_col,
            'ylabel': y_col,
            'title': f'{y_col} by {x_col}'
        })
        
        return config
    
    async def _configure_correlation_plot(self, question: str, df: pd.DataFrame, columns_mentioned: List[str]) -> Dict[str, Any]:
        """Configure correlation plot parameters."""
        config = {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if columns_mentioned:
            # Use mentioned columns
            cols_to_correlate = [col for col in columns_mentioned if col in numeric_cols]
        else:
            # Use all numeric columns
            cols_to_correlate = numeric_cols[:10]  # Limit to first 10 to avoid clutter
        
        config.update({
            'columns': cols_to_correlate,
            'title': 'Correlation Matrix'
        })
        
        return config
    
    async def _create_scatterplot(self, ax, config: Dict[str, Any]):
        """Create a scatter plot."""
        df = config['dataset']
        x_col = config['x_col']
        y_col = config['y_col']
        
        # Create scatter plot
        ax.scatter(df[x_col], df[y_col], alpha=0.6)
        
        # Add regression line if requested
        if config.get('add_regression', False):
            x_numeric = pd.to_numeric(df[x_col], errors='coerce').dropna()
            y_numeric = pd.to_numeric(df[y_col], errors='coerce').dropna()
            
            if len(x_numeric) > 1 and len(y_numeric) > 1:
                # Align the data
                common_idx = x_numeric.index.intersection(y_numeric.index)
                x_vals = x_numeric.loc[common_idx]
                y_vals = y_numeric.loc[common_idx]
                
                if len(x_vals) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                    line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
                    line_y = slope * line_x + intercept
                    
                    style = '--' if config.get('regression_style') == 'dotted' else '-'
                    color = config.get('regression_color', 'red')
                    
                    ax.plot(line_x, line_y, style, color=color, linewidth=2, 
                           label=f'R² = {r_value**2:.3f}')
                    ax.legend()
    
    async def _create_line_plot(self, ax, config: Dict[str, Any]):
        """Create a line plot."""
        df = config['dataset']
        x_col = config['x_col']
        y_col = config['y_col']
        
        ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=4)
    
    async def _create_bar_plot(self, ax, config: Dict[str, Any]):
        """Create a bar plot."""
        df = config['dataset']
        x_col = config['x_col']
        y_col = config['y_col']
        
        # Group by categorical column and aggregate numeric column
        if df[x_col].dtype in ['object', 'category']:
            grouped = df.groupby(x_col)[y_col].mean().head(20)  # Limit to top 20 categories
            ax.bar(range(len(grouped)), grouped.values)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        else:
            ax.bar(df[x_col], df[y_col])
    
    async def _create_correlation_plot(self, ax, config: Dict[str, Any]):
        """Create a correlation heatmap."""
        df = config['dataset']
        columns = config['columns']
        
        if len(columns) > 1:
            corr_matrix = df[columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f', ax=ax)
        else:
            ax.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation', 
                   ha='center', va='center', transform=ax.transAxes)
    
    async def _create_distribution_plot(self, ax, config: Dict[str, Any]):
        """Create a distribution plot."""
        df = config['dataset']
        
        # Use the first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            ax.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'No numeric columns found for distribution plot', 
                   ha='center', va='center', transform=ax.transAxes)
    
    async def _create_box_plot(self, ax, config: Dict[str, Any]):
        """Create a box plot."""
        df = config['dataset']
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:5]  # Limit to 5 columns
        
        if numeric_cols:
            ax.boxplot([df[col].dropna() for col in numeric_cols], 
                      labels=numeric_cols)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No numeric columns found for box plot', 
                   ha='center', va='center', transform=ax.transAxes)
    
    async def _convert_to_base64(self, fig) -> str:
        """Convert matplotlib figure to Base64 data URI."""
        # Try different formats and compression levels to stay under 100KB
        formats_to_try = [
            ('png', {'format': 'png', 'dpi': 100, 'bbox_inches': 'tight', 'pad_inches': 0.1}),
            ('png', {'format': 'png', 'dpi': 80, 'bbox_inches': 'tight', 'pad_inches': 0.1}),
            ('png', {'format': 'png', 'dpi': 60, 'bbox_inches': 'tight', 'pad_inches': 0.1}),
            ('png', {'format': 'png', 'dpi': 50, 'bbox_inches': 'tight', 'pad_inches': 0.1}),
        ]
        
        for format_name, kwargs in formats_to_try:
            try:
                buffer = io.BytesIO()
                fig.savefig(buffer, **kwargs)
                buffer.seek(0)
                
                # Read the buffer content
                image_bytes = buffer.read()
                
                # Check file size
                if len(image_bytes) <= self.max_file_size and len(image_bytes) > 0:
                    # Encode to base64
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Validate the base64 string
                    if self._is_valid_base64(image_base64):
                        data_uri = f"data:image/{format_name};base64,{image_base64}"
                        logger.info(f"Successfully created {format_name} image: {len(image_bytes)} bytes")
                        return data_uri
                
                buffer.close()
                
            except Exception as e:
                logger.warning(f"Failed to save as {format_name}: {str(e)}")
                continue
        
        # If all formats fail, return a minimal valid image
        logger.warning("All image formats failed, returning minimal image")
        return await self._create_minimal_image()
    
    def _is_valid_base64(self, s: str) -> bool:
        """Validate if string is valid base64."""
        try:
            if len(s) < 10:  # Too short to be a valid image
                return False
            
            # Try to decode and re-encode
            decoded = base64.b64decode(s, validate=True)
            reencoded = base64.b64encode(decoded).decode('utf-8')
            
            # Check if it starts with PNG header (common case)
            if decoded.startswith(b'\x89PNG'):
                return True
                
            # For other formats, just check if decode/encode works
            return len(decoded) > 50  # Reasonable minimum size for an image
            
        except Exception:
            return False
    
    async def _create_error_image(self, error_message: str) -> str:
        """Create a simple error image."""
        try:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
            ax.text(0.5, 0.5, f'Visualization Error', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='red', weight='bold')
            ax.text(0.5, 0.3, 'Unable to generate chart', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, color='black')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Use the same conversion method to ensure consistency
            return await self._convert_to_base64(fig)
        except Exception as e:
            logger.error(f"Failed to create error image: {str(e)}")
            return await self._create_minimal_image()
        finally:
            plt.close('all')
    
    async def _create_minimal_image(self) -> str:
        """Create a minimal 1x1 transparent PNG image with valid base64."""
        try:
            # Create a minimal 10x10 transparent PNG programmatically
            fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.patch.set_alpha(0)  # Transparent background
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=50, bbox_inches='tight', 
                       pad_inches=0, transparent=True)
            buffer.seek(0)
            
            image_bytes = buffer.read()
            if len(image_bytes) > 0:
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                if self._is_valid_base64(image_base64):
                    return f"data:image/png;base64,{image_base64}"
            
            buffer.close()
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to create minimal image: {str(e)}")
        
        # Final fallback: use a known valid minimal PNG
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        return f"data:image/png;base64,{minimal_png}" 