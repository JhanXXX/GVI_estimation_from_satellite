"""
Separated GVI Calculator with English Interface
(1) Independent GVI calculation methods (semantic or pixel)
(2) Independent database comparison functionality with customizable output paths
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_project_logger


class GVICalculator:
    """
    Separated GVI Calculator
    Feature 1: Independent GVI calculation (semantic or pixel)
    Feature 2: Independent database comparison analysis
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize GVI Calculator"""
        self.config = ConfigLoader(config_path)
        self.logger = get_project_logger(__name__)
        
        # GSV configuration
        self.gsv_config = self.config.get_gsv_config()
        self.expected_headings = self.gsv_config.headings
        
        # Lazy loading for models (load only when needed)
        self._semantic_model = None
        self._transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pixel method configuration
        self.pixel_config = self._load_pixel_config()
        
        # Initialize database
        self._setup_database()
        self.config.create_directories()
        
        self.logger.info(f"GVI Calculator initialized - Available methods: semantic, pixel")
    
    def _load_pixel_config(self) -> Dict:
        """Load pixel method configuration"""
        gvi_config = self.config.get('gvi_calculation', {})
        
        default_config = {
            'green_ranges': [
                {'lower': [35, 40, 40], 'upper': [85, 255, 255]},   # Primary green range
                {'lower': [25, 40, 40], 'upper': [35, 255, 255]}    # Yellow-green range
            ],
            'morphology': {
                'enabled': True,
                'kernel_size': 3,
                'opening_iterations': 1,
                'closing_iterations': 2
            },
            'min_area_filter': {
                'enabled': True,
                'min_area_pixels': 100
            }
        }
        
        pixel_config = gvi_config.get('pixel_method', default_config)
        
        # Backward compatibility with old color_threshold config
        if 'color_threshold' in gvi_config and 'green_ranges' not in pixel_config:
            old_config = gvi_config['color_threshold']
            pixel_config['green_ranges'] = [{
                'lower': old_config.get('lower_green', [35, 40, 40]),
                'upper': old_config.get('upper_green', [85, 255, 255])
            }]
        
        return pixel_config
    
    def _load_semantic_model(self):
        """Lazy load semantic model"""
        if self._semantic_model is None:
            self.logger.info("Loading DeepLab model for semantic method...")
            self._semantic_model = deeplabv3_resnet50(weights='DEFAULT')
            self._semantic_model.eval()
            self._semantic_model.to(self.device)
            
            self._transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self.logger.info(f"DeepLab model loaded on {self.device}")
    
    def _setup_database(self):
        """Setup database tables"""
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS directional_gvi (
                    panoid TEXT,
                    city TEXT,
                    country TEXT,
                    method TEXT,
                    total_directions INTEGER,
                    gvi_average REAL,
                    gvi_max REAL,
                    gvi_min REAL,
                    gvi_std REAL,
                    directions_processed TEXT,
                    config_headings TEXT,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (panoid, city, country, method)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS directional_gvi_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    panoid TEXT,
                    city TEXT,
                    country TEXT,
                    heading TEXT,
                    gvi_value REAL,
                    vegetation_pixels INTEGER,
                    total_pixels INTEGER,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (panoid, city, country) REFERENCES directional_gvi (panoid, city, country)
                )
            """)
    
    # ================== Feature 1: Independent GVI Calculation Methods ==================
    
    def calculate_gvi_pixel_method(self, image_path: str) -> Dict[str, float]:
        """
        Calculate GVI using pixel-based method
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing GVI value and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create combined mask
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for green_range in self.pixel_config['green_ranges']:
            lower_green = np.array(green_range['lower'], dtype=np.uint8)
            upper_green = np.array(green_range['upper'], dtype=np.uint8)
            
            range_mask = cv2.inRange(hsv, lower_green, upper_green)
            combined_mask = cv2.bitwise_or(combined_mask, range_mask)
        
        # Apply morphological operations
        if self.pixel_config['morphology']['enabled']:
            kernel_size = self.pixel_config['morphology']['kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Opening (noise removal)
            opening_iters = self.pixel_config['morphology']['opening_iterations']
            if opening_iters > 0:
                combined_mask = cv2.morphologyEx(
                    combined_mask, cv2.MORPH_OPEN, kernel, iterations=opening_iters
                )
            
            # Closing (hole filling)
            closing_iters = self.pixel_config['morphology']['closing_iterations']
            if closing_iters > 0:
                combined_mask = cv2.morphologyEx(
                    combined_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iters
                )
        
        # Apply minimum area filtering
        if self.pixel_config['min_area_filter']['enabled']:
            min_area = self.pixel_config['min_area_filter']['min_area_pixels']
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            filtered_mask = np.zeros_like(combined_mask)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.fillPoly(filtered_mask, [contour], 255)
            
            combined_mask = filtered_mask
        
        # Calculate GVI
        vegetation_pixels = np.sum(combined_mask > 0)
        total_pixels = combined_mask.size
        gvi_value = vegetation_pixels / total_pixels
        
        return {
            "gvi_value": float(gvi_value),
            "method": "pixel_based",
            "vegetation_pixels": int(vegetation_pixels),
            "total_pixels": int(total_pixels),
            "image_size": original_size
        }
    
    def calculate_gvi_semantic_method(self, image_path: str) -> Dict[str, float]:
        """
        Calculate GVI using semantic segmentation method
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing GVI value and metadata
        """
        # Lazy load model
        self._load_semantic_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform for model input
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)
        
        # Get segmentation results
        with torch.no_grad():
            output = self._semantic_model(input_tensor)['out']
            segmentation = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        # Resize segmentation to original size
        segmentation_resized = cv2.resize(
            segmentation.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create vegetation mask (PASCAL VOC class 15 for vegetation)
        vegetation_mask = (segmentation_resized == 15)
        vegetation_pixels = np.sum(vegetation_mask)
        total_pixels = vegetation_mask.size
        
        gvi_value = vegetation_pixels / total_pixels
        
        return {
            "gvi_value": float(gvi_value),
            "method": "semantic_deeplab",
            "vegetation_pixels": int(vegetation_pixels),
            "total_pixels": int(total_pixels),
            "image_size": original_size
        }
    
    def process_city_single_method(self, country: str, city: str, method: str,
                                  max_panoids: Optional[int] = None,
                                  skip_existing: bool = True) -> Dict[str, int]:
        """
        Process all panoids in a city using a single method
        
        Args:
            country: Country name
            city: City name
            method: Method name ('semantic' or 'pixel')
            max_panoids: Maximum number to process (for testing)
            skip_existing: Whether to skip existing results
            
        Returns:
            Processing statistics
        """
        if method not in ['semantic', 'pixel']:
            raise ValueError(f"Method must be 'semantic' or 'pixel', got '{method}'")
        
        self.logger.info(f"Processing {country}/{city} with method: {method}")
        
        # Discover all panoids
        all_panoids = self._discover_panoids(country, city)
        if not all_panoids:
            self.logger.warning(f"No panoids found for {country}/{city}")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        
        # Check existing results
        existing_panoids = set()
        if skip_existing:
            existing_panoids = self._get_existing_panoids(country, city, method)
            self.logger.info(f"Found {len(existing_panoids)} existing results for method {method}")
        
        # Determine panoids to process
        panoids_to_process = [p for p in all_panoids if p not in existing_panoids]
        
        if max_panoids and len(panoids_to_process) > max_panoids:
            panoids_to_process = panoids_to_process[:max_panoids]
        
        # Statistics
        stats = {
            "total": len(all_panoids),
            "processed": 0,
            "skipped": len(existing_panoids),
            "failed": 0
        }
        
        self.logger.info(f"Will process {len(panoids_to_process)} panoids with method {method}")
        
        # Process each panoid
        for idx, panoid in enumerate(panoids_to_process):
            try:
                result = self._process_single_panoid(panoid, country, city, method)
                if result:
                    self._save_gvi_result(result, country, city)
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                
                # Progress reporting
                if (idx + 1) % self.gsv_config.progress_interval == 0:
                    self.logger.info(f"Progress: {idx + 1}/{len(panoids_to_process)} processed")
                    
            except Exception as e:
                self.logger.error(f"Error processing panoid {panoid}: {e}")
                stats["failed"] += 1
        
        self.logger.info(f"Processing complete: {stats}")
        return stats
    
    def _process_single_panoid(self, panoid: str, country: str, city: str, method: str) -> Optional[Dict]:
        """Process a single panoid"""
        # Get image files
        image_files = self._get_panoid_images(panoid, country, city)
        if not image_files:
            return None
        
        # Calculate GVI for each direction
        gvi_values = []
        detailed_results = []
        
        for heading, image_path in image_files.items():
            try:
                if method == 'pixel':
                    gvi_result = self.calculate_gvi_pixel_method(image_path)
                elif method == 'semantic':
                    gvi_result = self.calculate_gvi_semantic_method(image_path)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                gvi_values.append(gvi_result["gvi_value"])
                detailed_results.append({
                    "heading": heading,
                    "gvi_value": gvi_result["gvi_value"],
                    "vegetation_pixels": gvi_result["vegetation_pixels"],
                    "total_pixels": gvi_result["total_pixels"]
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {panoid} heading {heading}: {e}")
                continue
        
        if not gvi_values:
            return None
        
        # Aggregate results
        return {
            "panoid": panoid,
            "country": country,
            "city": city,
            "method": method,
            "total_directions": len(gvi_values),
            "gvi_average": float(np.mean(gvi_values)),
            "gvi_max": float(np.max(gvi_values)),
            "gvi_min": float(np.min(gvi_values)),
            "gvi_std": float(np.std(gvi_values)),
            "directions_processed": list(image_files.keys()),
            "detailed_results": detailed_results
        }
    
    # ================== Feature 2: Independent Database Comparison ==================
    
    def compare_methods_from_database(self, country: str, city: str,
                                    methods: List[str] = None,
                                    output_dir: str = "./comparison_results",
                                    output_prefix: str = "method_comparison") -> Dict:
        """
        Compare different GVI methods from database results
        
        Args:
            country: Country name
            city: City name
            methods: List of methods to compare (default: ["semantic", "pixel"])
            output_dir: Output directory path
            output_prefix: Output file prefix
            
        Returns:
            Comparison results dictionary
        """
        if methods is None:
            methods = ["semantic", "pixel"]
        
        # Validate methods
        valid_methods = ["semantic", "pixel", "color"]
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        self.logger.info(f"Comparing methods {methods} for {country}/{city}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data from database
        comparison_data = self._load_comparison_data_from_db(country, city, methods)
        
        if comparison_data.empty:
            raise ValueError(f"No comparable data found for methods {methods}")
        
        # Perform statistical analysis
        stats_results = self._calculate_comparison_statistics(comparison_data, methods)
        
        # Generate visualization report
        report_path = self._generate_comparison_report(
            comparison_data, stats_results, methods, 
            output_dir, output_prefix, country, city
        )
        
        # Save data as CSV
        csv_path = output_dir / f"{output_prefix}_{country}_{city}_data.csv"
        self._save_comparison_csv(comparison_data, csv_path)
        
        self.logger.info(f"Comparison complete. Report saved to: {report_path}")
        self.logger.info(f"Data saved to: {csv_path}")
        
        return {
            "statistics": stats_results,
            "report_path": str(report_path),
            "data_path": str(csv_path),
            "sample_size": len(comparison_data['panoid'].unique()),
            "methods_compared": methods
        }
    
    def _load_comparison_data_from_db(self, country: str, city: str, methods: List[str]) -> pd.DataFrame:
        """Load comparison data from database"""
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        with sqlite3.connect(db_path) as conn:
            # Query data for all methods
            placeholders = ','.join(['?' for _ in methods])
            query = f"""
                SELECT panoid, method, gvi_average, gvi_max, gvi_min, gvi_std, total_directions
                FROM directional_gvi 
                WHERE country = ? AND city = ? AND method IN ({placeholders})
            """
            
            params = [country, city] + methods
            df = pd.read_sql_query(query, conn, params=params)
        
        # Filter panoids that have data for all methods
        panoid_counts = df.groupby('panoid')['method'].count()
        complete_panoids = panoid_counts[panoid_counts == len(methods)].index
        
        filtered_df = df[df['panoid'].isin(complete_panoids)]
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(complete_panoids)} panoids with data for all methods")
        return filtered_df
    
    def _calculate_comparison_statistics(self, data: pd.DataFrame, methods: List[str]) -> Dict:
        """Calculate comparison statistics"""
        if data.empty:
            return {}
        
        stats = {}
        
        # Calculate basic statistics for each method
        for method in methods:
            method_data = data[data['method'] == method]['gvi_average']
            stats[method] = {
                "count": len(method_data),
                "mean": float(method_data.mean()),
                "std": float(method_data.std()),
                "min": float(method_data.min()),
                "max": float(method_data.max()),
                "median": float(method_data.median())
            }
        
        # Pairwise comparisons
        pivot_data = data.pivot(index='panoid', columns='method', values='gvi_average')
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if method1 in pivot_data.columns and method2 in pivot_data.columns:
                    values1 = pivot_data[method1].dropna()
                    values2 = pivot_data[method2].dropna()
                    
                    # Find common panoids
                    common_panoids = values1.index.intersection(values2.index)
                    if len(common_panoids) > 0:
                        v1 = values1.loc[common_panoids]
                        v2 = values2.loc[common_panoids]
                        
                        correlation = np.corrcoef(v1, v2)[0, 1]
                        differences = v1 - v2
                        
                        stats[f"{method1}_vs_{method2}"] = {
                            "correlation": float(correlation),
                            "mean_difference": float(differences.mean()),
                            "std_difference": float(differences.std()),
                            "max_abs_difference": float(np.abs(differences).max()),
                            "sample_size": len(common_panoids)
                        }
        
        return stats
    
    def _generate_comparison_report(self, data: pd.DataFrame, stats: Dict, methods: List[str],
                                  output_dir: Path, prefix: str, country: str, city: str) -> Path:
        """Generate comparison report"""
        
        report_path = output_dir / f"{prefix}_{country}_{city}_report.pdf"
        
        with PdfPages(report_path) as pdf:
            plt.style.use('default')
            
            # Pivot data
            pivot_data = data.pivot(index='panoid', columns='method', values='gvi_average')
            
            # 1. Scatter plot comparison
            if len(methods) == 2:
                self._plot_scatter_comparison(pivot_data, methods, stats, pdf, country, city)
            
            # 2. Distribution comparison
            self._plot_distribution_comparison(pivot_data, methods, pdf, country, city)
            
            # 3. Statistics summary
            self._plot_statistics_summary(stats, methods, pdf, country, city)
            
            # 4. Difference analysis
            if len(methods) == 2:
                self._plot_difference_analysis(pivot_data, methods, stats, pdf, country, city)
        
        return report_path
    
    def _plot_scatter_comparison(self, data: pd.DataFrame, methods: List[str], 
                               stats: Dict, pdf: PdfPages, country: str, city: str):
        """Plot scatter comparison"""
        method1, method2 = methods[0], methods[1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get common data
        common_data = data[[method1, method2]].dropna()
        
        if len(common_data) > 0:
            x, y = common_data[method1], common_data[method2]
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=30)
            
            # 1:1 line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Labels and title
            ax.set_xlabel(f'{method1.title()} GVI', fontsize=12)
            ax.set_ylabel(f'{method2.title()} GVI', fontsize=12)
            
            # Add statistics
            comparison_key = f"{method1}_vs_{method2}"
            if comparison_key in stats:
                correlation = stats[comparison_key]["correlation"]
                ax.set_title(f'GVI Comparison: {method1.title()} vs {method2.title()}\n'
                           f'{country} - {city}\n'
                           f'Correlation: {correlation:.3f}, N = {len(common_data)}', 
                           fontsize=14)
            
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_comparison(self, data: pd.DataFrame, methods: List[str], 
                                    pdf: PdfPages, country: str, city: str):
        """Plot distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram comparison
        for method in methods:
            if method in data.columns:
                method_data = data[method].dropna()
                ax1.hist(method_data, bins=20, alpha=0.7, label=method.title(), density=True)
        
        ax1.set_xlabel('GVI Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'GVI Distribution Comparison\n{country} - {city}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        box_data = [data[method].dropna() for method in methods if method in data.columns]
        box_labels = [method.title() for method in methods if method in data.columns]
        
        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_ylabel('GVI Value', fontsize=12)
        ax2.set_title(f'GVI Distribution (Box Plot)\n{country} - {city}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_statistics_summary(self, stats: Dict, methods: List[str], 
                               pdf: PdfPages, country: str, city: str):
        """Plot statistics summary"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create summary text
        summary_text = f"GVI Method Comparison Summary\n"
        summary_text += f"Location: {country} - {city}\n"
        summary_text += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Method statistics
        for method in methods:
            if method in stats:
                method_stats = stats[method]
                summary_text += f"{method.upper()} Method Statistics:\n"
                summary_text += f"  Sample Size: {method_stats['count']}\n"
                summary_text += f"  Mean GVI: {method_stats['mean']:.4f}\n"
                summary_text += f"  Std Dev: {method_stats['std']:.4f}\n"
                summary_text += f"  Range: {method_stats['min']:.4f} - {method_stats['max']:.4f}\n"
                summary_text += f"  Median: {method_stats['median']:.4f}\n\n"
        
        # Comparison statistics
        for key, value in stats.items():
            if "_vs_" in key:
                summary_text += f"Comparison - {key.replace('_', ' ').title()}:\n"
                summary_text += f"  Correlation: {value['correlation']:.4f}\n"
                summary_text += f"  Mean Difference: {value['mean_difference']:.4f}\n"
                summary_text += f"  Std Difference: {value['std_difference']:.4f}\n"
                summary_text += f"  Max |Difference|: {value['max_abs_difference']:.4f}\n"
                summary_text += f"  Sample Size: {value['sample_size']}\n\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_difference_analysis(self, data: pd.DataFrame, methods: List[str],
                                stats: Dict, pdf: PdfPages, country: str, city: str):
        """Plot difference analysis"""
        if len(methods) != 2:
            return
        
        method1, method2 = methods[0], methods[1]
        common_data = data[[method1, method2]].dropna()
        
        if len(common_data) == 0:
            return
        
        differences = common_data[method1] - common_data[method2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residual plot
        predicted = common_data[method1]
        ax1.scatter(predicted, differences, alpha=0.6, s=30)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        ax1.set_xlabel(f'{method1.title()} GVI', fontsize=12)
        ax1.set_ylabel(f'Difference ({method1} - {method2})', fontsize=12)
        ax1.set_title(f'Residual Plot\n{country} - {city}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Difference distribution
        ax2.hist(differences, bins=20, alpha=0.7, color='orange')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel(f'Difference ({method1} - {method2})', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Difference Distribution\nMean: {differences.mean():.4f}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _save_comparison_csv(self, data: pd.DataFrame, csv_path: Path):
        """Save comparison data as CSV"""
        # Reshape data to wide format
        pivot_data = data.pivot(index='panoid', columns='method', values='gvi_average')
        pivot_data.reset_index().to_csv(csv_path, index=False)
    
    # ================== Helper Functions ==================
    
    def _discover_panoids(self, country: str, city: str) -> List[str]:
        """Discover panoid folders"""
        preview_dir = (self.config.get_data_dir() / "panorama" / "previews" / 
                      country / city)
        
        if not preview_dir.exists():
            return []
        
        panoids = []
        for item in preview_dir.iterdir():
            if item.is_dir():
                panoids.append(item.name)
        
        return sorted(panoids)
    
    def _get_panoid_images(self, panoid: str, country: str, city: str) -> Dict[str, str]:
        """Get all directional images for a panoid"""
        pano_dir = (self.config.get_data_dir() / "panorama" / "previews" / 
                   country / city / panoid)
        
        if not pano_dir.exists():
            return {}
        
        image_files = {}
        for image_file in pano_dir.glob("*.jpg"):
            heading = image_file.stem
            image_files[heading] = str(image_file)
        
        return image_files
    
    def _get_existing_panoids(self, country: str, city: str, method: str) -> set:
        """Get existing panoids from database"""
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        
        if not db_path.exists():
            return set()
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT panoid FROM directional_gvi 
                    WHERE country = ? AND city = ? AND method = ?
                """, (country, city, method))
                
                return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()
    
    def _save_gvi_result(self, gvi_result: Dict, country: str, city: str):
        """Save GVI result to database"""
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        
        with sqlite3.connect(db_path) as conn:
            # Insert main result
            conn.execute("""
                INSERT OR REPLACE INTO directional_gvi 
                (panoid, city, country, method, total_directions, gvi_average, gvi_max, gvi_min, gvi_std, 
                 directions_processed, config_headings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gvi_result['panoid'],
                city,
                country,
                gvi_result['method'],
                gvi_result['total_directions'],
                gvi_result['gvi_average'],
                gvi_result['gvi_max'],
                gvi_result['gvi_min'],
                gvi_result['gvi_std'],
                json.dumps(gvi_result['directions_processed']),
                json.dumps(self.expected_headings)
            ))
            
            # Delete existing detailed results
            conn.execute("""
                DELETE FROM directional_gvi_details 
                WHERE panoid = ? AND city = ? AND country = ?
            """, (gvi_result['panoid'], city, country))
            
            # Insert detailed results
            for detail in gvi_result['detailed_results']:
                conn.execute("""
                    INSERT INTO directional_gvi_details 
                    (panoid, city, country, heading, gvi_value, vegetation_pixels, total_pixels)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    gvi_result['panoid'],
                    city,
                    country,
                    detail['heading'],
                    detail['gvi_value'],
                    detail['vegetation_pixels'],
                    detail['total_pixels']
                ))


# ================== Usage Examples and Test Functions ==================

def process_multiple_cities(cities, method):
    """For processing multiple cities"""
    calculator = GVICalculator()
    
    print("=== Batch processing multiple cities with pixel method ===")
    
    for country, city in cities:
        print(f"\nProcessing {country}/{city}...")
        try:
            stats = calculator.process_city_single_method(
                country=country,
                city=city,
                method=method,
                skip_existing=True
            )
            print(f"  Completed: processed {stats['processed']}, skipped {stats['skipped']}, failed {stats['failed']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\nBatch processing completed!")


def compare_multiple_cities(cities, methods):
    """Compare method differences for multiple cities"""
    calculator = GVICalculator()
    
    """
    cities = [
        ("Finland", "Helsinki"),
        ("Sweden", "Stockholm"),
        ("Denmark", "Copenhagen"),
    ]"""

    """
    methods = ['semantic', 'pixel']
    """
    
    print("=== Batch comparison for multiple cities ===")
    
    for country, city in cities:
        print(f"\nComparing {country}/{city}...")
        
        try:
            results = calculator.compare_methods_from_database(
                country=country,
                city=city,
                methods=methods,
                output_dir=f"./comparison_results/{country}_{city}",
                output_prefix=f"{country}_{city}_comparison"
            )
            
            print(f"  Completed: {results['sample_size']} samples")
            print(f"  Report: {results['report_path']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue



def test_pixel_method():
    """Test pixel method"""
    try:
        calculator = GVICalculator()
        
        print("GVI Calculator initialized successfully")
        print("Available methods: semantic, pixel")
        
        # Test pixel configuration loading
        pixel_config = calculator.pixel_config
        print(f"Pixel configuration loaded: {len(pixel_config['green_ranges'])} green ranges")
        print(f"Morphology enabled: {pixel_config['morphology']['enabled']}")
        print(f"Area filtering enabled: {pixel_config['min_area_filter']['enabled']}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def test_single_image_pixel():
    """Test pixel method on a single image"""
    calculator = GVICalculator()
    
    # This requires an actual image path for testing
    # test_image_path = "./data/panorama/previews/Finland/Helsinki/some_panoid/0.jpg"
    
    # Example (requires actual image path):
    # try:
    #     result = calculator.calculate_gvi_pixel_method(test_image_path)
    #     print(f"Pixel GVI calculation successful: {result['gvi_value']:.4f}")
    #     return True
    # except Exception as e:
    #     print(f"Pixel GVI calculation failed: {e}")
    #     return False
    
    print("Need actual image path for testing")
    return True

"""
English Usage Examples for Separated GVI Calculator
Demonstrates how to use GVI calculation and database comparison features separately
"""


def process_multiple_cities(cities, method):
    """Example for processing multiple cities"""
    calculator = GVICalculator()
    
    """# Define list of cities to process
    cities = [
        ("Finland", "Helsinki"),
        ("Sweden", "Stockholm"),
        ("Denmark", "Copenhagen"),
        # Add more cities...
    ]"""
    
    print("=== Batch processing multiple cities with pixel method ===")
    
    for country, city in cities:
        print(f"\nProcessing {country}/{city}...")
        
        try:
            stats = calculator.process_city_single_method(
                country=country,
                city=city,
                method=method,
                skip_existing=True
            )
            print(f"  Completed: processed {stats['processed']}, skipped {stats['skipped']}, failed {stats['failed']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\nBatch processing completed!")


def compare_multiple_cities(cities, methods):
    """Compare method differences across multiple cities"""
    calculator = GVICalculator()
    
    print("=== Batch comparison for multiple cities ===")
    
    for country, city in cities:
        print(f"\nComparing {country}/{city}...")
        
        try:
            results = calculator.compare_methods_from_database(
                country=country,
                city=city,
                methods=methods,
                output_dir=f"./comparison_results/{country}_{city}",
                output_prefix=f"{country}_{city}_comparison"
            )
            
            print(f"  Completed: {results['sample_size']} samples")
            print(f"  Report: {results['report_path']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue


def test_single_panoid(method):
    """Test processing of a single panoid"""
    calculator = GVICalculator()
    
    # Replace with your actual data
    country = "Finland"
    city = "Helsinki"
    test_panoid = "your_test_panoid"  # Replace with actual panoid
    
    print(f"=== Test single panoid: {test_panoid} ===")
    
    # Test pixel method
    try:
        result = calculator._process_single_panoid(test_panoid, country, city, method)
        if result:
            print(f"{method} method successful:")
            print(f"Average GVI: {result['gvi_average']:.4f}")
            print(f"Directions processed: {result['total_directions']}")
            print(f"GVI range: {result['gvi_min']:.4f} - {result['gvi_max']:.4f}")
        else:
            print("Pixel method failed")
            
    except Exception as e:
        print(f"Test failed: {e}")


def advanced_comparison_analysis():
    """Advanced comparison analysis with custom settings"""
    calculator = GVICalculator()
    
    print("=== Advanced Comparison Analysis ===")
    
    # Example: Compare methods with custom output structure
    analysis_configs = [
        {
            "country": "Finland",
            "city": "Helsinki", 
            "output_dir": "./results/finland",
            "prefix": "helsinki_detailed_analysis"
        },
        {
            "country": "Sweden", 
            "city": "Stockholm",
            "output_dir": "./results/sweden", 
            "prefix": "stockholm_detailed_analysis"
        }
    ]
    
    for config in analysis_configs:
        print(f"\nAnalyzing {config['country']}/{config['city']}...")
        
        try:
            results = calculator.compare_methods_from_database(
                country=config['country'],
                city=config['city'],
                methods=["semantic", "pixel"],
                output_dir=config['output_dir'],
                output_prefix=config['prefix']
            )
            
            # Extract key metrics
            stats = results['statistics']
            print(f"  Sample size: {results['sample_size']}")
            
            if 'semantic_vs_pixel' in stats:
                comp_stats = stats['semantic_vs_pixel']
                print(f"  Correlation: {comp_stats['correlation']:.3f}")
                print(f"  Mean difference: {comp_stats['mean_difference']:.4f}")
                print(f"  Std difference: {comp_stats['std_difference']:.4f}")
            
            # Method-specific statistics
            for method in ["semantic", "pixel"]:
                if method in stats:
                    method_stats = stats[method]
                    print(f"  {method.title()} - Mean: {method_stats['mean']:.4f}, "
                          f"Std: {method_stats['std']:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")


def quality_control_analysis(cities_to_check):
    """Quality control analysis for processed data"""
    calculator = GVICalculator()
    
    print("=== Quality Control Analysis ===")
    
    # Check processing status for different cities

    for country, city in cities_to_check:
        print(f"\nChecking {country}/{city}:")
        
        # Check how many panoids have been processed for each method
        for method in ["semantic", "pixel"]:
            try:
                existing_panoids = calculator._get_existing_panoids(country, city, method)
                all_panoids = calculator._discover_panoids(country, city)
                
                if all_panoids:
                    completion_rate = len(existing_panoids) / len(all_panoids) * 100
                    print(f"  {method.title()}: {len(existing_panoids)}/{len(all_panoids)} "
                          f"({completion_rate:.1f}% complete)")
                else:
                    print(f"  {method.title()}: No panoids found in directory")
                    
            except Exception as e:
                print(f"  {method.title()}: Error checking - {e}")


if __name__ == "__main__":

    # Choose which functions to run:

    cities = [
        ("Greece","Athen"),
        ("France","Paris"),
        ("Netherlands","Amsterdam"),
        ("Hungary","Budapest")
    ]
    method = "pixel" # or "semantic" for DeepLab (much slower)
    
    # Batch process multiple cities
    process_multiple_cities(cities, method)
    
    # Batch compare multiple cities (optional) 
    # compare_multiple_cities(cities, method)
    
    # Test single panoid (for debugging)
    # test_single_panoid(method)
    
    # Advanced comparison analysis (optional)
    # advanced_comparison_analysis(cities)
    
    # Quality control check (optional)
    # quality_control_analysis(cities_to_check)