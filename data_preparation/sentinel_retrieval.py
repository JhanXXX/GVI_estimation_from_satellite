"""
Direct Sentinel-2A Data Retrieval using rasterio (bypassing stackstac)
More robust and reliable approach
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Point
import pystac_client
import planetary_computer as pc
import rasterio
import rioxarray
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.windows import from_bounds
import tempfile
import requests
from urllib.parse import urlparse

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_project_logger
from src.utils.spatial_utils import SpatialUtils
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class SentinelDataRetriever:
    """
    Retrieve and process Sentinel-2A data using direct rasterio access
    Bypasses stackstac for better reliability
    """

    """
    2.0 update: Enhanced Sentinel-2A data retrieval with clipping and raw band support
    """

    def __init__(self, config_path: str = "config.json"):
        """Initialize with configuration"""
        self.config = ConfigLoader(config_path)
        self.logger = get_project_logger(__name__)
        self.temporal_config = self.config.get_temporal_config()
        self.year_range = self.temporal_config.year_range
        
        # STAC API endpoint
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        
        # Load configurations
        rs_config = self.config.get_rs_data_config()
        self.base_resolution = rs_config.resolution
        self.ground_features = rs_config.ground_features
        
        # Raw band configuration
        self.raw_band_names = self.config.get('raw_band_names', ['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12'])
        
        # Band resolution mapping
        self.band_resolutions = {
            'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,  # 10m bands
            'B05': 20, 'B06': 20, 'B07': 20, 'B11': 20, 'B12': 20,  # 20m bands
            'B01': 60, 'B09': 60, 'B10': 60  # 60m bands (filtered out)
        }
        
        # Band mapping for Sentinel-2
        self.band_mapping = {
            'B02': 'blue',     # 490nm
            'B03': 'green',    # 560nm  
            'B04': 'red',      # 665nm
            'B05': 'red_edge', # 705nm
            'B08': 'nir',      # 842nm
            'B11': 'swir1',    # 1610nm
            'B12': 'swir2'     # 2190nm
        }
        
        # Create necessary directories
        self.config.create_directories()
        self.logger.info(f"Sentinel retriever initialized with {len(self.raw_band_names)} raw bands")

    def calculate_buffer_pixels(self, buffer_size: int) -> int:
        """
        Calculate pixel dimensions for buffer size
        Reusable function for consistent sizing logic
        
        Args:
            buffer_size: Buffer size in meters
            
        Returns:
            Pixels per side for square grid
        """
        pixels_per_side = int(2 * buffer_size / self.base_resolution)
        return pixels_per_side
    
    def clip_buffer_data(self, source_buffer: int, target_buffer: int, cities: List[Tuple[str, str]], method) -> Dict[str, int]:
        """
        Clip existing buffer data to smaller buffer size
        
        Args:
            source_buffer: Source buffer size (e.g., 600m)
            target_buffer: Target buffer size (e.g., 150m) 
            cities: List of (country, city) tuples
            
        Returns:
            Processing statistics
        """
        if target_buffer >= source_buffer:
            raise ValueError(f"Target buffer ({target_buffer}m) must be smaller than source buffer ({source_buffer}m)")
        
        self.logger.info(f"Starting buffer clipping: {source_buffer}m -> {target_buffer}m")
        
        # Calculate clipping parameters
        source_pixels = self.calculate_buffer_pixels(source_buffer)
        target_pixels = self.calculate_buffer_pixels(target_buffer)
        crop_pixels = (source_pixels - target_pixels) // 2
        
        self.logger.info(f"Clipping parameters: {source_pixels}x{source_pixels} -> {target_pixels}x{target_pixels}, crop {crop_pixels} pixels")
        
        total_stats = {"total_files": 0, "successful": 0, "failed": 0, "skipped": 0}
        
        for country, city in cities:
            self.logger.info(f"Processing {country}/{city}")
            
            city_stats = self._clip_city_buffer_data(
                country, city, source_buffer, target_buffer, 
                target_pixels, crop_pixels, method
            )
            
            # Aggregate statistics
            for key in total_stats:
                total_stats[key] += city_stats.get(key, 0)
            
            self.logger.info(f"{country}/{city}: {city_stats['successful']}/{city_stats['total_files']} successful")
        
        self.logger.info(f"Buffer clipping complete: {total_stats['successful']}/{total_stats['total_files']} files processed")
        
        return total_stats

    def _clip_city_buffer_data(self, country: str, city: str, source_buffer: int, target_buffer: int,
                              target_pixels: int, crop_pixels: int, method) -> Dict[str, int]:
        """Clip buffer data for a single city"""
        
        # Source and target directories
        source_dir = self.config.get_data_dir() / "base_maps" / country / city / f"{method}_{source_buffer}m"
        target_dir = self.config.get_data_dir() / "base_maps" / country / city / f"{method}_{target_buffer}m"
        
        if not source_dir.exists():
            self.logger.warning(f"Source directory not found: {source_dir}")
            return {"total_files": 0, "successful": 0, "failed": 0, "skipped": 0}
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all source tiff files
        source_files = list(source_dir.glob("*.tif"))
        stats = {"total_files": len(source_files), "successful": 0, "failed": 0, "skipped": 0}
        
        for source_file in source_files:
            panoid = source_file.stem
            target_file = target_dir / f"{panoid}.tif"
            
            # Skip if target already exists
            if target_file.exists():
                stats["skipped"] += 1
                continue
            
            try:
                success = self._clip_single_tiff(source_file, target_file, target_pixels, crop_pixels)
                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error clipping {panoid}: {e}")
                stats["failed"] += 1
        
        return stats
    
    def _clip_single_tiff(self, source_path: Path, target_path: Path, 
                         target_pixels: int, crop_pixels: int) -> bool:
        """Clip a single tiff file"""
        
        try:
            with rasterio.open(source_path) as src:
                # Read all bands
                data = src.read()  # Shape: (bands, height, width)
                bands, height, width = data.shape
                
                # Validate dimensions
                if height < target_pixels or width < target_pixels:
                    self.logger.error(f"Source image ({height}x{width}) too small for target ({target_pixels}x{target_pixels})")
                    return False
                
                # Apply center crop
                start_y = crop_pixels
                end_y = start_y + target_pixels
                start_x = crop_pixels  
                end_x = start_x + target_pixels
                
                # Ensure bounds are valid
                if end_y > height or end_x > width:
                    self.logger.error(f"Crop bounds exceed image size")
                    return False
                
                # Crop the data
                cropped_data = data[:, start_y:end_y, start_x:end_x]
                
                # Update metadata
                meta = src.meta.copy()
                meta.update({
                    'height': cropped_data.shape[1],
                    'width': cropped_data.shape[2],
                    'count': cropped_data.shape[0]
                })
                
                # Calculate new transform
                old_transform = src.transform
                pixel_size_x = old_transform[0]
                pixel_size_y = old_transform[4]
                
                offset_x = crop_pixels * pixel_size_x
                offset_y = crop_pixels * pixel_size_y
                
                new_transform = rasterio.transform.Affine(
                    old_transform[0],  
                    old_transform[1],  
                    old_transform[2] + offset_x,  
                    old_transform[3],  
                    old_transform[4],  
                    old_transform[5] + offset_y   
                )
                
                meta['transform'] = new_transform
            
            # Save clipped tiff
            with rasterio.open(target_path, 'w', **meta) as dst:
                for i in range(cropped_data.shape[0]):
                    dst.write(cropped_data[i], i + 1)
                    # Set band descriptions
                    if i < len(self.ground_features):
                        dst.set_band_description(i + 1, self.ground_features[i])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in _clip_single_tiff: {e}")
            return False

    def download_raw_band_data(self, cities: List[Tuple[str, str]], buffer_size: int) -> Dict[str, int]:
        """
        Download raw Sentinel-2 band data
        
        Args:
            cities: List of (country, city) tuples
            buffer_size: Buffer size in meters
            
        Returns:
            Processing statistics
        """
        self.logger.info(f"Starting raw band data download for buffer size: {buffer_size}m")
        self.logger.info(f"Raw bands to download: {self.raw_band_names}")
        
        # Filter bands by resolution (exclude 60m bands)
        valid_bands = [band for band in self.raw_band_names 
                      if band in self.band_resolutions and self.band_resolutions[band] <= 20]
        
        if len(valid_bands) != len(self.raw_band_names):
            filtered_out = set(self.raw_band_names) - set(valid_bands)
            self.logger.warning(f"Filtered out 60m resolution bands: {filtered_out}")
        
        self.logger.info(f"Valid bands for download: {valid_bands}")
        
        total_stats = {"total_points": 0, "successful": 0, "failed": 0}
        
        for country, city in cities:
            self.logger.info(f"Processing raw bands for {country}/{city}")
            
            city_stats = self._download_city_raw_bands(country, city, buffer_size, valid_bands)
            
            # Aggregate statistics
            for key in total_stats:
                total_stats[key] += city_stats.get(key, 0)
            
            self.logger.info(f"{country}/{city}: {city_stats['successful']}/{city_stats['total_points']} successful")
        
        self.logger.info(f"Raw band download complete: {total_stats['successful']}/{total_stats['total_points']} points processed")
        
        return total_stats
    
    def _download_city_raw_bands(self, country: str, city: str, buffer_size: int, 
                               valid_bands: List[str]) -> Dict[str, int]:
        """Download raw bands for a single city"""
        
        # Load GSV metadata
        metadata_file = (self.config.get_data_dir() / "panorama" / "metadata" / 
                        country / f"{city}.json")
        
        if not metadata_file.exists():
            self.logger.error(f"GSV metadata not found: {metadata_file}")
            return {"total_points": 0, "successful": 0, "failed": 0}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            gsv_metadata = json.load(f)
        
        # Filter features by year range
        filtered_features = []
        for feature in gsv_metadata.get('features', []):
            props = feature.get('properties', {})
            pano_year = props.get('year', None)
            if pano_year and self.year_range[0] <= pano_year <= self.year_range[1]:
                filtered_features.append(feature)
        
        self.logger.info(f"Found {len(filtered_features)} valid panoramas for {country}/{city}")
        
        # Create output directory
        output_dir = (self.config.get_data_dir() / "base_maps" / country / city / f"raw_{buffer_size}m")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {"total_points": len(filtered_features), "successful": 0, "failed": 0}
        
        for i, feature in enumerate(filtered_features):
            try:
                success = self._download_single_point_raw_bands(
                    feature, buffer_size, output_dir, valid_bands
                )
                
                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                
                # Progress reporting
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Progress: {i + 1}/{len(filtered_features)} processed")
                    
            except Exception as e:
                self.logger.error(f"Error processing point {i+1}: {e}")
                stats["failed"] += 1
        
        return stats
    
    def _download_single_point_raw_bands(self, feature: Dict, buffer_size: int, 
                                       output_dir: Path, valid_bands: List[str]) -> bool:
        """Download raw bands for a single point"""
        
        # Extract point information
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        if geom.get('type') != 'Point':
            return False
        
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            return False
        
        lon, lat = coords[0], coords[1]
        pano_id = props.get('pano_id', '')
        pano_year = props.get('year', 0)
        pano_month = props.get('month', 1)
        
        if not pano_id:
            return False
        
        # Check if file already exists
        output_file = output_dir / f"{pano_id}.tif"
        if output_file.exists():
            return True
        
        # Create square AOI using existing logic
        bbox, target_crs = self.create_square_aoi_around_point(lat, lon, buffer_size)
        
        # Search for data
        start_date = f"{pano_year}-{pano_month:02d}-01"
        if pano_month == 12:
            end_date = f"{pano_year + 1}-01-01"
        else:
            end_date = f"{pano_year}-{pano_month + 1:02d}-01"
        
        items = self.search_sentinel2_items(bbox, start_date, end_date)
        
        if not items:
            self.logger.debug(f"No data found for {pano_id}")
            return False
        
        try:
            # Calculate target shape
            pixels_per_side = self.calculate_buffer_pixels(buffer_size)
            target_shape = (pixels_per_side, pixels_per_side)
            
            # Create raw band composite
            raw_composite = self.create_raw_band_composite(items, bbox, target_crs, target_shape, valid_bands)
            
            # Save as GeoTIFF
            self.save_raw_bands_as_geotiff(raw_composite, output_file, bbox, target_crs, valid_bands)
            
            self.logger.debug(f"Successfully saved raw bands for {pano_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process raw bands for {pano_id}: {e}")
            return False
    
    def create_raw_band_composite(self, items: List, bbox: Tuple, target_crs: str, 
                                target_shape: Tuple[int, int], valid_bands: List[str]) -> Dict[str, np.ndarray]:
        """
        Create composite from raw Sentinel-2 bands with proper resampling
        
        Args:
            items: List of STAC items
            bbox: Bounding box
            target_crs: Target CRS
            target_shape: Target shape for output arrays
            valid_bands: List of valid band names
            
        Returns:
            Dictionary of band arrays
        """
        if not items:
            raise ValueError("No items found for composite creation")
        
        self.logger.debug(f"Creating raw band composite from {len(items)} scenes")
        
        band_data = {band: [] for band in valid_bands}
        
        successful_items = 0
        for i, item in enumerate(items):
            try:
                self.logger.debug(f"Processing item {i+1}/{len(items)}: {item.id}")
                
                item_bands = {}
                for band_name in valid_bands:
                    try:
                        # Get band resolution
                        band_resolution = self.band_resolutions.get(band_name, 20)
                        
                        # Download and process band with proper resampling
                        band_array = self.download_and_process_raw_band(
                            item, band_name, bbox, target_crs, target_shape, band_resolution
                        )
                        item_bands[band_name] = band_array
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process {band_name} for item {item.id}: {e}")
                        break
                
                # Only add if all bands were successful
                if len(item_bands) == len(valid_bands):
                    for band_name, band_array in item_bands.items():
                        band_data[band_name].append(band_array)
                    successful_items += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process item {item.id}: {e}")
                continue
        
        if successful_items == 0:
            raise ValueError("No items could be processed successfully")
        
        self.logger.debug(f"Successfully processed {successful_items}/{len(items)} items")
        
        # Create median composite for each band
        composite = {}
        for band_name, arrays in band_data.items():
            if arrays:
                # Stack arrays and compute median
                stacked = np.stack(arrays, axis=0)
                composite[band_name] = np.nanmedian(stacked, axis=0)
            else:
                self.logger.warning(f"No data for band {band_name}")
                composite[band_name] = np.full(target_shape, np.nan, dtype=np.float32)
        
        return composite
    
    def download_and_process_raw_band(self, item, band_name: str, bbox: Tuple, target_crs: str, 
                                    target_shape: Tuple[int, int], band_resolution: int) -> np.ndarray:
        """
        Download and process a single raw band with proper resampling
        
        Args:
            item: STAC item
            band_name: Band name (e.g., 'B04')
            bbox: Bounding box in WGS84
            target_crs: Target CRS
            target_shape: Target shape (height, width)
            band_resolution: Original band resolution (10m or 20m)
            
        Returns:
            Numpy array with the processed band data resampled to 20m
        """
        if band_name not in item.assets:
            raise ValueError(f"Band {band_name} not found in item {item.id}")
        
        # Get the asset URL
        asset = item.assets[band_name]
        asset_url = asset.href
        
        # Read the raster data using rasterio
        with rasterio.open(asset_url) as src:
            # Transform bbox to source CRS
            left, bottom, right, top = bbox
            dst_crs = CRS.from_string(target_crs)
            
            # Calculate the window in source coordinates
            src_crs = src.crs
            
            # Transform bounds to source CRS
            if src_crs != CRS.from_epsg(4326):
                left, bottom, right, top = rasterio.warp.transform_bounds(
                    CRS.from_epsg(4326), src_crs, left, bottom, right, top
                )
            
            # Get window for the bounds
            window = from_bounds(left, bottom, right, top, src.transform)
            
            # Read the data for the window
            data = src.read(1, window=window)
            
            # Get the transform for the window
            window_transform = src.window_transform(window)
            
            # Calculate target transform for 20m resolution
            target_pixel_size = self.base_resolution  # 20m
            target_transform, target_width, target_height = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], 
                left, bottom, right, top,
                dst_width=target_shape[1], dst_height=target_shape[0]
            )
            
            # Reproject to target CRS and resolution
            reprojected = np.empty(target_shape, dtype=np.float32)
            
            # Use appropriate resampling method based on resolution change
            if band_resolution == 10:
                # Upsampling 10m to 20m - use average to maintain data integrity
                resampling_method = Resampling.average
            else:
                # 20m to 20m - use bilinear
                resampling_method = Resampling.bilinear
            
            reproject(
                source=data,
                destination=reprojected,
                src_transform=window_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=dst_crs,
                resampling=resampling_method
            )
            
            # Convert to reflectance (Sentinel-2 L2A is scaled by 10000)
            reprojected = reprojected.astype(np.float32) / 10000.0
            
            # Mask invalid values
            reprojected[reprojected <= 0] = np.nan
            reprojected[reprojected >= 1] = np.nan
            
            return reprojected
        
    def save_raw_bands_as_geotiff(self, bands_dict: Dict[str, np.ndarray], output_file: Path,
                                bbox: Tuple, target_crs: str, band_names: List[str]):
        """Save raw bands as a multi-band GeoTIFF"""
        
        # Calculate transform from bbox and array shape
        height, width = list(bands_dict.values())[0].shape
        left, bottom, right, top = bbox
        
        # Transform bbox to target CRS if needed
        if target_crs != "EPSG:4326":
            left, bottom, right, top = rasterio.warp.transform_bounds(
                CRS.from_epsg(4326), CRS.from_string(target_crs), left, bottom, right, top
            )
        
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
        
        # Stack bands in specified order
        band_stack = np.stack([bands_dict[name] for name in band_names], axis=0)
        
        self.logger.debug(f"Saving {len(band_names)} raw bands as {band_stack.shape} array")
        
        # Write to GeoTIFF
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=len(band_names),
            dtype=band_stack.dtype,
            crs=target_crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            for i, (band_name, band_array) in enumerate(zip(band_names, band_stack)):
                dst.write(band_array, i + 1)
                dst.set_band_description(i + 1, band_name)

    def search_sentinel2_items(self, bbox: Tuple[float, float, float, float],
                               start_date: str, end_date: str,
                               max_cloud_cover: Optional[float] = None) -> List:
        """Search for Sentinel-2 items"""
        if max_cloud_cover is None:
            max_cloud_cover = self.config.get_cloud_threshold()
        
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}
        )
        
        items = list(search.get_items())
        return items
    
    def create_square_aoi_around_point(self, lat: float, lon: float, 
                                     buffer_meters: int) -> Tuple[Tuple[float, float, float, float], str]:
        """Create a guaranteed square AOI around a point"""
        
        # Get appropriate UTM CRS for the location
        target_crs = SpatialUtils.get_utm_crs(lat, lon)
        
        # Create point in WGS84
        point_wgs84 = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point_wgs84], crs="EPSG:4326")
        
        # Transform to UTM
        point_utm = point_gdf.to_crs(target_crs)
        utm_x, utm_y = point_utm.geometry.iloc[0].x, point_utm.geometry.iloc[0].y
        
        # Ensure minimum buffer size
        min_buffer = max(buffer_meters, 150)  # At least 150m buffer
        
        # Create perfect square in UTM coordinates
        square_utm = box(
            utm_x - min_buffer,  # minx
            utm_y - min_buffer,  # miny
            utm_x + min_buffer,  # maxx
            utm_y + min_buffer   # maxy
        )
        
        # Transform back to WGS84 for STAC search
        square_gdf = gpd.GeoDataFrame([1], geometry=[square_utm], crs=target_crs)
        square_wgs84 = square_gdf.to_crs("EPSG:4326")
        bounds_wgs84 = square_wgs84.bounds.iloc[0]
        bbox = (bounds_wgs84['minx'], bounds_wgs84['miny'], 
                bounds_wgs84['maxx'], bounds_wgs84['maxy'])
        
        return bbox, target_crs
    
    def download_and_process_band(self, item, band_name: str, bbox: Tuple, target_crs: str, 
                                target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Download and process a single band using direct rasterio access
        
        Args:
            item: STAC item
            band_name: Band name (e.g., 'B04')
            bbox: Bounding box in WGS84
            target_crs: Target CRS
            target_shape: Target shape (height, width)
            
        Returns:
            Numpy array with the processed band data
        """
        if band_name not in item.assets:
            raise ValueError(f"Band {band_name} not found in item {item.id}")
        
        # Get the asset URL (Microsoft Planetary Computer handles signing)
        asset = item.assets[band_name]
        asset_url = asset.href
        
        # Read the raster data using rasterio
        with rasterio.open(asset_url) as src:
            # Transform bbox to source CRS
            left, bottom, right, top = bbox
            dst_crs = CRS.from_string(target_crs)
            
            # Calculate the window in source coordinates
            src_crs = src.crs
            
            # Transform bounds to source CRS
            if src_crs != CRS.from_epsg(4326):
                left, bottom, right, top = rasterio.warp.transform_bounds(
                    CRS.from_epsg(4326), src_crs, left, bottom, right, top
                )
            
            # Get window for the bounds
            window = from_bounds(left, bottom, right, top, src.transform)
            
            # Read the data for the window
            data = src.read(1, window=window)
            
            # Get the transform for the window
            window_transform = src.window_transform(window)
            
            # Calculate target transform
            target_transform, target_width, target_height = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], 
                left, bottom, right, top,
                dst_width=target_shape[1], dst_height=target_shape[0]
            )
            
            # Reproject to target CRS and shape
            reprojected = np.empty(target_shape, dtype=np.float32)
            
            reproject(
                source=data,
                destination=reprojected,
                src_transform=window_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            
            # Convert to reflectance (Sentinel-2 L2A is scaled by 10000)
            reprojected = reprojected.astype(np.float32) / 10000.0
            
            # Mask invalid values
            reprojected[reprojected <= 0] = np.nan
            reprojected[reprojected >= 1] = np.nan
            
            return reprojected
    
    def create_monthly_composite(self, items: List, bbox: Tuple, target_crs: str, 
                               target_shape: Tuple[int, int] = (50, 50)) -> Dict[str, np.ndarray]:
        """
        Create monthly composite using direct rasterio access
        
        Args:
            items: List of STAC items
            bbox: Bounding box
            target_crs: Target CRS
            target_shape: Target shape for output arrays
            
        Returns:
            Dictionary of band arrays
        """
        if not items:
            raise ValueError("No items found for composite creation")
        
        self.logger.info(f"Creating composite from {len(items)} scenes using direct rasterio")
        
        required_bands = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12']
        band_data = {band: [] for band in required_bands}
        
        successful_items = 0
        for i, item in enumerate(items):
            try:
                self.logger.debug(f"Processing item {i+1}/{len(items)}: {item.id}")
                
                item_bands = {}
                for band_name in required_bands:
                    try:
                        band_array = self.download_and_process_band(
                            item, band_name, bbox, target_crs, target_shape
                        )
                        item_bands[band_name] = band_array
                    except Exception as e:
                        self.logger.warning(f"Failed to process {band_name} for item {item.id}: {e}")
                        break
                
                # Only add if all bands were successful
                if len(item_bands) == len(required_bands):
                    for band_name, band_array in item_bands.items():
                        band_data[band_name].append(band_array)
                    successful_items += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process item {item.id}: {e}")
                continue
        
        if successful_items == 0:
            raise ValueError("No items could be processed successfully")
        
        self.logger.info(f"Successfully processed {successful_items}/{len(items)} items")
        
        # Create median composite for each band
        composite = {}
        for band_name, arrays in band_data.items():
            if arrays:
                # Stack arrays and compute median
                stacked = np.stack(arrays, axis=0)
                composite[band_name] = np.nanmedian(stacked, axis=0)
            else:
                self.logger.warning(f"No data for band {band_name}")
                composite[band_name] = np.full(target_shape, np.nan, dtype=np.float32)
        
        return composite
    
    def calculate_ground_features(self, composite: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate ground features from band composite"""
        
        # Extract bands with validation
        required_bands = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12']
        missing_bands = [band for band in required_bands if band not in composite]
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")
        
        blue = composite['B02']
        green = composite['B03']  
        red = composite['B04']
        red_edge = composite['B05']
        nir = composite['B08']
        swir1 = composite['B11']
        swir2 = composite['B12']
        
        self.logger.debug(f"Calculating features from bands with shape: {blue.shape}")
        
        features = {}
        
        # NDVI
        ndvi_denom = nir + red
        features["NDVI"] = np.where(ndvi_denom != 0, (nir - red) / ndvi_denom, np.nan)
        
        # EVI
        evi_denom = nir + 6 * red - 7.5 * blue + 1
        features["EVI"] = np.where(evi_denom != 0, 2.5 * (nir - red) / evi_denom, np.nan)
        
        # MSAVI
        discriminant = (2 * nir + 1)**2 - 8 * (nir - red)
        features["MSAVI"] = np.where(
            discriminant >= 0,
            (2 * nir + 1 - np.sqrt(discriminant)) / 2,
            np.nan
        )
        
        # GNDVI
        gndvi_denom = nir + green
        features["GNDVI"] = np.where(gndvi_denom != 0, (nir - green) / gndvi_denom, np.nan)
        
        # NDRE
        ndre_denom = nir + red_edge
        features["NDRE"] = np.where(ndre_denom != 0, (nir - red_edge) / ndre_denom, np.nan)
        
        # MNDWI
        mndwi_denom = green + swir1
        features["MNDWI"] = np.where(mndwi_denom != 0, (green - swir1) / mndwi_denom, np.nan)
        
        # UI (Urban Index)
        ui_denom = swir2 + nir
        features["UI"] = np.where(ui_denom != 0, (swir2 - nir) / ui_denom, np.nan)
        
        # BSI (Bare Soil Index)
        bsi_denom = (swir1 + red) + (nir + blue)
        features["BSI"] = np.where(bsi_denom != 0, ((swir1 + red) - (nir + blue)) / bsi_denom, np.nan)
        
        # Verify we have all 8 features
        expected_features = ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"]
        calculated_features = list(features.keys())
        
        if len(calculated_features) != 8:
            raise ValueError(f"Expected 8 features, got {len(calculated_features)}: {calculated_features}")
        
        self.logger.info(f"Successfully calculated {len(features)} ground features: {list(features.keys())}")
        
        return features
    
    def save_features_as_geotiff(self, features: Dict[str, np.ndarray], output_file: Path,
                                bbox: Tuple, target_crs: str):
        """Save features as a multi-band GeoTIFF"""
        
        # Calculate transform from bbox and array shape
        height, width = list(features.values())[0].shape
        left, bottom, right, top = bbox
        
        # Transform bbox to target CRS if needed
        if target_crs != "EPSG:4326":
            left, bottom, right, top = rasterio.warp.transform_bounds(
                CRS.from_epsg(4326), CRS.from_string(target_crs), left, bottom, right, top
            )
        
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
        
        # ALL 8 feature names in correct order
        feature_names = ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"]
        
        # Verify all features exist
        missing_features = [name for name in feature_names if name not in features]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Stack features into a single array (8 bands)
        feature_stack = np.stack([features[name] for name in feature_names], axis=0)
        
        self.logger.info(f"Saving {len(feature_names)} features as {feature_stack.shape} array")
        
        # Write to GeoTIFF
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=len(feature_names),  # Should be 8
            dtype=feature_stack.dtype,
            crs=target_crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            for i, (feature_name, feature_array) in enumerate(zip(feature_names, feature_stack)):
                dst.write(feature_array, i + 1)
                dst.set_band_description(i + 1, feature_name)
                
        self.logger.debug(f"Successfully saved {len(feature_names)} bands to {output_file}")
    
    def download_point_based_data(self, country: str, city: str) -> Dict[str, int]:
        """Download data for individual points"""
        
        # Load GSV metadata
        metadata_file = (self.config.get_data_dir() / "panorama" / "metadata" / 
                        country / f"{city}.json")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"GSV metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            gsv_metadata = json.load(f)
        
        # Filter features by year range
        filtered_features = []
        total_features = 0
        
        for feature in gsv_metadata.get('features', []):
            total_features += 1
            props = feature.get('properties', {})
            
            # Extract year and filter
            pano_year = props.get('year', None)
            if pano_year:
                if pano_year >= self.year_range[0] and pano_year <= self.year_range[1]:
                    filtered_features.append(feature)
        
        if not filtered_features:
            raise ValueError(f"No features found in year range {self.year_range} for {country}/{city}")
        
        self.logger.info(f"Processing {len(filtered_features)} panoramas from {total_features} total")
        
        # Process each buffer size
        buffer_sizes = self.config.get_buffer_sizes()
        statistics = {
            "total_points": len(filtered_features),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "buffer_sizes": buffer_sizes
        }
        
        for buffer_size in buffer_sizes:
            self.logger.info(f"Processing buffer size: {buffer_size}m")
            buffer_stats = self._process_buffer_size(country, city, filtered_features, buffer_size)
            statistics[f"buffer_{buffer_size}m"] = buffer_stats
        
        return statistics
    
    def _process_buffer_size(self, country: str, city: str, 
                           features: List[Dict], buffer_size: int) -> Dict[str, int]:
        """Process all points for a specific buffer size"""
        
        # Create output directory
        output_dir = (self.config.get_data_dir() / "base_maps" / country / city / f"buffer_{buffer_size}m")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {"successful": 0, "failed": 0, "skipped": 0}
        
        for i, feature in enumerate(features):
            if i % 50 == 0:
                self.logger.info(f"Processing point {i+1}/{len(features)} for buffer {buffer_size}m")
            
            try:
                result = self._download_single_point(feature, buffer_size, output_dir)
                if result:
                    stats["successful"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                self.logger.error(f"Error processing point {i+1}: {e}")
                stats["failed"] += 1
        
        return stats
    
    def _download_single_point(self, feature: Dict, buffer_size: int, output_dir: Path) -> bool:
        """Download data for a single point using direct rasterio"""
        
        # Extract point information
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        if geom.get('type') != 'Point':
            return False
        
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            return False
        
        lon, lat = coords[0], coords[1]
        pano_id = props.get('pano_id', '')
        pano_year = props.get('year', 0)
        pano_month = props.get('month', 1)
        
        if not pano_id:
            return False
        
        # Check if file already exists
        output_file = output_dir / f"{pano_id}.tif"
        if output_file.exists():
            return True
        
        # Create square AOI
        bbox, target_crs = self.create_square_aoi_around_point(lat, lon, buffer_size)
        
        # Search for data
        start_date = f"{pano_year}-{pano_month:02d}-01"
        if pano_month == 12:
            end_date = f"{pano_year + 1}-01-01"
        else:
            end_date = f"{pano_year}-{pano_month + 1:02d}-01"
        
        items = self.search_sentinel2_items(bbox, start_date, end_date)
        
        if not items:
            self.logger.debug(f"No data found for {pano_id}")
            return False
        
        try:
            # Calculate target shape based on buffer size
            rs_config = self.config.get_rs_data_config()
            base_resolution = rs_config.resolution
            pixels_per_side = max(int(2 * buffer_size / base_resolution), 20)
            target_shape = (pixels_per_side, pixels_per_side)
            
            # Create composite
            composite = self.create_monthly_composite(items, bbox, target_crs, target_shape)
            
            # Calculate features
            features = self.calculate_ground_features(composite)
            
            # Save as GeoTIFF
            self.save_features_as_geotiff(features, output_file, bbox, target_crs)
            
            self.logger.debug(f"Successfully saved {pano_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process {pano_id}: {e}")
            return False


# Test function
def test_enhanced_sentinel_retrieval():
    """Test enhanced Sentinel retrieval functionality"""
    
    try:
        retriever = SentinelDataRetriever()
        
        print("Enhanced SentinelDataRetriever initialized successfully!")
        print(f"Raw bands configured: {retriever.raw_band_names}")
        print(f"Ground features: {retriever.ground_features}")
        
        # Test buffer calculation
        buffer_size = 600
        pixels = retriever.calculate_buffer_pixels(buffer_size)
        print(f"Buffer {buffer_size}m -> {pixels}x{pixels} pixels")
        
        return True
        
    except Exception as e:
        print(f"Enhanced retrieval test failed: {e}")
        return False

def download_raw_rs_data(cities, buffer_size):
    retriever = SentinelDataRetriever()

    stats = retriever.download_raw_band_data(
        cities, 
        buffer_size
    )

    return stats

def download_ground_feature_rs_data(cities):
    for country, city in cities:
        print(f"Processing {city}, {country}...")
        retriever = SentinelDataRetriever()
        base_maps = retriever.download_point_based_data(
            country, city
        )

def clip_buffer_data(cities, method, source_buffer, target_buffer):
    retriever = SentinelDataRetriever()
    stats = retriever.clip_buffer_data(
        source_buffer, 
        target_buffer, 
        cities,
        method # method = "buffer"->ground features or "raw"->raw bands
    )

    return stats

if __name__ == "__main__":
    test_enhanced_sentinel_retrieval()