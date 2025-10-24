"""
Spatial utilities for GeoAI-GVI project
Contains common geospatial operations and coordinate transformations
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
import pyproj
from pyproj import Transformer
from typing import Tuple, List, Union, Optional
import math


class SpatialUtils:
    """
    Collection of spatial utility functions for the GeoAI-GVI project
    """
    
    # Class-level cache for UTM CRS to avoid repeated calculations
    _utm_cache = {}
    
    @staticmethod
    def get_utm_crs(latitude: float, longitude: float) -> str:
        """
        Get appropriate UTM CRS for given coordinates
        
        Args:
            latitude: Latitude in WGS84
            longitude: Longitude in WGS84
            
        Returns:
            EPSG code string for UTM zone
        """
        # Calculate UTM zone
        utm_zone = int((longitude + 180) / 6) + 1
        
        # Determine hemisphere
        if latitude >= 0:
            # Northern hemisphere
            epsg_code = f"EPSG:{32600 + utm_zone}"
        else:
            # Southern hemisphere  
            epsg_code = f"EPSG:{32700 + utm_zone}"
        
        return epsg_code
    
    @staticmethod
    def get_utm_crs_for_area(points_gdf: gpd.GeoDataFrame) -> str:
        """
        Get UTM CRS for an area (using center point)
        More efficient for processing multiple points in same area
        
        Args:
            points_gdf: GeoDataFrame with points
            
        Returns:
            EPSG code string for UTM zone
        """
        # Use area bounds center for UTM determination
        bounds = points_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        return SpatialUtils.get_utm_crs(center_lat, center_lon)
    
    @staticmethod
    def get_utm_crs_cached(latitude: float, longitude: float, 
                          cache_key: Optional[str] = None) -> str:
        """
        Get UTM CRS with caching for repeated calls
        
        Args:
            latitude: Latitude in WGS84
            longitude: Longitude in WGS84
            cache_key: Optional cache key (e.g., city name)
            
        Returns:
            EPSG code string for UTM zone
        """
        if cache_key is None:
            # Create cache key from rounded coordinates (100m precision)
            cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}"
        
        if cache_key not in SpatialUtils._utm_cache:
            SpatialUtils._utm_cache[cache_key] = SpatialUtils.get_utm_crs(latitude, longitude)
        
        return SpatialUtils._utm_cache[cache_key]
    
    @staticmethod
    def create_buffer_polygon(latitude: float, longitude: float, 
                            buffer_meters: float, crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Create buffered polygon around a point
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            buffer_meters: Buffer distance in meters
            crs: Target CRS for accurate distance calculation (auto-detect if None)
            
        Returns:
            GeoDataFrame with buffered polygon in WGS84
        """
        # Create point
        point = Point(longitude, latitude)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        
        # Get appropriate CRS for accurate distance calculation
        if crs is None:
            crs = SpatialUtils.get_utm_crs(latitude, longitude)
        
        # Project to UTM, buffer, then back to WGS84
        projected = point_gdf.to_crs(crs)
        buffered = projected.buffer(buffer_meters)
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=crs)
        result = buffered_gdf.to_crs("EPSG:4326")
        
        return result
    
    @staticmethod
    def batch_create_buffers(points_gdf: gpd.GeoDataFrame, 
                           buffer_meters: float) -> gpd.GeoDataFrame:
        """
        Create buffers for multiple points efficiently (single CRS calculation)
        
        Args:
            points_gdf: GeoDataFrame with points
            buffer_meters: Buffer distance in meters
            
        Returns:
            GeoDataFrame with buffered polygons in WGS84
        """
        # Get single UTM CRS for all points
        utm_crs = SpatialUtils.get_utm_crs_for_area(points_gdf)
        
        # Project all points to UTM
        projected = points_gdf.to_crs(utm_crs)
        
        # Buffer all points at once
        buffered = projected.buffer(buffer_meters)
        buffered_gdf = gpd.GeoDataFrame(
            data=points_gdf.drop('geometry', axis=1) if len(points_gdf.columns) > 1 else None,
            geometry=buffered, 
            crs=utm_crs
        )
        
        # Transform back to WGS84
        result = buffered_gdf.to_crs("EPSG:4326")
        
        return result
    
    @staticmethod
    def create_square_aoi(latitude: float, longitude: float,
                         size_meters: float, crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Create square Area of Interest around a point
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            size_meters: Square side length in meters
            crs: Target CRS (auto-detect if None)
            
        Returns:
            GeoDataFrame with square polygon in WGS84
        """
        # Get appropriate CRS
        if crs is None:
            crs = SpatialUtils.get_utm_crs(latitude, longitude)
        
        # Create point and project to UTM
        point = Point(longitude, latitude)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326")
        projected_point = point_gdf.to_crs(crs)
        
        # Get UTM coordinates
        utm_x = projected_point.geometry.x.iloc[0]
        utm_y = projected_point.geometry.y.iloc[0]
        
        # Create square
        half_size = size_meters / 2
        square = box(
            utm_x - half_size, utm_y - half_size,
            utm_x + half_size, utm_y + half_size
        )
        
        # Create GeoDataFrame and transform back to WGS84
        square_gdf = gpd.GeoDataFrame([1], geometry=[square], crs=crs)
        result = square_gdf.to_crs("EPSG:4326")
        
        return result
    
    @staticmethod
    def calculate_bounds_area(bounds: Tuple[float, float, float, float]) -> float:
        """
        Calculate area of bounding box in km²
        
        Args:
            bounds: (minx, miny, maxx, maxy) in WGS84
            
        Returns:
            Area in square kilometers
        """
        # Create polygon from bounds
        bbox_polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs="EPSG:4326")
        
        # Get center point for appropriate UTM zone
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        utm_crs = SpatialUtils.get_utm_crs(center_lat, center_lon)
        
        # Project and calculate area
        projected = bbox_gdf.to_crs(utm_crs)
        area_m2 = projected.geometry.area.iloc[0]
        area_km2 = area_m2 / 1e6
        
        return area_km2
    
    @staticmethod
    def get_study_area_bounds(points_gdf: gpd.GeoDataFrame, 
                            buffer_meters: float = 1000) -> Tuple[float, float, float, float]:
        """
        Get bounding box for study area with buffer
        
        Args:
            points_gdf: GeoDataFrame with sampling points
            buffer_meters: Buffer distance in meters
            
        Returns:
            Buffered bounds (minx, miny, maxx, maxy)
        """
        # Get original bounds
        original_bounds = points_gdf.total_bounds
        
        # Get center point for UTM calculation
        center_lat = (original_bounds[1] + original_bounds[3]) / 2
        center_lon = (original_bounds[0] + original_bounds[2]) / 2
        
        # Create polygon from bounds and buffer it
        bbox_polygon = box(original_bounds[0], original_bounds[1], 
                          original_bounds[2], original_bounds[3])
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_polygon], crs=points_gdf.crs)
        
        # Get appropriate UTM CRS
        utm_crs = SpatialUtils.get_utm_crs(center_lat, center_lon)
        
        # Project, buffer, and transform back
        projected = bbox_gdf.to_crs(utm_crs)
        buffered = projected.buffer(buffer_meters)
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=utm_crs)
        result = buffered_gdf.to_crs("EPSG:4326")
        
        return tuple(result.total_bounds)
    
    @staticmethod
    def transform_coordinates(coordinates: List[Tuple[float, float]], 
                            from_crs: str, to_crs: str) -> List[Tuple[float, float]]:
        """
        Transform coordinates between different CRS
        
        Args:
            coordinates: List of (x, y) coordinate tuples
            from_crs: Source CRS (e.g., "EPSG:4326")
            to_crs: Target CRS (e.g., "EPSG:3857")
            
        Returns:
            List of transformed coordinates
        """
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        
        transformed = []
        for x, y in coordinates:
            new_x, new_y = transformer.transform(x, y)
            transformed.append((new_x, new_y))
        
        return transformed
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float, 
                          method: str = "haversine") -> float:
        """
        Calculate distance between two points
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates  
            method: Distance calculation method ("haversine" or "vincenty")
            
        Returns:
            Distance in meters
        """
        if method == "haversine":
            return SpatialUtils._haversine_distance(lat1, lon1, lat2, lon2)
        elif method == "vincenty":
            return SpatialUtils._vincenty_distance(lat1, lon1, lat2, lon2)
        else:
            raise ValueError(f"Unknown distance method: {method}")
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two points
        
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        
        return c * r
    
    @staticmethod
    def _vincenty_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate Vincenty distance between two points (more accurate)
        
        Returns:
            Distance in meters
        """
        # Use pyproj for accurate distance calculation
        geod = pyproj.Geod(ellps='WGS84')
        azimuth1, azimuth2, distance = geod.inv(lon1, lat1, lon2, lat2)
        return distance
    
    @staticmethod
    def create_grid_points(bounds: Tuple[float, float, float, float],
                          spacing_meters: float) -> gpd.GeoDataFrame:
        """
        Create grid of points within bounding box
        
        Args:
            bounds: (minx, miny, maxx, maxy) in WGS84
            spacing_meters: Grid spacing in meters
            
        Returns:
            GeoDataFrame with grid points
        """
        # Get center for UTM projection
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        utm_crs = SpatialUtils.get_utm_crs(center_lat, center_lon)
        
        # Create bbox and project to UTM
        bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
        bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326")
        bbox_utm = bbox_gdf.to_crs(utm_crs)
        utm_bounds = bbox_utm.total_bounds
        
        # Create grid in UTM coordinates
        x_coords = np.arange(utm_bounds[0], utm_bounds[2], spacing_meters)
        y_coords = np.arange(utm_bounds[1], utm_bounds[3], spacing_meters)
        
        # Create grid points
        points = []
        for x in x_coords:
            for y in y_coords:
                points.append(Point(x, y))
        
        # Create GeoDataFrame and transform back to WGS84
        grid_utm = gpd.GeoDataFrame(geometry=points, crs=utm_crs)
        grid_wgs84 = grid_utm.to_crs("EPSG:4326")
        
        return grid_wgs84
    
    @staticmethod
    def clip_raster_to_geometry(raster_path: str, geometry: gpd.GeoDataFrame,
                               output_path: Optional[str] = None) -> str:
        """
        Clip raster to geometry bounds
        
        Args:
            raster_path: Path to input raster
            geometry: GeoDataFrame with clipping geometry
            output_path: Output path (auto-generate if None)
            
        Returns:
            Path to clipped raster
        """
        import rasterio
        from rasterio.mask import mask
        from pathlib import Path
        
        # Generate output path if not provided
        if output_path is None:
            input_path = Path(raster_path)
            output_path = input_path.parent / f"{input_path.stem}_clipped{input_path.suffix}"
        
        # Open raster and clip
        with rasterio.open(raster_path) as src:
            # Ensure geometry is in same CRS as raster
            if geometry.crs != src.crs:
                geometry = geometry.to_crs(src.crs)
            
            # Clip raster
            clipped_data, clipped_transform = mask(
                src, geometry.geometry, crop=True, nodata=src.nodata
            )
            
            # Update metadata
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform
            })
            
            # Write clipped raster
            with rasterio.open(output_path, "w", **clipped_meta) as dst:
                dst.write(clipped_data)
        
        return str(output_path)
    
    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> bool:
        """
        Validate coordinate values
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            
        Returns:
            True if coordinates are valid
        """
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
    
    @staticmethod
    def get_coordinate_info(latitude: float, longitude: float) -> dict:
        """
        Get coordinate system information for a point
        
        Args:
            latitude: Latitude in WGS84
            longitude: Longitude in WGS84
            
        Returns:
            Dictionary with coordinate system information
        """
        # Validate coordinates
        if not SpatialUtils.validate_coordinates(latitude, longitude):
            raise ValueError("Invalid coordinates")
        
        # Get UTM information
        utm_crs = SpatialUtils.get_utm_crs(latitude, longitude)
        utm_zone = int(utm_crs.split(':')[1]) - 32600
        if utm_zone < 0:
            utm_zone = int(utm_crs.split(':')[1]) - 32700
            hemisphere = "South"
        else:
            hemisphere = "North"
        
        # Determine general region
        if -180 <= longitude < -30:
            region = "Americas"
        elif -30 <= longitude < 60:
            region = "Europe/Africa"
        elif 60 <= longitude < 150:
            region = "Asia"
        else:
            region = "Pacific"
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "utm_crs": utm_crs,
            "utm_zone": utm_zone,
            "hemisphere": hemisphere,
            "region": region,
            "is_valid": True
        }


# Test function
def test_spatial_utils():
    """Test spatial utilities functionality"""
    
    try:
        # Test coordinate validation
        assert SpatialUtils.validate_coordinates(59.3293, 18.0686), "Valid coordinates failed"
        assert not SpatialUtils.validate_coordinates(91, 0), "Invalid latitude not caught"
        
        # Test UTM CRS detection
        utm_crs = SpatialUtils.get_utm_crs(59.3293, 18.0686)  # Stockholm
        assert utm_crs == "EPSG:32633", f"Expected EPSG:32633, got {utm_crs}"
        
        # Test buffer creation
        buffer_gdf = SpatialUtils.create_buffer_polygon(59.3293, 18.0686, 500)
        assert len(buffer_gdf) == 1, "Buffer should contain one polygon"
        assert buffer_gdf.crs == "EPSG:4326", "Buffer should be in WGS84"
        
        # Test square AOI creation
        square_gdf = SpatialUtils.create_square_aoi(59.3293, 18.0686, 1000)
        assert len(square_gdf) == 1, "Square should contain one polygon"
        
        # Test distance calculation
        dist = SpatialUtils.calculate_distance(59.3293, 18.0686, 59.3393, 18.0786)
        assert 1000 < dist < 2000, f"Distance seems incorrect: {dist}"
        
        # Test coordinate info
        coord_info = SpatialUtils.get_coordinate_info(59.3293, 18.0686)
        assert coord_info["utm_zone"] == 33, "UTM zone should be 33"
        assert coord_info["hemisphere"] == "North", "Should be northern hemisphere"
        
        print("✓ All spatial utility tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Spatial utilities test failed: {e}")
        return False


if __name__ == "__main__":
    test_spatial_utils()