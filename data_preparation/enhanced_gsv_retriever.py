"""
Enhanced Google Street View Panorama Retrieval Module
Added logic to clean up orphaned metadata records and ensure file consistency
"""

import json
import time
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from PIL import Image
import geopandas as gpd
from dataclasses import dataclass

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_project_logger


@dataclass
class PanoramaMetadata:
    """Panorama metadata structure"""
    panoid: str
    lat: float
    lon: float
    date: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    original_lat: Optional[float] = None
    original_lon: Optional[float] = None
    original_id: Optional[int] = None
    status: Optional[str] = None
    copyright: Optional[str] = None


class GSVPanoramaRetriever:
    """
    Google Street View Panorama Retriever using official Street View Static API
    Enhanced with metadata cleanup and file consistency checking
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize GSV retriever"""
        self.config = ConfigLoader(config_path)
        self.logger = get_project_logger(__name__)
        
        # Load GSV configuration
        self.gsv_config = self.config.get_gsv_config()
        
        # Official Google Street View Static API endpoints
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        self.image_url = "https://maps.googleapis.com/maps/api/streetview"
        
        # Get API key
        self.api_key = self.config.get_gsv_api_key()
        if not self.api_key:
            self.logger.warning("No GSV API key found. Please add key to keys/gsv_keys.txt")
        
        # Load configuration settings
        self.request_delay = self.gsv_config.delay
        self.timeout = self.gsv_config.timeout
        self.max_retries = self.gsv_config.max_retries
        self.default_size = self.gsv_config.size
        self.default_fov = self.gsv_config.fov
        self.default_headings = self.gsv_config.headings
        
        self.logger.info(f"GSV Config loaded: size={self.default_size}, fov={self.default_fov}, "
                        f"headings={self.default_headings}")
        
        # Create directories
        self.config.create_directories()
    
    def get_panorama_metadata(self, lat: float, lon: float) -> Optional[PanoramaMetadata]:
        """
        Get panorama metadata using official Street View Static API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            PanoramaMetadata or None
        """
        if not self.api_key:
            self.logger.error("GSV API key required for metadata requests")
            return None
        
        # Round coordinates to 6 decimal places - use this as the standard precision
        lat_standard = round(lat, 6)
        lon_standard = round(lon, 6)
        
        params = {
            'location': f"{lat_standard},{lon_standard}",
            'key': self.api_key
        }
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = requests.get(self.metadata_url, params=params, timeout=self.timeout)
                
                if response.status_code != 200:
                    if attempt < self.max_retries - 1:
                        self.logger.debug(f"HTTP {response.status_code} for {lat_standard}, {lon_standard}, retrying...")
                        continue
                    else:
                        self.logger.warning(f"HTTP {response.status_code} for {lat_standard}, {lon_standard} after {self.max_retries} attempts")
                        return None
                
                # Parse JSON response
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON parse error for {lat_standard}, {lon_standard}: {e}")
                    return None
                
                # Check if panorama exists
                status = data.get('status', '')
                if status != 'OK':
                    self.logger.debug(f"No panorama found for {lat_standard}, {lon_standard}: {status}")
                    return None
                
                # Extract metadata
                pano_id = data.get('pano_id')
                location = data.get('location', {})
                pano_lat = location.get('lat')
                pano_lon = location.get('lng')
                pano_date = data.get('date')
                copyright_info = data.get('copyright')
                
                if not all([pano_id, pano_lat, pano_lon]):
                    self.logger.warning(f"Incomplete panorama data for {lat_standard}, {lon_standard}")
                    return None
                
                # Parse date if available
                year, month, date_str = None, None, None
                if pano_date:
                    try:
                        # Parse date format (usually YYYY-MM)
                        if '-' in pano_date:
                            date_parts = pano_date.split('-')
                            year = int(date_parts[0])
                            if len(date_parts) > 1:
                                month = int(date_parts[1])
                                date_str = f"{year}-{month:02d}"
                        else:
                            # Handle other date formats
                            year = int(pano_date[:4]) if len(pano_date) >= 4 else None
                            date_str = pano_date
                    except (ValueError, IndexError):
                        self.logger.warning(f"Could not parse date '{pano_date}' for {lat_standard}, {lon_standard}")
                        date_str = pano_date
                
                return PanoramaMetadata(
                    panoid=pano_id,
                    lat=float(pano_lat),
                    lon=float(pano_lon),
                    date=date_str,
                    year=year,
                    month=month,
                    original_lat=lat_standard,  # Use standardized 6-decimal precision
                    original_lon=lon_standard,  # Use standardized 6-decimal precision
                    status=status,
                    copyright=copyright_info
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.logger.debug(f"Error getting metadata for {lat_standard}, {lon_standard} (attempt {attempt + 1}): {e}")
                    continue
                else:
                    self.logger.error(f"Error getting metadata for {lat_standard}, {lon_standard} after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def download_directional_images(self, panoid: str,
                                   headings: Optional[List[int]] = None,
                                   size: Optional[str] = None,
                                   fov: Optional[int] = None,
                                   country: str = "",
                                   city: str = "") -> Dict[int, np.ndarray]:
        """
        Download directional images for a panoid using config parameters
        
        Args:
            panoid: Panorama ID
            headings: Camera headings (uses config default if None)
            size: Image size (uses config default if None)
            fov: Field of view (uses config default if None)
            country: Country name
            city: City name
            
        Returns:
            Dictionary mapping heading to image array
        """
        if not self.api_key:
            self.logger.error("GSV API key required for image requests")
            return {}
        
        # Use config defaults if not specified
        if headings is None:
            headings = self.default_headings
        if size is None:
            size = self.default_size
        if fov is None:
            fov = self.default_fov
        
        images = {}
        
        for heading in headings:
            params = {
                'pano': panoid,
                'size': size,
                'fov': fov,
                'heading': heading,
                'key': self.api_key,
                'return_error_code': 'true'
            }
            
            # Retry mechanism for each heading
            success = False
            for attempt in range(self.max_retries):
                try:
                    time.sleep(self.request_delay)
                    response = requests.get(self.image_url, params=params, timeout=self.timeout)
                    
                    if response.status_code == 404:
                        self.logger.debug(f"No imagery available for panoid {panoid} heading {heading}")
                        break
                    elif response.status_code != 200:
                        if attempt < self.max_retries - 1:
                            self.logger.debug(f"HTTP {response.status_code} for panoid {panoid} heading {heading}, retrying...")
                            continue
                        else:
                            self.logger.warning(f"HTTP {response.status_code} for panoid {panoid} heading {heading} after {self.max_retries} attempts")
                            break
                    
                    # Load image
                    image = Image.open(BytesIO(response.content))
                    images[heading] = np.array(image)
                    success = True
                    
                    # Save individual directional image if paths provided
                    if country and city:
                        pano_dir = (self.config.get_data_dir() / "panorama" / "previews" / 
                                   country / city / panoid)
                        pano_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_path = pano_dir / f"{heading}.jpg"
                        image.save(output_path, "JPEG", quality=95)
                        self.logger.debug(f"Directional image saved: {output_path}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.logger.debug(f"Error downloading heading {heading} for panoid {panoid} (attempt {attempt + 1}): {e}")
                        continue
                    else:
                        self.logger.warning(f"Error downloading heading {heading} for panoid {panoid} after {self.max_retries} attempts: {e}")
                        break
        
        success_count = len(images)
        self.logger.debug(f"Downloaded {success_count}/{len(headings)} directional images for panoid {panoid}")
        
        return images
    
    def _load_existing_metadata(self, country: str, city: str) -> Tuple[Dict[str, dict], List[dict]]:
        """
        Load existing metadata from GeoJSON file and return both indexed and raw features
        
        Returns:
            Tuple of (panoid_indexed_metadata, raw_features_list)
        """
        metadata_file = (self.config.get_data_dir() / "panorama" / "metadata" / 
                        country / f"{city}.json")
        
        existing_metadata = {}
        raw_features = []
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                
                raw_features = geojson_data.get('features', [])
                
                # Index by panoid
                for feature in raw_features:
                    props = feature['properties']
                    panoid = props.get('pano_id')
                    if panoid:
                        existing_metadata[panoid] = props
                
                self.logger.info(f"Loaded {len(existing_metadata)} existing panoramas")
                
            except Exception as e:
                self.logger.warning(f"Error loading existing metadata: {e}")
        
        return existing_metadata, raw_features
    
    def _check_directional_images_exist(self, panoid: str, country: str, city: str,
                                       headings: Optional[List[int]] = None) -> bool:
        """Check if all directional images exist for a panoid"""
        if headings is None:
            headings = self.default_headings
            
        pano_dir = (self.config.get_data_dir() / "panorama" / "previews" / 
                   country / city / panoid)
        
        if not pano_dir.exists():
            return False
        
        for heading in headings:
            image_path = pano_dir / f"{heading}.jpg"
            if not image_path.exists():
                return False
        
        return True
    
    def _cleanup_orphaned_metadata(self, country: str, city: str, 
                                  headings: Optional[List[int]] = None) -> Dict[str, dict]:
        """
        Clean up metadata records that don't have corresponding image files
        
        Args:
            country: Country name
            city: City name
            headings: Expected headings to check
            
        Returns:
            Dictionary of valid metadata (panoid -> metadata_dict)
        """
        if headings is None:
            headings = self.default_headings
        
        self.logger.info("Checking for orphaned metadata records...")
        
        existing_metadata, raw_features = self._load_existing_metadata(country, city)
        
        if not existing_metadata:
            self.logger.info("No existing metadata to clean up")
            return {}
        
        valid_metadata = {}
        orphaned_panoids = []
        
        # Check each panoid for corresponding files
        for panoid, metadata in existing_metadata.items():
            if self._check_directional_images_exist(panoid, country, city, headings):
                valid_metadata[panoid] = metadata
            else:
                orphaned_panoids.append(panoid)
                self.logger.info(f"Found orphaned metadata: {panoid} (missing image files)")
        
        # If we found orphaned records, rewrite the metadata file
        if orphaned_panoids:
            self.logger.info(f"Cleaning up {len(orphaned_panoids)} orphaned metadata records")
            
            # Filter out orphaned features
            valid_features = []
            for feature in raw_features:
                panoid = feature['properties'].get('pano_id')
                if panoid and panoid not in orphaned_panoids:
                    valid_features.append(feature)
            
            # Save cleaned metadata file
            self._save_metadata_geojson(valid_features, country, city)
            
            self.logger.info(f"Metadata cleanup complete: {len(valid_metadata)} valid records remaining")
        else:
            self.logger.info("No orphaned metadata found - all records have corresponding files")
        
        return valid_metadata
    
    def _check_point_exists_with_files(self, lat: float, lon: float, 
                                      existing_metadata: Dict[str, dict],
                                      country: str, city: str,
                                      tolerance: float = 1e-6,
                                      headings: Optional[List[int]] = None) -> bool:
        """
        Check if point exists in metadata AND has corresponding image files
        
        Args:
            lat: Latitude
            lon: Longitude  
            existing_metadata: Existing metadata dictionary (after cleanup)
            country: Country name
            city: City name
            tolerance: Coordinate matching tolerance
            headings: Expected headings to check
            
        Returns:
            True if point exists in metadata and has all image files
        """
        if headings is None:
            headings = self.default_headings
            
        # Use same 6-decimal standardization as get_panorama_metadata
        lat_standard = round(lat, 6)
        lon_standard = round(lon, 6)
        
        for panoid, metadata in existing_metadata.items():
            existing_lat = metadata.get('original_lat')
            existing_lon = metadata.get('original_lon')

            if (existing_lat is not None and existing_lon is not None and
                abs(lat_standard - existing_lat) < tolerance and 
                abs(lon_standard - existing_lon) < tolerance):
                
                # Check if files exist for this panoid
                if self._check_directional_images_exist(panoid, country, city, headings):
                    return True
                else:
                    # This should not happen after cleanup, but just in case
                    self.logger.warning(f"Found coordinate match for {lat_standard},{lon_standard} "
                                      f"but panoid {panoid} missing files")
                    return False
        
        return False
    
    def _save_metadata_geojson(self, features: List[dict], country: str, city: str,
                              additional_metadata: dict = None) -> str:
        """Save features to metadata GeoJSON file"""
        output_dir = self.config.get_data_dir() / "panorama" / "metadata" / country
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{city}.json"
        
        # Create or update metadata info
        metadata_info = {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_panoramas": len(features),
            "country": country,
            "city": city,
            "api_version": "Street View Static API",
            "config_used": {
                "headings": self.default_headings,
                "fov": self.default_fov,
                "size": self.default_size
            }
        }
        
        if additional_metadata:
            metadata_info.update(additional_metadata)
        
        # Create complete GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
            "metadata": metadata_info
        }
        
        # Save file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(features)} panoramas to {output_path}")
        
        return str(output_path)
    
    def load_points_from_shapefile(self, shapefile_path: str, 
                                  sample_size: Optional[int] = None) -> gpd.GeoDataFrame:
        """
        Load points from shapefile for testing
        
        Args:
            shapefile_path: Path to shapefile
            sample_size: Number of points to sample (all if None)
            
        Returns:
            GeoDataFrame with points
        """
        try:
            points_gdf = gpd.read_file(shapefile_path)
            
            self.logger.info(f"Loaded {len(points_gdf)} points from {shapefile_path}")
            self.logger.info(f"CRS: {points_gdf.crs}")
            self.logger.info(f"Bounds: {points_gdf.total_bounds}")
            
            # Sample if requested
            if sample_size and len(points_gdf) > sample_size:
                points_gdf = points_gdf.sample(n=sample_size, random_state=42)
                self.logger.info(f"Sampled {sample_size} points for testing")
            
            # Ensure we're in WGS84
            if points_gdf.crs != "EPSG:4326":
                points_gdf = points_gdf.to_crs("EPSG:4326")
                self.logger.info("Converted to WGS84 (EPSG:4326)")
            
            return points_gdf
            
        except Exception as e:
            self.logger.error(f"Error loading shapefile {shapefile_path}: {e}")
            raise
    
    def batch_download_panoramas(self, points_gdf: gpd.GeoDataFrame,
                                country: str, city: str,
                                max_points: Optional[int] = None,
                                skip_existing: bool = True,
                                headings: Optional[List[int]] = None) -> int:
        """
        Enhanced batch download with metadata cleanup and file consistency checking
        
        Args:
            points_gdf: GeoDataFrame with coordinate points
            country: Country name
            city: City name
            max_points: Maximum points to process (uses config test_batch_size if None)
            skip_existing: Whether to skip existing points (with both metadata and files)
            headings: Camera headings (uses config default if None)
            
        Returns:
            Number of points successfully processed
        """
        if not self.api_key:
            self.logger.error("GSV API key required for batch processing")
            return 0
        
        # Use config default for test batch size
        if max_points is None:
            max_points = self.gsv_config.test_batch_size
        
        if max_points and len(points_gdf) > max_points:
            points_gdf = points_gdf.sample(n=max_points, random_state=42)
            self.logger.info(f"Processing {max_points} points for testing")
        elif max_points and len(points_gdf) <= max_points:
            max_points = len(points_gdf)
            points_gdf = points_gdf.sample(n=max_points, random_state=42)
            self.logger.info(f"Processing {max_points} points for testing")

        # Use config default headings
        if headings is None:
            headings = self.default_headings
        
        self.logger.info(f"=== Starting enhanced batch processing for {country}/{city} ===")
        
        # Step 1: Clean up orphaned metadata (metadata without corresponding files)
        valid_existing_metadata = {}
        if skip_existing:
            valid_existing_metadata = self._cleanup_orphaned_metadata(country, city, headings)
        
        # Step 2: Filter points that need processing
        points_to_process = []
        skipped_count = 0
        
        for idx, (_, row) in enumerate(points_gdf.iterrows()):
            lat = row.geometry.y
            lon = row.geometry.x
            
            should_skip = False
            
            if skip_existing:
                # Check if point exists with both metadata and files
                if self._check_point_exists_with_files(lat, lon, valid_existing_metadata, 
                                                     country, city, headings=headings):
                    self.logger.debug(f"Point {lat:.6f},{lon:.6f} exists with complete files, skipping")
                    should_skip = True

            if should_skip:
                skipped_count += 1
            else:
                points_to_process.append((idx, row))
        
        total_points = len(points_gdf)
        points_to_download = len(points_to_process)
        
        self.logger.info(f"Batch processing summary:")
        self.logger.info(f"  Total points: {total_points}")
        self.logger.info(f"  Valid existing (metadata + files): {len(valid_existing_metadata)}")
        self.logger.info(f"  Points to download: {points_to_download}")
        self.logger.info(f"  Skipped (complete): {skipped_count}")
        
        if points_to_download == 0:
            self.logger.info("No new points to process")
            return 0
        
        successful_points = 0
        new_metadata_list = []
        
        self.logger.info(f"Starting directional panorama download: {points_to_download} points")
        
        for idx, (original_idx, row) in enumerate(points_to_process):
            lat = row.geometry.y
            lon = row.geometry.x
            
            try:
                # Get panorama metadata first
                metadata = self.get_panorama_metadata(lat, lon)
                
                if not metadata:
                    self.logger.debug(f"No panorama metadata found for {lat:.6f},{lon:.6f}")
                    continue
                
                # Check if we already have this panoid in valid existing metadata
                if skip_existing and metadata.panoid in valid_existing_metadata:
                    self.logger.debug(f"Panoid {metadata.panoid} already exists with complete files, skipping")
                    continue
                
                # Check year range before downloading
                if int(metadata.year) < self.gsv_config.year_range[0]:
                    self.logger.debug(f"Panorama {metadata.panoid} year {metadata.year} out of range, skipping")
                    continue
                
                # Download directional images
                directional_images = self.download_directional_images(
                    metadata.panoid, headings=headings, country=country, city=city
                )
                
                if directional_images:
                    successful_points += 1
                    
                    # Set additional metadata
                    metadata.original_id = row.get('id', original_idx)
                    new_metadata_list.append(metadata)
                    
                    self.logger.debug(f"Successfully processed panoid {metadata.panoid} "
                                    f"with {len(directional_images)} directional images, date: {metadata.date}")
                else:
                    self.logger.warning(f"Failed to download directional images for panoid {metadata.panoid}")
                
                # Progress logging using config interval
                if (idx + 1) % self.gsv_config.progress_interval == 0:
                    self.logger.info(f"Progress: {idx + 1}/{points_to_download} points, "
                                   f"{successful_points} successful")
                    
            except Exception as e:
                self.logger.error(f"Error processing point {lat:.6f},{lon:.6f}: {e}")
                continue
                
            # Save new metadata every 50 points
            if idx % 50 == 0 and new_metadata_list:
                self._append_metadata_geojson(new_metadata_list, country, city)
                self.logger.info(f"Intermediate save: {len(new_metadata_list)} new panoramas added")
                new_metadata_list = []
                time.sleep(5)  # Sleep 5 seconds
        
        # Final save of remaining metadata
        if new_metadata_list:
            self._append_metadata_geojson(new_metadata_list, country, city)
        
        self.logger.info(f"Batch download complete:")
        self.logger.info(f"  New points processed: {successful_points}/{points_to_download}")
        self.logger.info(f"  Total new panoramas: {len(new_metadata_list)}")
        
        return successful_points
    
    def _append_metadata_geojson(self, new_metadata_list: List[PanoramaMetadata], 
                                country: str, city: str) -> str:
        """Append new metadata to existing GeoJSON file"""
        
        # Load existing data
        _, existing_features = self._load_existing_metadata(country, city)
        
        # Create new features - IMPORTANT: Use GSV returned coordinates, not original
        new_features = []
        for metadata in new_metadata_list:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [metadata.lon, metadata.lat]  # GSV returned coordinates
                },
                "properties": {
                    "pano_id": metadata.panoid,
                    "date": metadata.date,
                    "year": metadata.year,
                    "month": metadata.month,
                    "original_lat": metadata.original_lat,  # Original query coordinates
                    "original_lon": metadata.original_lon,  # Original query coordinates
                    "gsv_lat": metadata.lat,  # GSV returned coordinates (for clarity)
                    "gsv_lon": metadata.lon,  # GSV returned coordinates (for clarity)
                    "original_id": metadata.original_id,
                    "status": metadata.status,
                    "copyright": metadata.copyright
                }
            }
            new_features.append(feature)
        
        # Combine existing and new features
        all_features = existing_features + new_features
        
        # Save with updated metadata info
        additional_metadata = {
            "new_panoramas_added": len(new_features),
            "cleanup_performed": True
        }
        
        output_path = self._save_metadata_geojson(all_features, country, city, additional_metadata)
        
        self.logger.info(f"Appended {len(new_features)} new panoramas to {output_path}")
        self.logger.info(f"Total panoramas: {len(all_features)}")
        
        return output_path
    
    def get_pano_batch(self, city: str, country: str) -> bool:
        """
        Enhanced batch processing with metadata cleanup
        
        Args:
            city: City name
            country: Country name
            
        Returns:
            True if processing successful
        """
        shapefile_path: str = f"./data/resource/{country}/{city}.shp"
        
        try:
            self.logger.info(f"=== Enhanced {country}/{city} Processing ===")
            
            # Load points
            points_gdf = self.load_points_from_shapefile(shapefile_path)
            
            if len(points_gdf) == 0:
                self.logger.error("No points loaded from shapefile")
                return False
            
            # Process batch with enhanced logic
            successful_count = self.batch_download_panoramas(
                points_gdf, 
                country, 
                city,
                skip_existing=True
            )
            
            self.logger.info(f"{city} enhanced batch processing complete: {successful_count} panoramas processed")
            
            # Check output files
            metadata_file = self.config.get_data_dir() / "panorama" / "metadata" / country / f"{city}.json"
            images_dir = self.config.get_data_dir() / "panorama" / "previews" / country / f"{city}"
            
            metadata_exists = metadata_file.exists()
            images_exist = images_dir.exists() and len(list(images_dir.glob("*"))) > 0
            
            self.logger.info(f"Output verification:")
            self.logger.info(f"  Metadata file: {metadata_exists}")
            self.logger.info(f"  Images directory: {images_exist}")
            
            return successful_count >= 0 and metadata_exists
            
        except Exception as e:
            self.logger.error(f"{city} enhanced batch processing failed: {e}")
            return False


# Test function
def test_enhanced_gsv_retriever():
    """Test enhanced GSV retriever functionality"""
    
    try:
        retriever = GSVPanoramaRetriever()
        print("Enhanced GSV Panorama Retriever initialized")
        print(f"Config: headings={retriever.default_headings}, fov={retriever.default_fov}, size={retriever.default_size}")
        
        if not retriever.api_key:
            print("Warning: No API key found - please add GSV API key to keys/gsv_keys.txt")
            return False
        
        # Test metadata cleanup functionality
        print("🧹 Testing metadata cleanup functionality...")
        
        # Test with a sample city (you can modify this)
        test_country = "Finland"
        test_city = "Helsinki"
        
        # Check for orphaned metadata
        valid_metadata = retriever._cleanup_orphaned_metadata(test_country, test_city)
        print(f"Metadata cleanup test completed: {len(valid_metadata)} valid records found")
        
        # Test single point
        test_lat = 60.1699  # Helsinki center
        test_lon = 24.9384
        
        print(f"Testing single point metadata retrieval: {test_lat}, {test_lon}")
        metadata = retriever.get_panorama_metadata(test_lat, test_lon)
        
        if metadata:
            print(f"Metadata retrieved: panoid={metadata.panoid}, date={metadata.date}")
            print(f"   Original coords: {metadata.original_lat}, {metadata.original_lon}")
            print(f"   GSV coords: {metadata.lat}, {metadata.lon}")
            
            # Test file existence check
            files_exist = retriever._check_directional_images_exist(
                metadata.panoid, test_country, test_city
            )
            print(f"   Files exist: {files_exist}")
            
        else:
            print("No metadata found for test point")
        
        print("Enhanced GSV API functionality test completed")
        return True
        
    except Exception as e:
        print(f"Enhanced GSV Retriever test failed: {e}")
        return False

def download_gsv_panos(cities):
    retriever = GSVPanoramaRetriever()
    for country, city in cities:
        retriever.get_pano_batch(city, country)

if __name__ == "__main__":
    test_enhanced_gsv_retriever()