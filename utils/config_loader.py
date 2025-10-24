"""
Configuration loader utility
Handles loading and validation of project configuration
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str


@dataclass  
class RSDataConfig:
    """Remote sensing data configuration"""
    resolution: int
    merge_method: str
    bands: list
    ground_features: list


@dataclass
class GSVConfig:
    """Google Street View configuration"""
    api_key_file: str
    size: str
    fov: int
    year_range: list
    headings: list
    delay: float
    timeout: int
    max_retries: int
    test_batch_size: int
    progress_interval: int


@dataclass
class TemporalConfig:
    """Temporal alignment configuration"""
    strategy: str
    year_range: list


@dataclass
class ModelTrainingConfig:
    """Model training configuration"""
    test_size: float
    random_state: int
    cv_folds: int


class ConfigLoader:
    """
    Load and manage project configuration
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            # Validate configuration
            self._validate_config()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_keys = [
            'data_dir', 'log_dir', 'buffer_sizes', 
            'rs_data', 'database', 'gsv'
        ]
        
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate buffer sizes
        if not isinstance(self._config['buffer_sizes'], list):
            raise ValueError("buffer_sizes must be a list")
        
        if not all(isinstance(x, int) and x > 0 for x in self._config['buffer_sizes']):
            raise ValueError("buffer_sizes must contain positive integers")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'rs_data.resolution')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self.get('database', {})
        return DatabaseConfig(
            path=db_config.get('path', './data/processed/GVI_training_labels.db')
        )
    
    def get_rs_data_config(self) -> RSDataConfig:
        """Get remote sensing data configuration"""
        rs_config = self.get('rs_data', {})
        return RSDataConfig(
            resolution=rs_config.get('resolution', 10),
            merge_method=rs_config.get('merge_method', 'median'),
            bands=rs_config.get('bands', ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']),
            ground_features=rs_config.get('ground_features', ['NDVI', 'EVI', 'MSAVI', 'GNDVI', 'NDRE', 'MNDWI', 'UI', 'BSI'])
        )
    
    def get_gsv_config(self) -> GSVConfig:
        """Get Google Street View configuration"""
        gsv_config = self.get('gsv', {})
        image_settings = gsv_config.get('image_settings', {})
        request_settings = gsv_config.get('request_settings', {})
        batch_settings = gsv_config.get('batch_settings', {})

        return GSVConfig(
            api_key_file=gsv_config.get('api_key_file', './keys/gsv_keys.txt'),
            year_range=gsv_config.get('year_range', [2020, 2024]),
            size=image_settings.get('size', '640x640'),
            fov=image_settings.get('fov', 120),
            headings=image_settings.get('headings', [0, 90, 180, 270]),
            delay=request_settings.get('delay', 0.1),
            timeout=request_settings.get('timeout', 15),
            max_retries=request_settings.get('max_retries', 3),
            test_batch_size=batch_settings.get('test_batch_size', 20),
            progress_interval=batch_settings.get('progress_interval', 5),
        )
    
    def get_temporal_config(self) -> TemporalConfig:
        """Get temporal alignment configuration"""
        temporal_config = self.get('temporal_alignment', {})
        return TemporalConfig(
            strategy=temporal_config.get('strategy', 'same_month'),
            year_range=temporal_config.get('year_range', [2022, 2024])
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        """Get model training configuration"""
        training_config = self.get('model_training', {})
        return ModelTrainingConfig(
            test_size=training_config.get('test_size', 0.2),
            random_state=training_config.get('random_state', 42),
            cv_folds=training_config.get('cv_folds', 5)
        )
    
    def get_data_dir(self) -> Path:
        """Get data directory path"""
        return Path(self.get('data_dir', './data'))
    
    def get_log_dir(self) -> Path:
        """Get log directory path"""
        return Path(self.get('log_dir', './logs'))
    
    def get_keys_dir(self) -> Path:
        """Get keys directory path"""
        return Path(self.get('keys_dir', './keys'))
    
    def get_models_dir(self) -> Path:
        """Get models directory path"""
        return Path(self.get('models_dir', './models/final_models'))
    
    def get_buffer_sizes(self) -> list:
        """Get buffer sizes"""
        return self.get('buffer_sizes', [200, 400, 600, 800, 1000])
    
    def get_cloud_threshold(self) -> float:
        """Get cloud coverage threshold"""
        return self.get('cloud_filter_threshold', 10.0)
    
    def update_config(self, key: str, value: Any, save: bool = False):
        """
        Update configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
            save: Whether to save to file immediately
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        # Support dot notation for nested keys
        keys = key.split('.')
        config_ref = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
        
        if save:
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        if self._config is None:
            raise RuntimeError("No configuration to save")
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Error saving configuration: {e}")
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
    
    def get_gsv_api_key(self) -> Optional[str]:
        """Get Google Street View API key from file"""
        gsv_config = self.get_gsv_config()
        key_file = Path(gsv_config.api_key_file)
        
        if not key_file.exists():
            return None
        
        try:
            with open(key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None
    
    def create_directories(self):
        """Create necessary project directories"""
        directories = [
            self.get_data_dir(),
            self.get_log_dir(), 
            self.get_keys_dir(),
            self.get_models_dir(),
            self.get_data_dir() / "base_maps",
            self.get_data_dir() / "panorama" / "metadata",
            self.get_data_dir() / "panorama" / "previews", 
            self.get_data_dir() / "resource",
            self.get_data_dir() / "processed",
            self.get_models_dir() / "checkpoints",
            self.get_models_dir() / "final_models",
            self.get_models_dir() / "model_reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        if self._config is None:
            return "ConfigLoader(not loaded)"
        
        return f"ConfigLoader(config_path={self.config_path}, keys={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration"""
        return self.__str__()


# Convenience function for quick access
def load_config(config_path: str = "config.json") -> ConfigLoader:
    """
    Load configuration with default path
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded ConfigLoader instance
    """
    return ConfigLoader(config_path)


# Test function
def test_config_loader():
    """Test configuration loader functionality"""
    try:
        # Test with default config
        config = ConfigLoader()
        
        print("✓ Configuration loaded successfully")
        print(f"✓ Data directory: {config.get_data_dir()}")
        print(f"✓ Buffer sizes: {config.get_buffer_sizes()}")
        print(f"✓ Cloud threshold: {config.get_cloud_threshold()}")
        
        # Test configuration objects
        rs_config = config.get_rs_data_config()
        print(f"✓ Ground features: {rs_config.ground_features}")
        
        # Test GSV config
        gsv_config = config.get_gsv_config()
        print(f"✓ GSV Config: size={gsv_config.size}, fov={gsv_config.fov}, headings={gsv_config.headings}")
        
        # Test directory creation
        config.create_directories()
        print("✓ Directories created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


#if __name__ == "__main__":
#    test_config_loader()