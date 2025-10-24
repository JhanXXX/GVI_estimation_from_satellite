"""
Refactored CNN Training Pipeline with Feature Matching and Enhanced Reporting
Integrates smart feature selection, adaptive model creation, and streamlined reporting
"""


import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_project_logger
from src.cnn_models_config import ModelFactory, DataQualityAnalyzer


import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, )):
            return int(obj)
        elif isinstance(obj, (np.floating, )):
            return float(obj)
        elif isinstance(obj, (np.ndarray, )):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class AdaptiveGVIDataset(Dataset):
    """Adaptive dataset supporting both raw bands and ground features"""
    
    def __init__(self, df: pd.DataFrame, data_dir: Path, buffer_size: int,
                 input_features: List[str], feature_config: Dict):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.input_features = input_features
        self.feature_config = feature_config
        self.augment = True
        
        # Determine data source and feature mapping
        self.data_source, self.feature_indices = self._determine_data_source()
        self.logger = get_project_logger(__name__)

    def _determine_data_source(self) -> Tuple[str, List[int]]:
        """
        Determine data source and feature indices based on input features
        
        Returns:
            Tuple of (data_source, feature_indices)
            data_source: "raw" or "ground"
            feature_indices: List of indices to extract from tiff files
        """

        raw_band_names = self.feature_config["raw_band_names"]
        ground_feature_names = self.feature_config["ground_features"]

        raw_features = [f for f in self.input_features if f in raw_band_names]   
        ground_features = [f for f in self.input_features if f in ground_feature_names]

        if raw_features and ground_features:
            raise ValueError(f"Cannot mix raw bands and ground features. "
                           f"Raw: {raw_features}, Ground: {ground_features}")
        if raw_features:
            data_source = "raw"
            raw_band_names = self.feature_config.get("raw_band_names", ["B02", "B03", "B04", "B05","B06", "B07", "B08", "B11", "B12"])
            feature_indices = []
            
            for feature in self.input_features:
                if feature in raw_band_names:
                    feature_indices.append(raw_band_names.index(feature))
                else:
                    raise ValueError(f"Raw band {feature} not found in config raw_band_names: {raw_band_names}")
   
        else:
            data_source = "ground"
            # Map to ground feature indices
            ground_features = self.feature_config.get("ground_features", ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"])
            feature_indices = []
            
            for feature in self.input_features:
                if feature in ground_features:
                    feature_indices.append(ground_features.index(feature))
                else:
                    raise ValueError(f"Ground feature {feature} not found in config ground_features: {ground_features}")
        
        return data_source, feature_indices

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        panoid = row['panoid']
        city = row['city']
        country = row['country'] 
        gvi = row['gvi_average']

        # Load appropriate tiff file
        if self.data_source == "raw":
            tiff_path = (self.data_dir / "base_maps" / country / city / 
                        f"raw_{self.buffer_size}m" / f"{panoid}.tif")
        else:  # ground features
            tiff_path = (self.data_dir / "base_maps" / country / city / 
                        f"buffer_{self.buffer_size}m" / f"{panoid}.tif")

        if not tiff_path.exists():
            self.logger.warning(f"Tiff file not found: {tiff_path}")
            # Return zeros with correct channel count
            return torch.zeros(len(self.input_features), 32, 32), torch.tensor(0.0, dtype=torch.float32)
        
        # Load and process satellite data
        satellite_tensor = self._load_tiff_data(tiff_path)

        if satellite_tensor is None:
            return torch.zeros(len(self.input_features), 32, 32), torch.tensor(0.0, dtype=torch.float32)
        
        gvi_tensor = torch.tensor(gvi, dtype=torch.float32)
        
        # Apply data augmentation
        if self.augment:
            satellite_tensor = self._apply_augmentation(satellite_tensor)
        
        return satellite_tensor, gvi_tensor

    def _load_tiff_data(self, tiff_path: Path) -> Optional[torch.Tensor]:
        """Load tiff data and extract selected features"""
        try:
            with rasterio.open(tiff_path) as src:
                data = src.read()  # Shape: (bands, height, width)
                data = data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)
            
            # Extract selected feature channels
            if len(self.feature_indices) > data.shape[0]:
                self.logger.error(f"Requested {len(self.feature_indices)} features but tiff has {data.shape[0]} bands")
                return None
            
            selected_data = data[self.feature_indices]  # Extract selected bands
            
            return torch.from_numpy(selected_data)
            
        except Exception as e:
            self.logger.error(f"Error loading tiff data from {tiff_path}: {e}")
            return None
    
    def _apply_augmentation(self, satellite_tensor: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        import random
        
        if random.random() > 0.5:
            # Random rotation (90 degree multiples)
            k = random.randint(0, 3)
            satellite_tensor = torch.rot90(satellite_tensor, k, [1, 2])
            
            # Random flips
            if random.random() > 0.5:
                satellite_tensor = torch.flip(satellite_tensor, [1])
            if random.random() > 0.5:
                satellite_tensor = torch.flip(satellite_tensor, [2])
        
        return satellite_tensor

class CNNTrainingPipeline:
    """Enhanced CNN training pipeline with smart feature matching"""
    
    def __init__(self, config_path: str = "config.json", training_params: Dict = None):
        self.config = ConfigLoader(config_path)
        self.logger = get_project_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load feature configurations
        rs_config = self.config.get_rs_data_config()
        self.feature_config = {
            "ground_features": rs_config.ground_features,
            "raw_band_names": self.config.get('raw_band_names', ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']),
            "base_resolution": rs_config.resolution
        }

        # Training parameters with defaults
        default_params = {
            "batch_size": 16,
#            "learning_rate": 0.001,
            "weight_decay": 1e-4,
#            "max_epochs": 100,
            "early_stop_patience": 15,
            "lr_scheduler_patience": 10,
            "lr_scheduler_factor": 0.5,
            "validation_split": 0.2,
            "test_split": 0.15,
            "random_state": 42,
            "test_city": [],
            "model_type": "simple",
            "analyze_data": False,
            "use_k_fold": False,
            "k_folds": 5
        }
        
        self.training_params = {**default_params, **(training_params or {})}
        
        # Initialize database
        self._setup_database()
        
        self.logger.info(f"CNN Training Pipeline initialized with device: {self.device}")


    def _setup_database(self):
        """Setup database tables if needed"""
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        
        if db_path.exists():
            self.logger.debug(f"Using existing database: {db_path}")
        else:
            self.logger.warning(f"Database not found: {db_path}")

    def calculate_buffer_pixels(self, buffer_size: int) -> int:
        """Calculate pixel dimensions for buffer size"""
        base_resolution = self.feature_config["base_resolution"]
        pixels_per_side = int(2 * buffer_size / base_resolution)
        return pixels_per_side
    
    def validate_input_features(self, input_features: List[str]) -> Dict[str, any]:
        """
        Validate input features and determine data source
        
        Args:
            input_features: List of feature names
            
        Returns:
            Validation results with data source information
        """
        if not input_features:
            raise ValueError("input_features cannot be empty")
        
        # Check feature consistency using config
        raw_band_names = self.feature_config["raw_band_names"]
        ground_feature_names = self.feature_config["ground_features"]
        
        raw_features = [f for f in input_features if f in raw_band_names]
        ground_features = [f for f in input_features if f in ground_feature_names]
        
        if raw_features and ground_features:
            raise ValueError(f"Cannot mix raw bands and ground features. "
                           f"Raw: {raw_features}, Ground: {ground_features}")
        
        # Determine data source and validate features
        if raw_features:
            data_source = "raw"
            available_features = self.feature_config["raw_band_names"]
            features_to_check = raw_features
        else:
            data_source = "ground"
            available_features = self.feature_config["ground_features"]
            features_to_check = ground_features
        
        # Validate all features exist
        missing_features = [f for f in features_to_check if f not in available_features]
        if missing_features:
            raise ValueError(f"Features not found in config: {missing_features}. "
                           f"Available {data_source} features: {available_features}")
        
        self.logger.info(f"Feature validation successful:")
        self.logger.info(f"  Data source: {data_source}")
        self.logger.info(f"  Input features: {input_features}")
        self.logger.info(f"  Feature count: {len(input_features)}")
        
        return {
            "data_source": data_source,
            "feature_count": len(input_features),
            "validated_features": input_features
        }

    def extract_gvi_labels(self, cities: List[str], method: str = "pixel") -> pd.DataFrame:
        """Extract GVI labels from database"""
        
        db_config = self.config.get_database_config()
        db_path = Path(db_config.path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        query = """
            SELECT panoid, city, country, gvi_average 
            FROM directional_gvi 
            WHERE method = ? AND city IN ({})
        """.format(','.join(['?' for _ in cities]))
        
        params = [method] + cities
        
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        self.logger.info(f"Extracted {len(df)} GVI labels from {len(cities)} cities using {method} method")
        return df
    
    def check_data_availability(self, gvi_df: pd.DataFrame, buffer_size: int, 
                              data_source: str) -> pd.DataFrame:
        """Check availability of tiff files for the specified data source"""
        
        valid_records = []
        
        for _, row in gvi_df.iterrows():
            panoid = row['panoid']
            city = row['city'] 
            country = row['country']
            
            # Determine tiff path based on data source
            if data_source == "raw":
                tiff_path = (self.config.get_data_dir() / "base_maps" / country / city / 
                           f"raw_{buffer_size}m" / f"{panoid}.tif")
            else:  # ground features
                tiff_path = (self.config.get_data_dir() / "base_maps" / country / city / 
                           f"buffer_{buffer_size}m" / f"{panoid}.tif")
            
            if tiff_path.exists():
                valid_records.append(row)
        
        valid_df = pd.DataFrame(valid_records)
        
        self.logger.info(f"Data availability check:")
        self.logger.info(f"  Available files: {len(valid_df)}/{len(gvi_df)} ({len(valid_df)/len(gvi_df)*100:.1f}%)")
        self.logger.info(f"  Data source: {data_source}")
        self.logger.info(f"  Buffer size: {buffer_size}m")
        
        return valid_df
    
    def prepare_data(self, cities: List[str], buffer_size: int, input_features: List[str],
                    strategy: str, method: str = "pixel") -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Prepare training, validation, and test data with smart feature matching
        
        Args:
            cities: List of cities
            buffer_size: Buffer size in meters
            input_features: List of feature names to use
            strategy: Training strategy ("train" or "generalizability")
            method: GVI calculation method
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        
        # Validate input features
        feature_info = self.validate_input_features(input_features)
        data_source = feature_info["data_source"]
        
        # Extract GVI labels
        gvi_df = self.extract_gvi_labels(cities, method)
        
        if len(gvi_df) == 0:
            raise ValueError(f"No GVI data found for cities: {cities}")
        
        # Check data availability
        valid_df = self.check_data_availability(gvi_df, buffer_size, data_source)
        
        if len(valid_df) == 0:
            raise ValueError(f"No valid data files found for {data_source} features at {buffer_size}m buffer")
        
        # Calculate input size for models
        input_size = self.calculate_buffer_pixels(buffer_size)
        
        # Data splitting
        train_df, val_df, test_df = self._split_data(valid_df, strategy)
        
        # Create datasets
        train_dataset = AdaptiveGVIDataset(
            train_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        )
        val_dataset = AdaptiveGVIDataset(
            val_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        )
        test_dataset = AdaptiveGVIDataset(
            test_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        ) if test_df is not None else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.training_params["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.training_params["batch_size"], shuffle=False
        ) if test_dataset else None
        
        # Log data split information
        self.logger.info(f"Data preparation completed:")
        self.logger.info(f"  Train samples: {len(train_df)}")
        self.logger.info(f"  Validation samples: {len(val_df)}")
        self.logger.info(f"  Test samples: {len(test_df) if test_df is not None else 0}")
        self.logger.info(f"  Input size: {len(input_features)} channels, {input_size}x{input_size} spatial")
        
        return train_loader, val_loader, test_loader
    
    def _split_data(self, valid_df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Split data based on strategy"""
        
        val_split = self.training_params["validation_split"]
        test_split = self.training_params["test_split"]
        random_state = self.training_params["random_state"]
        
        if strategy == "train":
            # Standard random split across all cities
            self.logger.info("Using standard training strategy with random data split")
            
            # First separate test set
            train_val_df, test_df = train_test_split(
                valid_df, 
                test_size=test_split,
                random_state=random_state,
                stratify=valid_df['city'] if len(valid_df['city'].unique()) > 1 else None
            )
            
            # Then split training and validation
            adjusted_val_split = val_split / (1 - test_split)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_split, 
                random_state=random_state,
                stratify=train_val_df['city'] if len(train_val_df['city'].unique()) > 1 else None
            )
            
        elif strategy == "generalizability":
            # Cross-city generalizability test
            test_cities = self.training_params["test_city"]
            if isinstance(test_cities, str):
                test_cities = [test_cities]
            
            available_cities = valid_df['city'].unique()
            valid_test_cities = [city for city in test_cities if city in available_cities]
            
            if not valid_test_cities:
                # Auto-select test cities
                valid_test_cities = list(available_cities[-1:]) if len(available_cities) >= 1 else []
                self.logger.info(f"Auto-selected test cities: {valid_test_cities}")
            
            self.logger.info(f"Using generalizability strategy with test cities: {valid_test_cities}")
            
            # Separate test cities and training cities
            test_df = valid_df[valid_df['city'].isin(valid_test_cities)].copy()
            train_cities_df = valid_df[~valid_df['city'].isin(valid_test_cities)].copy()
            
            if len(train_cities_df) == 0:
                raise ValueError(f"No training data left after excluding test cities {valid_test_cities}")
            
            # Split training cities into train and validation
            train_df, val_df = train_test_split(
                train_cities_df,
                test_size=val_split, 
                random_state=random_state,
                stratify=train_cities_df['city'] if len(train_cities_df['city'].unique()) > 1 else None
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'train' or 'generalizability'")
        
        return train_df, val_df, test_df
    
    def prepare_data_with_kfold(self, cities: List[str], buffer_size: int, input_features: List[str],
                               strategy: str, method: str, fold_id: int, k_folds: int = 5) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Prepare data for K-fold cross validation"""
        
        # Validate input features
        feature_info = self.validate_input_features(input_features)
        data_source = feature_info["data_source"]
        
        # Extract and validate data
        gvi_df = self.extract_gvi_labels(cities, method)
        valid_df = self.check_data_availability(gvi_df, buffer_size, data_source)
        
        self.logger.info(f"K-fold cross-validation setup: K={k_folds}, Fold={fold_id+1}")
        
        if strategy == "train":
            # Standard K-fold on entire dataset
            try:
                kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.training_params["random_state"])
                splits = list(kfold.split(valid_df, valid_df['city']))
            except ValueError:
                kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.training_params["random_state"])
                splits = list(kfold.split(valid_df))
            
            train_val_idx, test_idx = splits[fold_id]
            train_val_df = valid_df.iloc[train_val_idx]
            test_df = valid_df.iloc[test_idx]
            
            # Split train_val into train and validation
            val_split = self.training_params["validation_split"]
            train_df, val_df = train_test_split(
                train_val_df, 
                test_size=val_split,
                random_state=self.training_params["random_state"],
                stratify=train_val_df['city'] if len(train_val_df['city'].unique()) > 1 else None
            )
            
        elif strategy == "generalizability":
            # Generalizability K-fold: test cities fixed, training cities undergo K-fold
            test_cities = self.training_params["test_city"]
            if isinstance(test_cities, str):
                test_cities = [test_cities]
            
            test_df = valid_df[valid_df['city'].isin(test_cities)].copy()
            train_cities_df = valid_df[~valid_df['city'].isin(test_cities)].copy()
            
            # Apply K-fold on training cities only
            try:
                kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.training_params["random_state"])
                splits = list(kfold.split(train_cities_df, train_cities_df['city']))
            except ValueError:
                kfold = KFold(n_splits=k_folds, shuffle=True, random_state=self.training_params["random_state"])
                splits = list(kfold.split(train_cities_df))
            
            train_idx, val_idx = splits[fold_id]
            train_df = train_cities_df.iloc[train_idx]
            val_df = train_cities_df.iloc[val_idx]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create datasets and loaders
        train_dataset = AdaptiveGVIDataset(
            train_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        )
        val_dataset = AdaptiveGVIDataset(
            val_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        )
        test_dataset = AdaptiveGVIDataset(
            test_df, self.config.get_data_dir(), buffer_size, input_features, self.feature_config
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.training_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_params["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.training_params["batch_size"], shuffle=False)
        
        self.logger.info(f"K-fold fold {fold_id+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   input_features: List[str], buffer_size: int, output_dir: str) -> Dict:
        """Train CNN model with adaptive configuration"""
        
        # Calculate input dimensions
        input_channels = len(input_features)
        input_size = self.calculate_buffer_pixels(buffer_size)
        
        # Perform data analysis if enabled
        if self.training_params.get("analyze_data", True):
            self.logger.info("Performing data quality analysis...")
            data_analysis = DataQualityAnalyzer.analyze_data_quality(train_loader, val_loader)
        else:
            data_analysis = {"baseline_r2": 0.5, "train_gvi_std": 0.1}  # Default values
        
        # Create model using ModelFactory
        model_type = self.training_params["model_type"]
        model = ModelFactory.create_model(model_type, input_channels, input_size)
        model.to(self.device)
        
        # Get recommended parameters based on data analysis
        recommended_params = ModelFactory.get_recommended_params(model_type, data_analysis)
        
        # Update training params with recommendations (but keep explicit overrides)
        for key, value in recommended_params.items():
            if key not in self.training_params or key in ["learning_rate", "dropout_rate"]:
                self.training_params[key] = value

        # Setup optimizer and scheduler
        optimizer = optim.AdamW(  # 使用AdamW而不是Adam
            model.parameters(), 
            lr=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        criterion = nn.SmoothL1Loss()
        
        # Training tracking
        train_losses = []
        val_losses = []
        val_r2_scores = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Training {model_type} model:")
        self.logger.info(f"  Input: {input_channels} channels, {input_size}x{input_size} spatial")
        self.logger.info(f"  Parameters: {model.get_param_count():,}")
        self.logger.info(f"  Learning rate: {self.training_params['learning_rate']}")
        self.logger.info(f"  Max epochs: {self.training_params['max_epochs']}")
        
        # Training loop
        for epoch in range(self.training_params["max_epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            predictions = []
            targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data).squeeze()
                    val_loss += criterion(output, target).item()
                    
                    predictions.extend(output.cpu().numpy().flatten().tolist())
                    targets.extend(target.cpu().numpy().flatten().tolist())
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Calculate R² score
            val_r2 = r2_score(targets, predictions)
            val_r2_scores.append(val_r2)
            
            # Learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "model_type": model_type,
                            "params": {
                                "dropout_rate": self.training_params.get("dropout_rate"),
                                "input_channels": self.training_params.get("input_channels", 8),
                                "num_classes": self.training_params.get("num_classes", 1)
                            }
                        }
                    }, Path(output_dir) / f"best_model_{buffer_size}m.pth")
                
            else:
                patience_counter += 1
            
            # Progress logging
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch:3d}: Train={avg_train_loss:.4f}, "
                               f"Val={avg_val_loss:.4f}, R²={val_r2:.4f}")
            
            # Early stopping
            if patience_counter >= self.training_params["early_stop_patience"]:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        checkpoint = torch.load(Path(output_dir) / f"best_model_{buffer_size}m.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return {
            "model": model,
            "model_type": model_type,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_r2_scores": val_r2_scores,
            "final_val_r2": val_r2_scores[-1] if val_r2_scores else 0,
            "epochs_trained": len(train_losses),
            "data_analysis": data_analysis,
            "input_features": input_features,
            "input_dimensions": {"channels": input_channels, 
                                 "spatial_size": input_size
                                 }
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data).squeeze()
                
                predictions.extend(output.cpu().numpy().flatten().tolist())
                targets.extend(target.cpu().numpy().flatten().tolist())
        
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        return {
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "predictions": predictions,
            "targets": targets
        }
    
    def generate_report(self, training_results: Dict, test_results: Dict, buffer_size: int,
                       cities: List[str], test_cities: List[str], strategy: str, 
                       output_dir: str, font_config: Dict = None):
        """
        Generate streamlined training report with configurable fonts
        
        Args:
            training_results: Training results dictionary
            test_results: Test results dictionary  
            buffer_size: Buffer size in meters
            cities: List of training cities
            test_cities: List of test cities
            strategy: Training strategy
            output_dir: Output directory
            font_config: Font configuration dictionary
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up font configuration
        if font_config is None:
            font_config = {
                "family": "Arial",
                "size": 10,
                "title_size": 12,
                "axis_size": 10
            }
        
        # Configure matplotlib fonts
        plt.rcParams.update({
            'font.family': font_config.get("family", "Arial"),
            'font.size': font_config.get("size", 10),
            'axes.titlesize': font_config.get("title_size", 12),
            'axes.labelsize': font_config.get("axis_size", 10),
            'xtick.labelsize': font_config.get("axis_size", 10),
            'ytick.labelsize': font_config.get("axis_size", 10)
        })
        
        # Create streamlined plots
        report_path = output_path / f"training_report_{buffer_size}m.pdf"
        
        with PdfPages(report_path) as pdf:
            plt.style.use('default')
            
            # Plot 1: Training curves (essential only)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            epochs = range(1, len(training_results["train_losses"]) + 1)
            ax.plot(epochs, training_results["train_losses"], label="Train", linewidth=2)
            ax.plot(epochs, training_results["val_losses"], label="Validation", linewidth=2)
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Minimal title with key info
            final_r2 = training_results["final_val_r2"]
            ax.set_title(f"Training Progress (Final R² = {final_r2:.3f})")
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Predictions vs Actual (if test data available)
            if test_results and "predictions" in test_results:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                targets = test_results["targets"]
                predictions = test_results["predictions"]
                
                ax.scatter(targets, predictions, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(min(targets), min(predictions))
                max_val = max(max(targets), max(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                ax.set_xlabel("Ground Truth GVI")
                ax.set_ylabel("Predicted GVI")
                ax.set_title(f"Prediction Accuracy (R² = {test_results['r2_score']:.3f})")
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Plot 3: R² progression
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(epochs, training_results["val_r2_scores"], linewidth=2, color='green')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation R²")
            ax.set_title("Model Performance")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Save detailed results to JSON
        detailed_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "buffer_size": buffer_size,
                "strategy": strategy,
                "cities": cities,
                "test_cities": test_cities,
                "training_method": "standard" if not self.training_params.get("use_k_fold") else "k_fold"
            },
            "model_configuration": {
                "model_type": training_results["model_type"],
                "input_features": training_results["input_features"],
                "input_channels": training_results["input_dimensions"]["channels"],
                "input_spatial_size": training_results["input_dimensions"]["spatial_size"],
                "parameter_count": training_results["model"].get_param_count()
            },
            "training_parameters": {
                "learning_rate": self.training_params["learning_rate"],
                "batch_size": self.training_params["batch_size"],
                "max_epochs": self.training_params["max_epochs"],
                "early_stop_patience": self.training_params["early_stop_patience"],
                "weight_decay": self.training_params["weight_decay"],
                "dropout_rate": self.training_params.get("dropout_rate", "model_default")
            },
            "training_results": {
                "epochs_trained": training_results["epochs_trained"],
                "final_val_r2": float(training_results["final_val_r2"]),
                "best_val_loss": float(min(training_results["val_losses"])) if training_results["val_losses"] else 0,
                "data_analysis_baseline_r2": float(training_results["data_analysis"].get("baseline_r2", 0))
            },
            "test_results": {
                "r2_score": float(test_results.get("r2_score", 0)),
                "mae": float(test_results.get("mae", 0)),
                "rmse": float(test_results.get("rmse", 0))
            } if test_results else {},
            "data_statistics": {
                "train_samples": "see_data_loaders",
                "val_samples": "see_data_loaders", 
                "test_samples": "see_data_loaders",
                "gvi_range": training_results["data_analysis"].get("train_gvi_range", [0, 1]),
                "gvi_std": float(training_results["data_analysis"].get("train_gvi_std", 0))
            }
        }
        
        json_path = output_path / f"results_{buffer_size}m.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, cls=NpEncoder)
        
        self.logger.info(f"Report generated:")
        self.logger.info(f"  Plots: {report_path}")
        self.logger.info(f"  Details: {json_path}")
    
    def train_with_kfold(self, cities: List[str], buffer_size: int, input_features: List[str],
                        strategy: str, method: str, output_dir: str, k_folds: int = 5) -> Dict:
        """Train model with K-fold cross validation"""
        
        self.logger.info(f"Starting {k_folds}-fold cross validation")
        
        fold_results = []
        fold_models = []
        
        for fold in range(k_folds):
            self.logger.info(f"Training fold {fold + 1}/{k_folds}")
            
            # Prepare data for current fold
            train_loader, val_loader, test_loader = self.prepare_data_with_kfold(
                cities, buffer_size, input_features, strategy, method, fold, k_folds
            )
            
            # Create fold-specific output directory
            fold_output_dir = Path(output_dir) / f"fold_{fold + 1}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            training_results = self.train_model(
                train_loader, val_loader, input_features, buffer_size, str(fold_output_dir)
            )
            
            # Evaluate model
            test_results = {}
            if test_loader is not None:
                test_results = self.evaluate_model(training_results["model"], test_loader)
                self.logger.info(f"Fold {fold + 1} test R²: {test_results['r2_score']:.4f}")
            
            # Store fold results
            fold_result = {
                "fold": fold + 1,
                "val_r2": training_results["final_val_r2"],
                "test_r2": test_results.get("r2_score", 0.0),
                "epochs": training_results["epochs_trained"],
                "model_path": fold_output_dir / f"best_model_{buffer_size}m.pth"
            }
            
            fold_results.append(fold_result)
            fold_models.append(training_results["model"])
        
        # Calculate cross-validation statistics
        val_r2_scores = [r["val_r2"] for r in fold_results]
        test_r2_scores = [r["test_r2"] for r in fold_results]
        
        cv_results = {
            "k_folds": k_folds,
            "fold_results": fold_results,
            "models": fold_models,
            "mean_val_r2": np.mean(val_r2_scores),
            "std_val_r2": np.std(val_r2_scores),
            "mean_test_r2": np.mean(test_r2_scores),
            "std_test_r2": np.std(test_r2_scores),
            "best_fold": np.argmax(val_r2_scores) + 1,
            "best_model": fold_models[np.argmax(val_r2_scores)]
        }
        
        self.logger.info(f"K-fold cross validation results:")
        self.logger.info(f"  Mean validation R²: {cv_results['mean_val_r2']:.4f} ± {cv_results['std_val_r2']:.4f}")
        self.logger.info(f"  Mean test R²: {cv_results['mean_test_r2']:.4f} ± {cv_results['std_test_r2']:.4f}")
        self.logger.info(f"  Best fold: {cv_results['best_fold']}")
        
        # Save best model to main output directory
        best_model_source = Path(output_dir) / f"fold_{cv_results['best_fold']}" / f"best_model_{buffer_size}m.pth"
        best_model_dest = Path(output_dir) / f"best_model_{buffer_size}m.pth"
        
        import shutil
        shutil.copy2(best_model_source, best_model_dest)
        
        cv_results['model_path'] = best_model_dest
        
        return cv_results


    def run_training_pipeline(self, cities: List[str], buffer_size: int, input_features: List[str],
                             strategy: str, method: str, output_dir: str, 
                             font_config: Dict = None) -> Dict:
        """
        Main training pipeline with smart feature matching
        
        Args:
            cities: List of cities
            buffer_size: Buffer size in meters
            input_features: List of feature names (B* for raw bands, others for ground features)
            strategy: Training strategy ("train" or "generalizability")
            method: GVI calculation method ("pixel" or "semantic")
            output_dir: Output directory
            font_config: Font configuration for plots
            
        Returns:
            Training results dictionary
        """
        
        self.logger.info(f"Starting CNN training pipeline:")
        self.logger.info(f"  Cities: {cities}")
        self.logger.info(f"  Buffer size: {buffer_size}m")
        self.logger.info(f"  Input features: {input_features}")
        self.logger.info(f"  Strategy: {strategy}")
        self.logger.info(f"  GVI method: {method}")
        
        # Check if using K-fold
        use_k_fold = self.training_params.get("use_k_fold", False)
        k_folds = self.training_params.get("k_folds", 5)
        
        if use_k_fold:
            self.logger.info(f"Using K-fold cross validation (K={k_folds})")
            return self.train_with_kfold(cities, buffer_size, input_features, strategy, method, output_dir, k_folds)
        
        else:
            self.logger.info("Using standard training mode")
            
            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data(
                cities, buffer_size, input_features, strategy, method
            )
            
            # Train model
            training_results = self.train_model(train_loader, val_loader, input_features, buffer_size, output_dir)
            
            # Evaluate on test set
            test_results = {}
            if test_loader is not None:
                test_results = self.evaluate_model(training_results["model"], test_loader)
                self.logger.info(f"Test R² score: {test_results['r2_score']:.4f}")
            
            # Generate report
            test_cities = self.training_params.get("test_city", [])
            if isinstance(test_cities, str):
                test_cities = [test_cities]
            
            self.generate_report(
                training_results, test_results, buffer_size, 
                cities, test_cities, strategy, output_dir, font_config
            )
            
            return {
                "training_results": training_results,
                "test_results": test_results,
                "model_path": Path(output_dir) / f"best_model_{buffer_size}m.pth"
            }
        
def model_tests (model_path, input_feature, buffer_size, test_cities, model_type,
                 input_channels = 8,
                 batch_size = 32,
                 data_source = "buffer", # "buffer" or "raw"
                output_dir = "/workspace/data/processed/model_tests"):
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temperary_pipeline = CNNTrainingPipeline()

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")


    input_size = int(2 * buffer_size / 20)

    # 初始化模型
    model = ModelFactory.create_model(model_type, input_channels, input_size)
#    model = ModelFactory.create_from_config(checkpoint["config"], device=device)
    # 加载权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    config = ConfigLoader()
    rs_config = config.get_rs_data_config()
    feature_config = {
            "ground_features": rs_config.ground_features,
            "raw_band_names": config.get('raw_band_names', ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']),
            "base_resolution": rs_config.resolution
        }
    # Load test data for all cities combined

    # Extract GVI labels
    gvi_df = temperary_pipeline.extract_gvi_labels(test_cities, method="pixel")
    
    if len(gvi_df) == 0:
        raise ValueError("No GVI data found")
    
    # Check data availability
    valid_df = temperary_pipeline.check_data_availability(gvi_df, buffer_size, data_source)
    

    test_dataset = AdaptiveGVIDataset(valid_df, config.get_data_dir(), buffer_size,
                                     input_feature, feature_config)
    logger = test_dataset.logger
    test_dataset.augment = False
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    ) 
    
    # evaluate model
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data).squeeze()
                
                # Handle single sample case
                if output.dim() == 0:
                    output = output.unsqueeze(0)
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                all_predictions.extend(output.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    if not all_predictions:
        raise ValueError("No valid predictions generated")
    logger.info(f"Generated {len(all_predictions)} predictions")
    
    # Get city information for each prediction
    # This assumes the order is preserved from the dataset
    all_cities = valid_df['city'].tolist()
    
    # 确保 all_cities 长度与预测结果匹配
    all_cities = all_cities[:len(all_predictions)]
    # Calculate metrics by city
    city_results = _calculate_city_metrics(
        all_predictions, all_targets, all_cities, test_cities, logger
    )

    # Generate prediction accuracy plots
    pdf_path = output_path / "cross_experiment_prediction_accuracy.pdf"
    _create_prediction_accuracy_plots(city_results, pdf_path)




    # Calculate overall metrics
    overall_r2 = r2_score(all_targets, all_predictions)
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    
    # Prepare summary metrics
    summary_metrics = [
        {
            'city': city,
            'r2_score': results['r2_score'],
            'mae': results['mae'],
            'rmse': results['rmse'],
            'num_samples': results['num_samples']
        }
        for city, results in city_results.items()
    ]
    
    # Save results to JSON
    results_json = {
        'model_path': model_path,
        'test_cities': test_cities,
        'input_features': input_feature,
        'buffer_size': buffer_size,
        'overall_metrics': {
            'r2_score': float(overall_r2),
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'total_samples': len(all_predictions)
        },
        'city_results': {
            city: {
                'r2_score': float(results['r2_score']),
                'mae': float(results['mae']),
                'rmse': float(results['rmse']),
                'num_samples': int(results['num_samples'])
            }
            for city, results in city_results.items()
        },
        'summary_metrics': summary_metrics
    }
    
    json_path = output_path / "cross_experiment_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    logger.info(f"Cross-city experiment completed!")
    logger.info(f"Overall R²: {overall_r2:.4f}, RMSE: {overall_rmse:.4f}, MAE: {overall_mae:.4f}")
    logger.info(f"Results saved to: {json_path}")
    logger.info(f"Plots saved to: {pdf_path}")




def run_cnn_training(cities: List[str], buffer_size: int, input_features: List[str],
                    strategy: str, output_dir: str, method: str = "pixel",
                    training_params: Dict = None, font_config: Dict = None) -> Dict:
    """
    Main function to run CNN training with smart feature matching
    
    Args:
        cities: List of cities
        buffer_size: Buffer size in meters
        input_features: List of feature names
        strategy: Training strategy ("train" or "generalizability")
        output_dir: Output directory
        method: GVI calculation method ("pixel" or "semantic")
        training_params: Custom training parameters
        font_config: Font configuration for plots
        
    Returns:
        Training results dictionary
    """
    
    pipeline = CNNTrainingPipeline(training_params=training_params)
    results = pipeline.run_training_pipeline(
        cities, buffer_size, input_features, strategy, method, output_dir, font_config
    )
    
    print(f"CNN training completed successfully!")
    print(f"Model saved to: {results['model_path']}")
    
    return results


def test_feature_validation():
    """Test feature validation functionality"""
    
    try:
        pipeline = CNNTrainingPipeline()
        
        # Test valid raw bands
        result1 = pipeline.validate_input_features(["B02", "B03", "B04"])
        print(f"Raw bands validation: {result1}")
        
        # Test valid ground features  
        result2 = pipeline.validate_input_features(["NDVI", "EVI", "MSAVI"])
        print(f"Ground features validation: {result2}")
        
        # Test mixed features (should fail)
        try:
            result3 = pipeline.validate_input_features(["B02", "NDVI"])
            print(f"Mixed features validation: {result3}")
        except ValueError as e:
            print(f"Mixed features correctly rejected: {e}")
        
        print("Feature validation test completed!")
        
    except Exception as e:
        print(f"Feature validation test failed: {e}")



def _calculate_city_metrics(predictions: np.ndarray, targets: np.ndarray, 
                            cities: List[str], test_cities: List[str], logger) -> Dict:
        """Calculate metrics for each city"""
        
        city_results = {}
        
        # Convert to DataFrame for easier grouping
        results_df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'city': cities
        })
        
        for city in test_cities:
            city_data = results_df[results_df['city'] == city]
            
            if len(city_data) == 0:
                logger.warning(f"No predictions found for city: {city}")
                continue
            
            city_preds = city_data['prediction'].values
            city_targets = city_data['target'].values
            
            # Calculate metrics
            r2 = r2_score(city_targets, city_preds)
            mae = mean_absolute_error(city_targets, city_preds)
            rmse = np.sqrt(mean_squared_error(city_targets, city_preds))
            
            city_results[city] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'num_samples': len(city_data),
                'predictions': city_preds.tolist(),
                'targets': city_targets.tolist()
            }
            
            logger.info(f"{city}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, N={len(city_data)}")
        
        return city_results

def _create_prediction_accuracy_plots(city_results: Dict, pdf_path: Path):
    """Create prediction accuracy plots and save as PDF"""
    with PdfPages(pdf_path) as pdf:
        plt.style.use('default')
        sns.set_palette("husl")
        
        n_cities = len(city_results)
        if n_cities == 0:
            return
        
        # Page 1: Individual city scatter plots
        n_cols = min(3, n_cities)
        n_rows = (n_cities + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle different subplot arrangements
        if n_cities == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (city, results) in enumerate(city_results.items()):
            ax = axes[i]
            
            predictions = np.array(results['predictions'])
            targets = np.array(results['targets'])
            
            # Scatter plot
            ax.scatter(targets, predictions, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Observed GVI (Pixel Method)')
            ax.set_ylabel('Predicted GVI')
            ax.set_title(f'{city}\nR² = {results["r2_score"]:.3f}, RMSE = {results["rmse"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cities, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Cross-City Prediction Accuracy', fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Summary bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        cities = list(city_results.keys())
        r2_scores = [city_results[city]['r2_score'] for city in cities]
        rmse_scores = [city_results[city]['rmse'] for city in cities]
        
        # R² bar chart
        bars1 = ax1.bar(cities, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Test R² by City')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        if len(cities) > 5:
            ax1.tick_params(axis='x', rotation=45)
        
        # RMSE bar chart
        bars2 = ax2.bar(cities, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('RMSE')
        ax2.set_title('Test RMSE by City')
        ax2.grid(True, alpha=0.3)
        
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        if len(cities) > 5:
            ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Cross-City Performance Summary', fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
if __name__ == "__main__":

    blue_cities = ["Athen",
                   "Barcelona",
                   "Bologna",
                   "Tallin",
                   "London",
                   "Copenhagen",
                   "Amsterdam"
                   ]
    
    azur_cities = ["Stockholm",
                   "Helsinki",
                   "Paris",
                   "Milan",
                   "Dusseldorf",
                   "Koln",
                   "Gothengurg",
                   "Manchester",
                   "Hamburg",
                   "Budapest",
                   "Berlin",
                   "Zurich"]
    
    all_cities = ["Stockholm",
                "Gothenburg",
                "Helsinki",
                "Berlin",
                "Hamburg",
                "Barcelona",
                "Copenhagen",
                "London",
                "Athen",
                "Paris",
                "Amsterdam",
                "Budapest",
                "Manchester",
                "Koln",
                "Dusseldorf",
                "Zurich",
                "Bologna",
                "Milan",
                "Tallin"
    ]