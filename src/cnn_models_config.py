"""
Refactored CNN Models Configuration with Adaptive Input Support
Integrates all model definitions, configurations, and data analysis tools
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import os
import pandas as pd

class DataQualityAnalyzer:
    """Data quality analysis tool for training datasets"""
    
    @staticmethod
    def analyze_data_quality(train_loader, val_loader=None, output_dir = None):
        """Analyze data quality and distribution"""
        
        print("=" * 50)
        print("Data Quality Analysis")
        print("=" * 50)
        
        # Analyze training data
        train_targets, train_features = DataQualityAnalyzer._extract_data(train_loader, "Training")
        
        # Analyze validation data if provided
        if val_loader:
            val_targets, val_features = DataQualityAnalyzer._extract_data(val_loader, "Validation")
        
        # Feature correlation analysis
        DataQualityAnalyzer._feature_correlation_analysis(train_features, train_targets, output_dir)

        # Linear regression baseline
        baseline_r2 = DataQualityAnalyzer._linear_baseline(train_features, train_targets, output_dir)        
        
        return {
            "train_gvi_range": (train_targets.min(), train_targets.max()),
            "train_gvi_std": train_targets.std(),
            "baseline_r2": baseline_r2,
            "recommendations": DataQualityAnalyzer._get_recommendations(baseline_r2, train_targets)
        }
    
    @staticmethod
    def _extract_data(loader, data_type):
        """Extract data features and labels"""
        all_targets = []
        all_features = []
        
        for data, target in loader:
            all_targets.extend(target.numpy())
            # Calculate spatial average features per sample
            feature_stats = data.mean(dim=(2,3))  # (batch, channels)
            all_features.append(feature_stats)
        
        targets = np.array(all_targets)
        features = torch.cat(all_features, dim=0).numpy()
        
        print(f"\n{data_type} data statistics:")
        print(f"  Sample count: {len(targets)}")
        print(f"  GVI range: {targets.min():.3f} - {targets.max():.3f}")
        print(f"  GVI mean: {targets.mean():.3f}")
        print(f"  GVI std: {targets.std():.3f}")
        print(f"  Feature dimensions: {features.shape}")
        
        # Check data quality issues
        if targets.std() < 0.05:
            print(f"Warning: GVI std is very low ({targets.std():.3f}), may affect model learning")
        
        if np.any(np.isnan(features)) or np.any(np.isnan(targets)):
            print(f"Error: NaN values detected in data")
        
        return targets, features
    
    @staticmethod
    def _linear_baseline(features, targets, output_dir = None): #必须要在feature correlation函数之后调用！！
        """Linear regression baseline"""
        try:
            lr = LinearRegression()
            lr.fit(features, targets)
            baseline_score = lr.score(features, targets)
            
            print(f"\nLinear regression baseline:")
            print(f"  R² score: {baseline_score:.4f}")
            
            if baseline_score < 0.1:
                print(f"Warning: Linear baseline R² is low, features may have weak correlation with GVI")
            elif baseline_score > 0.5:
                print(f"Good: Linear baseline R² is high, features have predictive value")


            # 追加到文件
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, "feature_correlation.csv")

                # 如果文件存在 -> append
                if os.path.exists(out_path):
                    df = pd.read_csv(out_path)
                else:
                    df = pd.DataFrame(columns=["Feature", "Correlation"])

                df = pd.concat([df, pd.DataFrame([["LinearBaseline", baseline_score]], 
                                                 columns=["Feature", "Correlation"])])
                df.to_csv(out_path, index=False)
                print(f"[INFO] Linear baseline appended to {out_path}")
            
            return baseline_score

        except Exception as e:
            print(f"Linear regression failed: {e}")
            return 0.0
            
    @staticmethod
    def _feature_correlation_analysis(features, targets, output_dir=None):
        """Feature correlation analysis"""
        print(f"\nFeature correlation with GVI:")
        correlations = []

        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], targets)[0, 1]
            correlations.append((f"Feature_{i}", corr))
            print(f"  Feature {i}: {corr:.3f}")

        # 排序
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\nMost correlated features:")
        for name, corr in correlations[:3]:
            print(f"  {name}: {corr:.3f}")

        # 保存结果
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(correlations, columns=["Feature", "Correlation"])
            out_path = os.path.join(output_dir, "feature_correlation.csv")
            df.to_csv(out_path, index=False)
            print(f"\n[INFO] Feature correlation results saved to {out_path}")
        
    
    @staticmethod
    def _get_recommendations(baseline_r2, targets):
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if baseline_r2 < 0.1:
            recommendations.append("Consider more complex feature engineering or different modeling approaches")
            recommendations.append("Check data preprocessing correctness")
        
        if targets.std() < 0.05:
            recommendations.append("GVI variability is low, consider expanding data range or using regression normalization")
        
        if baseline_r2 > 0.3:
            recommendations.append("Feature quality is good, deep learning models should work well")
        
        return recommendations


class SimpleGVICNN(nn.Module):
    """Simple CNN to avoid overfitting"""
    
    def __init__(self, input_channels=8, input_size=32, **kwarg):
        super().__init__()
        
        # Calculate adaptive pooling sizes based on input size
        pool1_size = max(input_size // 4, 4)
        pool2_size = max(input_size // 8, 2)
        
        self.features = nn.Sequential(
            # First layer: process spectral features
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool1_size, pool1_size)),
            
            # Second layer: extract spatial features
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool2_size, pool2_size)),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Simple regression head
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.features(x)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class MicroGVICNN(nn.Module):
    """Micro CNN with minimal parameters"""
    
    def __init__(self, input_channels=8, input_size=32, dropout_rate = 0.5, **kwargs):
        super().__init__()
        
        # Depth-wise separable convolution for parameter reduction
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # Depth-wise
            nn.Conv2d(input_channels, 16, 1),  # Point-wise
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class LightweightGVICNN(nn.Module):
    """Original lightweight CNN architecture"""
    
    def __init__(self, input_channels=8, input_size=32, dropout_rate=0.3, 
                 conv_channels=(8, 16, 32), fc_sizes=(64, 16), **kwargs):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # 添加2D dropout
            
            # 第二层
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            # 第三层
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Calculate adaptive pooling size based on input size
        pool_size = max(input_size // 4, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(conv_channels[2] * pool_size * pool_size, fc_sizes[0]),
            nn.BatchNorm1d(fc_sizes[0]),  # 添加BN
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.BatchNorm1d(fc_sizes[1]),  # 添加BN
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(fc_sizes[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class ResNetGVICNN(nn.Module):
    """ResNet-based CNN with pretrained weights adaptation"""
    
    def __init__(self, input_channels=8, input_size=32, pretrained=True):
        super().__init__()
        
        # Use ResNet18 as backbone
        if pretrained:
            backbone = models.resnet18(pretrained=True)
            print(f"Using pretrained ResNet18 weights")
        else:
            backbone = models.resnet18(pretrained=False)
            print(f"Using randomly initialized ResNet18")
        
        # Modify first layer for custom input channels
        original_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Adapt pretrained weights for different input channels
        if pretrained and input_channels != 3:
            with torch.no_grad():
                old_weights = original_conv1.weight  # (64, 3, 7, 7)
                new_weights = torch.zeros(64, input_channels, 7, 7)
                
                if input_channels >= 3:
                    # Copy RGB weights to first 3 channels
                    new_weights[:, :3, :, :] = old_weights
                    
                    # For additional channels, use average of RGB weights
                    if input_channels > 3:
                        avg_weights = old_weights.mean(dim=1, keepdim=True)
                        for i in range(3, input_channels):
                            new_weights[:, i:i+1, :, :] = avg_weights
                else:
                    # For fewer than 3 channels, use average
                    avg_weights = old_weights.mean(dim=1, keepdim=True)
                    for i in range(input_channels):
                        new_weights[:, i:i+1, :, :] = avg_weights
                
                backbone.conv1.weight = nn.Parameter(new_weights)
        
        # Remove original classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),  # ResNet18 final feature dimension is 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class ModelFactory:
    """Factory class for creating CNN models with adaptive configurations"""
    
    # Model configuration registry
    MODEL_CONFIGS = {
        "simple": {
            "class": SimpleGVICNN,
            "description": "Simple CNN to avoid overfitting",
            "default_params": {
                "dropout_rate": 0.3,
                "learning_rate": 0.0005,
                "batch_size": 32,
                "weight_decay": 1e-4,
                "max_epochs": 100,
                "early_stop_patience": 15
            }
        },
        "micro": {
            "class": MicroGVICNN,
            "description": "Micro CNN with minimal parameters",
            "default_params": {
                "dropout_rate": 0.2,
                "learning_rate": 0.0003,
                "batch_size": 8,
                "weight_decay": 1e-4,
                "max_epochs": 200,
                "early_stop_patience": 30
            }
        },
        "original": { 
            "class": LightweightGVICNN,
            "description": "Original lightweight CNN architecture",
            "default_params": {
                "dropout_rate": 0.3,
                "learning_rate": 0.0003,
                "batch_size": 16,
                "weight_decay": 1e-4,
                "max_epochs": 100,
                "early_stop_patience": 15
            }
        },
        "resnet": {
            "class": ResNetGVICNN,
            "description": "ResNet18-based CNN with pretrained weights",
            "default_params": {
                "dropout_rate": 0.35,
                "learning_rate": 0.0005,
                "batch_size": 64,
                "weight_decay": 1e-3,
                "max_epochs": 80,
                "early_stop_patience": 12
            }
        }
    }

    @staticmethod
    def create_from_config(config, device="cpu"):
        """
        Create a model instance from config and optionally load weights safely.

        Args:
            config (dict): 包含至少 "model_type" 和可选的超参数
            checkpoint_path (str, optional): 如果提供，则加载 state_dict
            device (str): "cpu" 或 "cuda"
        
        Returns:
            model (nn.Module): 已初始化（并可能加载了权重）的模型
        """
        model_type = config.get("model_type")
        if model_type not in ModelFactory.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = ModelFactory.MODEL_CONFIGS[model_type]["class"]
 
        # 合并默认参数和用户传入参数
        model_params = config.get("params", {})  # 直接用保存下来的参数
        # 🚨 过滤掉模型构造函数不接受的参数
        import inspect
        valid_keys = inspect.signature(model_class.__init__).parameters.keys()
        valid_params = {k: v for k, v in model_params.items() if k in valid_keys}

        # 初始化模型
        model = model_class(**valid_params).to(device)
        #model = ModelFactory.create_model(model_type, input_channels, input_size)
        #model.to(device)
        return model
    
    @staticmethod
    def create_model(model_type: str, input_channels: int, input_size: int, **kwargs) -> nn.Module:
        """
        Create model with adaptive input configuration
        
        Args:
            model_type: Type of model ("simple", "micro", "original", "resnet")
            input_channels: Number of input channels
            input_size: Input spatial size (assumes square images)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Configured CNN model
        """
        if model_type not in ModelFactory.MODEL_CONFIGS:
            available_models = list(ModelFactory.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        config = ModelFactory.MODEL_CONFIGS[model_type]
        model_class = config["class"]
        
        # Create model with adaptive parameters
        if model_type == "original":
            # Original model has additional configuration options
            model = model_class(
                input_channels=input_channels,
                input_size=input_size,
                dropout_rate=kwargs.get("dropout_rate", 0.3),
                conv_channels=kwargs.get("conv_channels", (8, 16, 32)),
                fc_sizes=kwargs.get("fc_sizes", (64, 16))
            )
        else:
            # Other models use standard parameters
            model = model_class(
                input_channels=input_channels,
                input_size=input_size,
                **{k: v for k, v in kwargs.items() if k in ["pretrained"]}
            )
        
        param_count = model.get_param_count()
        print(f"Created {model_type} model: {param_count:,} parameters")
        print(f"Input configuration: {input_channels} channels, {input_size}x{input_size} spatial")
        
        return model
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """Get model information and description"""
        if model_type not in ModelFactory.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = ModelFactory.MODEL_CONFIGS[model_type]
        return {
            "description": config["description"],
            "default_params": config["default_params"].copy(),
            "class_name": config["class"].__name__
        }
    
    @staticmethod
    def get_recommended_params(model_type: str, data_analysis_results: Dict) -> Dict[str, Any]:
        """
        Get recommended training parameters based on data analysis
        
        Args:
            model_type: Type of model
            data_analysis_results: Results from DataQualityAnalyzer
            
        Returns:
            Recommended training parameters
        """
        if model_type not in ModelFactory.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Start with default parameters
        params = ModelFactory.MODEL_CONFIGS[model_type]["default_params"].copy()
        
        # Adjust based on data quality analysis
        baseline_r2 = data_analysis_results.get("baseline_r2", 0)
        gvi_std = data_analysis_results.get("train_gvi_std", 0.1)
        
        # Adjust learning rate based on baseline performance
        if baseline_r2 < 0.1:
            params["learning_rate"] *= 0.5  # Lower learning rate for difficult data
            params["max_epochs"] = int(params["max_epochs"] * 1.5)  # More epochs
        elif baseline_r2 > 0.5:
            params["learning_rate"] *= 1.2  # Slightly higher learning rate for good data
        
        # Adjust regularization based on GVI variability
        if gvi_std < 0.05:
            params["dropout_rate"] = min(0.7, params["dropout_rate"] + 0.2)  # More dropout
            params["weight_decay"] *= 2  # Stronger weight decay
        elif gvi_std > 0.2:
            params["dropout_rate"] = max(0.1, params["dropout_rate"] - 0.1)  # Less dropout
        
        # Model-specific adjustments
        if model_type == "micro":
            # Micro model needs more patience
            params["early_stop_patience"] = max(params["early_stop_patience"], 25)
        elif model_type == "resnet":
            # Pretrained model converges faster
            params["early_stop_patience"] = min(params["early_stop_patience"], 15)
        
        return params
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List all available model types"""
        return list(ModelFactory.MODEL_CONFIGS.keys())
    
    @staticmethod
    def print_model_summary():
        """Print summary of all available models"""
        print("Available CNN Models:")
        print("=" * 50)
        
        for model_type, config in ModelFactory.MODEL_CONFIGS.items():
            params = config["default_params"]
            print(f"\n{model_type.upper()}:")
            print(f"  Description: {config['description']}")
            print(f"  Default LR: {params['learning_rate']}")
            print(f"  Default Batch Size: {params['batch_size']}")
            print(f"  Default Epochs: {params['max_epochs']}")

def usage_example ():
    """
    # 1. 列出可用模型
    ModelFactory.print_model_summary()

    # 2. 创建适应性模型
    model = ModelFactory.create_model(
        model_type="resnet", 
        input_channels=7,     # 7个raw bands
        input_size=15         # 150m buffer → 15x15 pixels
    )

    # 3. 获取推荐参数
    params = ModelFactory.get_recommended_params("resnet", data_analysis_results)
    """
    
# Test and usage examples
if __name__ == "__main__":
    # Test model factory
    print("Testing ModelFactory:")
    ModelFactory.print_model_summary()
    
    # Test model creation with different configurations
    print(f"\nTesting adaptive model creation:")
    
    test_configs = [
        {"channels": 8, "size": 32, "type": "simple"},
        {"channels": 7, "size": 20, "type": "micro"},  # Raw bands
        {"channels": 5, "size": 15, "type": "original"},  # Selected features
        {"channels": 12, "size": 40, "type": "resnet"}   # Combined features
    ]
    
    for config in test_configs:
        try:
            model = ModelFactory.create_model(
                config["type"], config["channels"], config["size"]
            )
            print(f"  {config['type']} - {config['channels']}ch, {config['size']}x{config['size']}: OK")
            
        except Exception as e:
            print(f"  {config['type']} - {config['channels']}ch, {config['size']}x{config['size']}: Failed ({e})")
    
    print(f"\nModelFactory test completed!")