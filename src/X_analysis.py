"""
简化的CNN可解释性分析主函数
只保留Grad-CAM核心功能，输出PDF格式
"""

from src.cnn_training_pipeline import model_tests, run_cnn_training, CNNTrainingPipeline, AdaptiveGVIDataset
from src.cnn_models_config import ModelFactory
import torch
import numpy as np
import os
os.environ["TORCH_HOME"] = "/tmp/torch_cache"
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_grad_cam_report(results, feature_importances, feature_names, 
                          model_type, buffer_size, output_path):
    """创建Grad-CAM可视化报告"""
    
    # 计算平均特征重要性
    avg_importance = np.mean(feature_importances, axis=0)
    std_importance = np.std(feature_importances, axis=0)
    
    # 创建多页PDF报告
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_path / f"grad_cam_analysis_{model_type}_{buffer_size}m.pdf"
    
    with PdfPages(pdf_path) as pdf:
        
        # 第1页：特征重要性总结
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(feature_names))
        bars = ax.bar(x, avg_importance, yerr=std_importance, capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel("Remote Sensing Features", fontsize=12)
        ax.set_ylabel("Average Feature Importance", fontsize=12)
        ax.set_title(f"Feature Importance Analysis - {model_type.title()} Model ({buffer_size}m Buffer)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, avg, std) in enumerate(zip(bars, avg_importance, std_importance)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                    f'{avg:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 第2页：空间注意力图网格
        n_samples = min(12, len(results))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(n_samples):
            result = results[i]
            spatial_attention = result["spatial_attention"]
            target = result["target"]
            prediction = result["prediction"]
            
            im = axes[i].imshow(spatial_attention, cmap='hot', interpolation='nearest')
            axes[i].set_title(f'Sample {i+1}\nActual: {target:.3f}, Pred: {prediction:.3f}', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(n_samples, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Spatial Attention Maps - {model_type.title()} Model ({buffer_size}m Buffer)', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 第3页：预测准确性 vs 注意力强度
        targets = [r["target"] for r in results]
        predictions = [r["prediction"] for r in results]
        attention_means = [np.mean(r["spatial_attention"]) for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 预测vs实际散点图
        ax1.scatter(targets, predictions, alpha=0.7, s=60, color='darkblue')
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Actual GVI Values', fontsize=12)
        ax1.set_ylabel('Predicted GVI Values', fontsize=12)
        ax1.set_title('Prediction Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 注意力强度分布
        ax2.hist(attention_means, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(np.mean(attention_means), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(attention_means):.3f}')
        ax2.set_xlabel('Mean Attention Intensity', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Spatial Attention Distribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    # 保存数据摘要
    summary_data = {
        "model_type": model_type,
        "buffer_size": buffer_size,
        "total_samples": len(results),
        "feature_importance": {
            "features": feature_names,
            "mean_importance": avg_importance.tolist(),
            "std_importance": std_importance.tolist()
        },
        "performance_stats": {
            "mean_attention": float(np.mean(attention_means)),
            "std_attention": float(np.std(attention_means)),
            "prediction_r2": float(np.corrcoef(targets, predictions)[0,1]**2) if len(targets) > 1 else 0.0
        }
    }
    
    with open(output_path / f"grad_cam_summary_{model_type}_{buffer_size}m.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

def create_comparative_grad_cam_grid(all_model_paths, buffer_sizes, model_types, 
                                   input_features, test_cities, output_dir, n_samples=5,input_channels=8):
    """创建横向对比的Grad-CAM网格图"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 收集所有模型的注意力数据
    all_results = {}
    
    for buffer_size in buffer_sizes:
        for model_type in model_types:
            model_key = f"{model_type}_{buffer_size}m"
            model_path = all_model_paths.get(model_key)
            
            if not model_path or not Path(model_path).exists():
                continue
                
            # 加载模型
            input_size = int(2 * buffer_size / 20)

            # 初始化模型
            model = ModelFactory.create_model(model_type, input_channels, input_size)
        #    model = ModelFactory.create_from_config(checkpoint["config"], device=device)
            # 加载权重
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            # 获取测试数据
            pipeline = CNNTrainingPipeline()
            feature_info = pipeline.validate_input_features(input_features)
            data_source = feature_info["data_source"]
            
            gvi_df = pipeline.extract_gvi_labels(test_cities, method="pixel")
            valid_df = pipeline.check_data_availability(gvi_df, buffer_size, data_source)
            
            if len(valid_df) > n_samples:
                valid_df = valid_df.sample(n=n_samples, random_state=42)
            
            dataset = AdaptiveGVIDataset(
                valid_df, pipeline.config.get_data_dir(), buffer_size, 
                input_features, pipeline.feature_config
            )
            dataset.augment = False
            
            # 计算平均注意力图
            attention_maps = []
            
            for i in range(min(n_samples, len(dataset))):
                sample, target = dataset[i]
                sample = sample.to(device)
                
                sample.requires_grad_(True)
                output = model(sample.unsqueeze(0))
                model.zero_grad()
                output.backward()
                
                gradients = sample.grad
                spatial_attention = torch.mean(torch.abs(gradients), dim=0).detach().cpu().numpy()
                attention_maps.append(spatial_attention)
            
            # 计算平均注意力图
            avg_attention = np.mean(attention_maps, axis=0)
            all_results[model_key] = avg_attention
    
    # 创建对比网格图
    fig, axes = plt.subplots(len(model_types), len(buffer_sizes), figsize=(8, 6))
    
    if len(model_types) == 1:
        axes = axes.reshape(1, -1)
    if len(buffer_sizes) == 1:
        axes = axes.reshape(-1, 1)
    
    # 找到全局最大值用于统一色彩范围
    all_values = [result for result in all_results.values()]
    vmin = min(np.min(arr) for arr in all_values) if all_values else 0
    vmax = max(np.max(arr) for arr in all_values) if all_values else 1
    
    for i, model_type in enumerate(model_types):
        for j, buffer_size in enumerate(buffer_sizes):
            model_key = f"{model_type}_{buffer_size}m"
            
            if model_key in all_results:
                attention_map = all_results[model_key]
                im = axes[i, j].imshow(attention_map, cmap='hot', vmin=vmin, vmax=vmax, 
                                     interpolation='nearest')
                axes[i, j].set_title(f'{model_type.title()}\n{buffer_size}m Buffer', fontsize=12)
            else:
                axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                              transform=axes[i, j].transAxes, fontsize=14)
                axes[i, j].set_title(f'{model_type.title()}\n{buffer_size}m Buffer', fontsize=12)
            
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    # 添加统一的颜色条
    # 添加水平颜色条在底部
    """fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average Attention Intensity', fontsize=12)"""

    # plt.suptitle('Model Attention Pattern Comparison\n(Averaged across samples)', fontsize=16, y=0.95)
    # 使用matplotlib的自动布局
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                    fraction=0.05, pad=0.1, shrink=0.8)
    cbar.set_label('Average Attention Intensity', fontsize=11)
    
    # 保存PDF
    pdf_path = output_path / "comparative_grad_cam_analysis.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建特征重要性对比图
    create_feature_importance_comparison(all_model_paths, buffer_sizes, model_types,
                                       input_features, test_cities, output_path)
    
    return all_results

def create_feature_importance_comparison(all_model_paths, buffer_sizes, model_types,
                                       input_features, test_cities, output_path,input_channels=8):
    """创建特征重要性横向对比图"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_feature_importance = {}
    
    for buffer_size in buffer_sizes:
        for model_type in model_types:
            model_key = f"{model_type}_{buffer_size}m"
            model_path = all_model_paths.get(model_key)
            
            if not model_path or not Path(model_path).exists():
                continue
            input_size = int(2 * buffer_size / 20)
            # 加载模型和数据
            model = ModelFactory.create_model(model_type, input_channels, input_size)
        #    model = ModelFactory.create_from_config(checkpoint["config"], device=device)
            # 加载权重
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            pipeline = CNNTrainingPipeline()
            feature_info = pipeline.validate_input_features(input_features)
            data_source = feature_info["data_source"]
            
            gvi_df = pipeline.extract_gvi_labels(test_cities, method="pixel")
            valid_df = pipeline.check_data_availability(gvi_df, buffer_size, data_source)
            
            if len(valid_df) > 10:
                valid_df = valid_df.sample(n=10, random_state=42)
            
            dataset = AdaptiveGVIDataset(
                valid_df, pipeline.config.get_data_dir(), buffer_size, 
                input_features, pipeline.feature_config
            )
            dataset.augment = False
            
            # 计算特征重要性
            feature_importances = []
            for i in range(min(10, len(dataset))):
                sample, _ = dataset[i]
                sample = sample.to(device)
                
                sample.requires_grad_(True)
                output = model(sample.unsqueeze(0))
                model.zero_grad()
                output.backward()
                
                gradients = sample.grad
                channel_importance = torch.mean(torch.abs(gradients), dim=(1, 2)).detach().cpu().numpy()
                feature_importances.append(channel_importance)
            
            avg_importance = np.mean(feature_importances, axis=0)
            all_feature_importance[model_key] = avg_importance
    
    # 创建热力图
    if all_feature_importance:
        # 准备数据矩阵
        importance_matrix = []
        row_labels = []
        
        for model_type in model_types:
            for buffer_size in buffer_sizes:
                model_key = f"{model_type}_{buffer_size}m"
                if model_key in all_feature_importance:
                    importance_matrix.append(all_feature_importance[model_key])
                    row_labels.append(f"{model_type.title()}-{buffer_size}m")
        
        if importance_matrix:
            importance_matrix = np.array(importance_matrix)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(importance_matrix, cmap='RdYlBu_r', aspect='auto')
            
            # 设置标签
            ax.set_xticks(range(len(input_features)))
            ax.set_xticklabels(input_features, rotation=45, ha='right')
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            
            # 添加数值标注
            for i in range(len(row_labels)):
                for j in range(len(input_features)):
                    text = ax.text(j, i, f'{importance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            ax.set_title('Feature Importance Comparison Across Models and Buffer Sizes', 
                        fontsize=14, pad=20)
            ax.set_xlabel('Remote Sensing Features', fontsize=12)
            ax.set_ylabel('Model Configurations', fontsize=12)
            
            # 颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Feature Importance', rotation=270, labelpad=20)
            
            plt.tight_layout()
            
            # 保存PDF
            pdf_path = output_path / "feature_importance_comparison.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close()


def simple_grad_cam_analysis(model_path, model_type, buffer_size, 
                           input_features, test_cities, output_dir, max_samples=20,input_channels=8):
    """简化的Grad-CAM分析"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_size = int(2 * buffer_size / 20)
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelFactory.create_model(model_type, input_channels, input_size)
#    model = ModelFactory.create_from_config(checkpoint["config"], device=device)
    # 加载权重
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 创建测试数据
    pipeline = CNNTrainingPipeline()
    feature_info = pipeline.validate_input_features(input_features)
    data_source = feature_info["data_source"]
    
    gvi_df = pipeline.extract_gvi_labels(test_cities, method="pixel")
    valid_df = pipeline.check_data_availability(gvi_df, buffer_size, data_source)
    
    if len(valid_df) > max_samples:
        valid_df = valid_df.sample(n=max_samples, random_state=42)
    
    dataset = AdaptiveGVIDataset(
        valid_df, pipeline.config.get_data_dir(), buffer_size, 
        input_features, pipeline.feature_config
    )
    dataset.augment = False
    
    # 收集样本数据
    grad_cam_results = []
    feature_importances = []
    
    for i in range(len(dataset)):
        if i >= max_samples:
            break
            
        sample, target = dataset[i]
        sample = sample.to(device)
        
        # Grad-CAM计算
        sample.requires_grad_(True)
        output = model(sample.unsqueeze(0))
        model.zero_grad()
        output.backward()
        
        gradients = sample.grad
        spatial_importance = torch.mean(torch.abs(gradients), dim=0).detach().cpu().numpy()
        channel_importance = torch.mean(torch.abs(gradients), dim=(1, 2)).detach().cpu().numpy()
        
        grad_cam_results.append({
            "sample_idx": i,
            "target": target.item(),
            "prediction": output.item(),
            "spatial_attention": spatial_importance,
            "channel_importance": channel_importance
        })
        
        feature_importances.append(channel_importance)
    
    # 生成可视化报告
    create_grad_cam_report(grad_cam_results, feature_importances, input_features, 
                          model_type, buffer_size, output_path)
    
    return grad_cam_results

def create_grad_cam_report(results, feature_importances, feature_names, 
                          model_type, buffer_size, output_path):
    """创建Grad-CAM可视化报告"""
    
    # 计算平均特征重要性
    avg_importance = np.mean(feature_importances, axis=0)
    std_importance = np.std(feature_importances, axis=0)
    
    # 创建多页PDF报告
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = output_path / f"grad_cam_analysis_{model_type}_{buffer_size}m.pdf"
    
    with PdfPages(pdf_path) as pdf:
        
        # 第1页：特征重要性总结
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(feature_names))
        bars = ax.bar(x, avg_importance, yerr=std_importance, capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel("Remote Sensing Features", fontsize=12)
        ax.set_ylabel("Average Feature Importance", fontsize=12)
        ax.set_title(f"Feature Importance Analysis - {model_type.title()} Model ({buffer_size}m Buffer)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, avg, std) in enumerate(zip(bars, avg_importance, std_importance)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                    f'{avg:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 第2页：空间注意力图网格
        n_samples = min(12, len(results))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(n_samples):
            result = results[i]
            spatial_attention = result["spatial_attention"]
            target = result["target"]
            prediction = result["prediction"]
            
            im = axes[i].imshow(spatial_attention, cmap='hot', interpolation='nearest')
            axes[i].set_title(f'Sample {i+1}\nActual: {target:.3f}, Pred: {prediction:.3f}', fontsize=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # 隐藏多余的子图
        for i in range(n_samples, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Spatial Attention Maps - {model_type.title()} Model ({buffer_size}m Buffer)', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 第3页：预测准确性 vs 注意力强度
        targets = [r["target"] for r in results]
        predictions = [r["prediction"] for r in results]
        attention_means = [np.mean(r["spatial_attention"]) for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 预测vs实际散点图
        ax1.scatter(targets, predictions, alpha=0.7, s=60, color='darkblue')
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Actual GVI Values', fontsize=12)
        ax1.set_ylabel('Predicted GVI Values', fontsize=12)
        ax1.set_title('Prediction Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 注意力强度分布
        ax2.hist(attention_means, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(np.mean(attention_means), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(attention_means):.3f}')
        ax2.set_xlabel('Mean Attention Intensity', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Spatial Attention Distribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    # 保存数据摘要
    summary_data = {
        "model_type": model_type,
        "buffer_size": buffer_size,
        "total_samples": len(results),
        "feature_importance": {
            "features": feature_names,
            "mean_importance": avg_importance.tolist(),
            "std_importance": std_importance.tolist()
        },
        "performance_stats": {
            "mean_attention": float(np.mean(attention_means)),
            "std_attention": float(np.std(attention_means)),
            "prediction_r2": float(np.corrcoef(targets, predictions)[0,1]**2) if len(targets) > 1 else 0.0
        }
    }
    
    with open(output_path / f"grad_cam_summary_{model_type}_{buffer_size}m.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

def run_complete_analysis_pipeline():
    """运行完整的训练+可解释性分析流程"""
    
    green_cities = [
        "Dusseldorf", "Cologne", "Gothenburg", "Manchester", 
        "Hamburg", "Budapest", "Berlin", "Zurich"
    ]
    
    test_cities = [
        "Stockholm", "Helsinki", "Paris", "Milan", "Athens",
        "Barcelona", "Bologna", "Tallinn", "London", 
        "Copenhagen", "Amsterdam"
    ]
    
    model_types = ["micro","simple","original","resnet"]
    buffer_sizes = [40,60,80,100]
    ground_features = ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"]
    
    trained_models = {}
    
    # 训练模型（如果不存在）
    for buffer_size in buffer_sizes:
        for model_type in model_types:
            model_key = f"{model_type}_{buffer_size}m"
            outpath = f"/workspace/data/processed/final_process/green_model/{buffer_size}/{model_type}/train"
            model_path = f"{outpath}/best_model_{buffer_size}m.pth"
            
            if not Path(model_path).exists():
                run_cnn_training(
                    green_cities, buffer_size, ground_features, "train",
                    output_dir=outpath, training_params={"model_type": model_type}
                )
            
            trained_models[model_key] = model_path
    
    # 运行测试
    for buffer_size in buffer_sizes:
        for model_type in model_types:
            model_key = f"{model_type}_{buffer_size}m"
            model_path = trained_models[model_key]
            output_path = f"/workspace/data/processed/final_process/green_model/{buffer_size}/{model_type}"
            
            try:
                model_tests(
                    model_path=model_path, input_feature=ground_features,
                    test_cities=test_cities, buffer_size=buffer_size,
                    model_type=model_type, data_source="buffer", output_dir=output_path
                )
            except Exception as e:
                continue
    
    # 运行Grad-CAM分析
    """for buffer_size in buffer_sizes:
        for model_type in model_types:
            outpath = f"/workspace/data/processed/final_process/green_model/{buffer_size}/{model_type}/train"
            model_path = f"{outpath}/best_model_{buffer_size}m.pth"
            explainability_output = f"/workspace/data/processed/explainability_analysis/{model_type}_{buffer_size}m"
            
            try:
                simple_grad_cam_analysis(
                    model_path=model_path, model_type=model_type, buffer_size=buffer_size,
                    input_features=ground_features, test_cities=test_cities,
                    output_dir=explainability_output, max_samples=20
                )
            except Exception as e:
                continue"""

    # 创建横向对比分析
    """comparative_output = "/workspace/data/processed/comparative_explainability"
    create_comparative_grad_cam_grid(
        all_model_paths=trained_models,
        buffer_sizes=buffer_sizes,
        model_types=model_types,
        input_features=ground_features,
        test_cities=test_cities,
        output_dir=comparative_output,
        n_samples=10
    )"""

if __name__ == "__main__":
    # 运行完整流程
    run_complete_analysis_pipeline()
    
    # 或者单独运行可解释性分析
    # simple_grad_cam_analysis(
    #     model_path="/workspace/data/processed/final_process/green_model/40/simple/train/best_model_40m.pth",
    #     model_type="simple", buffer_size=40,
    #     input_features=["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"],
    #     test_cities=["Stockholm", "Helsinki", "Paris", "Milan"],
    #     output_dir="/workspace/data/processed/explainability_analysis/simple_40m"
    # )
    