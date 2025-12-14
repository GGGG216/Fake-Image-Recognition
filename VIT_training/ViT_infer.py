"""
推理脚本 - 带可视化
支持：
- 单张图片分类
- 批量图片分类
- 深度图可视化
- 着色图可视化
- 物理特征分析
# 基本推理 + 显示可视化
python infer.py \
    --checkpoint outputs/final_model.pt \
    --image test_images/sample.jpg \
    --show

# 保存所有可视化
python infer.py  --checkpoint outputs/final_model.pt --image data\test\real\img_905_real.jpg  --save_viz  --save_depth  --save_shading   --output_dir inference_results


python infer.py \
    --checkpoint outputs/final_model.pt \
    --image_dir test_images/ \
    --batch_size 8 \
    --output_dir batch_results
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import sys

# 导入主模型
from ViT_train import RealVsAIClassifier, Config


class Inferencer:
    """推理器（带可视化）"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # 加载检查点
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        self.config = checkpoint['config']
        
        # 创建模型
        self.model = RealVsAIClassifier(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 数据转换
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("Model loaded successfully!")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """加载单张图片"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        return_visualization: bool = True
    ) -> Dict:
        """预测单张图片"""
        image_tensor = self.load_image(image_path)
        
        # 推理
        outputs = self.model(image_tensor)
        
        # 整理结果
        result = {
            'image_path': image_path,
            'probability': outputs['probability'].item(),
            'prediction': 'AI-Generated' if outputs['prediction'].item() == 1 else 'Real',
            'confidence': abs(outputs['probability'].item() - 0.5) * 2,  # 0-1
            'physics_features': {
                k: v.item() for k, v in outputs['physics_features'].items()
            }
        }
        
        if return_visualization:
            result['depth_map'] = outputs['depth'].cpu()
            result['shading_map'] = outputs['predicted_shading'].cpu()
            result['original_image'] = image_tensor.cpu()
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 4
    ) -> List[Dict]:
        """批量预测"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = torch.cat([
                self.load_image(path) for path in batch_paths
            ], dim=0)
            
            outputs = self.model(batch_tensors)
            
            for j, path in enumerate(batch_paths):
                result = {
                    'image_path': path,
                    'probability': outputs['probability'][j].item(),
                    'prediction': 'AI-Generated' if outputs['prediction'][j].item() == 1 else 'Real',
                    'confidence': abs(outputs['probability'][j].item() - 0.5) * 2,
                    'physics_features': {
                        k: v[j].item() for k, v in outputs['physics_features'].items()
                    }
                }
                results.append(result)
        
        return results
    
    def visualize_result(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """可视化结果（包括深度图和着色图）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 1. 原始图片
        original = result['original_image'][0]
        original = original * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original = original + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        original = torch.clamp(original, 0, 1)
        original = original.permute(1, 2, 0).numpy()
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(
            f"Original Image\n{result['prediction']} ({result['confidence']*100:.1f}% confident)",
            fontsize=12,
            fontweight='bold'
        )
        axes[0, 0].axis('off')
        
        # 2. 深度图
        depth = result['depth_map'][0, 0].numpy()
        im_depth = axes[0, 1].imshow(depth, cmap='plasma')
        axes[0, 1].set_title("Estimated Depth Map", fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. 预测着色图
        shading = result['shading_map'][0, 0].numpy()
        im_shading = axes[1, 0].imshow(shading, cmap='gray')
        axes[1, 0].set_title("Predicted Shading (Lambertian)", fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im_shading, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 4. 物理特征柱状图
        features = result['physics_features']
        feature_names = [
            'Light\nError',
            'Lum.\nVariance',
            'Shading\nCorr.',
            'Shadow\nConsist.',
            'Highlight\nConsist.'
        ]
        feature_values = list(features.values())
        
        colors = ['red' if result['prediction'] == 'AI-Generated' else 'green'] * len(feature_values)
        bars = axes[1, 1].bar(range(len(feature_values)), feature_values, color=colors, alpha=0.6)
        axes[1, 1].set_xticks(range(len(feature_values)))
        axes[1, 1].set_xticklabels(feature_names, rotation=0, fontsize=9)
        axes[1, 1].set_title("Physics Features", fontsize=12)
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, feature_values):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8
            )
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_depth_map(
        self,
        result: Dict,
        save_path: str,
        colormap: str = 'plasma'
    ):
        """单独保存深度图"""
        depth = result['depth_map'][0, 0].numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(depth, cmap=colormap)
        plt.colorbar(label='Depth')
        plt.title('Depth Map')
        plt.axis('off')
        
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Depth map saved to {save_path}")
    
    def save_shading_map(
        self,
        result: Dict,
        save_path: str
    ):
        """单独保存着色图"""
        shading = result['shading_map'][0, 0].numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(shading, cmap='gray')
        plt.colorbar(label='Shading Intensity')
        plt.title('Predicted Shading (Lambertian Model)')
        plt.axis('off')
        
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Shading map saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Real vs AI-Generated Image Classification Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization')
    parser.add_argument('--save_depth', action='store_true', help='Save depth maps separately')
    parser.add_argument('--save_shading', action='store_true', help='Save shading maps separately')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = Inferencer(args.checkpoint)
    
    # 单张图片推理
    if args.image:
        print(f"\nProcessing: {args.image}")
        result = inferencer.predict(args.image, return_visualization=True)
        
        print(f"\n{'='*50}")
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"\nPhysics Features:")
        for k, v in result['physics_features'].items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*50}\n")
        
        # 可视化
        if args.save_viz or args.show:
            save_path = f"{args.output_dir}/viz_{Path(args.image).stem}.png" if args.save_viz else None
            inferencer.visualize_result(result, save_path=save_path, show=args.show)
        
        # 保存深度图
        if args.save_depth:
            inferencer.save_depth_map(
                result,
                f"{args.output_dir}/depth_{Path(args.image).stem}.png"
            )
        
        # 保存着色图
        if args.save_shading:
            inferencer.save_shading_map(
                result,
                f"{args.output_dir}/shading_{Path(args.image).stem}.png"
            )
    
    # 批量推理
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"\nProcessing {len(image_paths)} images from {args.image_dir}")
        results = inferencer.predict_batch(
            [str(p) for p in image_paths],
            batch_size=args.batch_size
        )
        
        # 打印结果
        print(f"\n{'='*80}")
        for result in results:
            print(f"{Path(result['image_path']).name:40s} | "
                  f"{result['prediction']:15s} | "
                  f"Prob: {result['probability']:.4f} | "
                  f"Conf: {result['confidence']*100:.1f}%")
        print(f"{'='*80}\n")
        
        # 统计
        ai_count = sum(1 for r in results if r['prediction'] == 'AI-Generated')
        real_count = len(results) - ai_count
        print(f"Summary: {real_count} Real, {ai_count} AI-Generated")
        
        # 保存结果到CSV
        import csv
        csv_path = f"{args.output_dir}/batch_results.csv"
        Path(csv_path).parent.mkdir(exist_ok=True, parents=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image_path', 'prediction', 'probability', 'confidence',
                *list(results[0]['physics_features'].keys())
            ])
            writer.writeheader()
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'confidence': result['confidence'],
                    **result['physics_features']
                }
                writer.writerow(row)
        
        print(f"Results saved to {csv_path}")
    
    else:
        print("Error: Please specify either --image or --image_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()