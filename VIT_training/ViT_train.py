"""
Real vs AI-Generated Image Classifier - 优化版
基于物理光照一致性 + Vision Transformer
改进：
- Robust光源估计（多区域采样）
- 自适应阈值
- Percentile归一化
- Sobel梯度
"""
"""
外部调用：
# your_code.py（与train.py同目录）

from train import PhysicsFeatureExtractor, Config
from PIL import Image
from torchvision import transforms
import torch

# 1. 初始化（只需一次）
config = Config()
extractor = PhysicsFeatureExtractor(config).to(config.device)
extractor.eval()

# 2. 图片预处理
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("your_image.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(config.device)  # [1, 3, H, W]

# 3. 调用forward获取depth和shading
with torch.no_grad():
    depth, predicted_shading, physics_features = extractor(image_tensor)
    
# 输出：
# depth: torch.Tensor [1, 1, H, W] - 深度图
# predicted_shading: torch.Tensor [1, 1, H, W] - 着色图
# physics_features: dict - 物理特征字典

print(f"Depth shape: {depth.shape}")
print(f"Shading shape: {predicted_shading.shape}")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

# ==================== 配置 ====================
@dataclass
class Config:
    # 路径配置
    real_dir: str = "data/real"
    generated_dir: str = "data/generated"
    output_dir: str = "outputs"
    
    # 模型配置
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"
    image_size: int = 518
    vit_hidden_dim: int = 768
    
    # 物理特征配置
    depth_percentile_min: float = 1.0  # 用percentile而不是min/max
    depth_percentile_max: float = 99.0
    num_light_samples: int = 10  # 光源估计采样数
    light_estimation_method: str = "multi_sample"  # "multi_sample" or "simple"
    
    # 训练配置
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 1. 物理特征提取器（优化版）====================
class PhysicsFeatureExtractor(nn.Module):
    """提取基于物理的光照一致性特征 - 优化版"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 使用pipeline加载Depth Anything
        print("Loading Depth Anything model...")
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=config.depth_model_name,
            device=0 if config.device == "cuda" else -1
        )
        
        # Sobel算子（用于更好的梯度计算）
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """估计深度图（使用percentile归一化）"""
        B, C, H, W = image.shape
        depths = []
        
        with torch.no_grad():
            for i in range(B):
                # 反归一化到[0,1]
                img = image[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                
                # 转PIL
                img_pil = transforms.ToPILImage()(img)
                
                # 深度估计
                result = self.depth_pipe(img_pil)
                depth_map = result["depth"]
                
                # 转tensor
                depth_np = np.array(depth_map)
                depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(0)
                depth_tensor = F.interpolate(
                    depth_tensor.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # *** Percentile归一化（更robust）***
                depth_flat = depth_tensor.view(-1)
                d_min = torch.quantile(depth_flat, self.config.depth_percentile_min / 100.0)
                d_max = torch.quantile(depth_flat, self.config.depth_percentile_max / 100.0)
                depth_tensor = torch.clamp(depth_tensor, d_min, d_max)
                depth_tensor = (depth_tensor - d_min) / (d_max - d_min + 1e-8)
                
                depths.append(depth_tensor)
            
            depth = torch.stack(depths, dim=0).to(self.config.device)
        
        return depth
    
    def compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """从深度图计算表面法线（使用Sobel算子）"""
        B, C, H, W = depth.shape
        
        # 使用Sobel算子计算梯度
        depth_padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        
        # 对每个batch应用Sobel
        grad_x = F.conv2d(depth_padded, self.sobel_x.repeat(C, 1, 1, 1), groups=C)
        grad_y = F.conv2d(depth_padded, self.sobel_y.repeat(C, 1, 1, 1), groups=C)
        
        # 构造法线向量
        # scale factor控制法线的"陡峭度"
        scale = 5.0  # 可调节参数
        normals = torch.stack([
            -grad_x.squeeze(1) * scale,
            -grad_y.squeeze(1) * scale,
            torch.ones(B, H, W, device=depth.device)
        ], dim=1)
        
        # 归一化
        normals = F.normalize(normals, dim=1, eps=1e-8)
        
        return normals
    
    def estimate_light_multi_sample(
        self, 
        image: torch.Tensor,
        normals: torch.Tensor,
        num_samples: int = 5
    ) -> torch.Tensor:
        """多区域采样的光源估计（更robust）"""
        B, _, H, W = image.shape
        
        # 计算亮度
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        
        light_dirs = []
        
        for b in range(B):
            lum = luminance[b]
            norm = normals[b]
            
            # *** 多阈值采样（而不是单一阈值）***
            candidate_lights = []
            
            for percentile in np.linspace(85, 99, num_samples):
                threshold = torch.quantile(lum, percentile / 100.0)
                bright_mask = (lum > threshold)
                
                if bright_mask.sum() > 10:  # 至少10个像素
                    # 这些亮区域的平均法线方向
                    bright_normals = norm[:, bright_mask]  # [3, N]
                    avg_normal = bright_normals.mean(dim=1)  # [3]
                    
                    # 加权：使用亮度作为权重
                    weights = lum[bright_mask]
                    weighted_normals = (bright_normals * weights.unsqueeze(0)).sum(dim=1)
                    weighted_normals = weighted_normals / (weights.sum() + 1e-8)
                    
                    candidate_lights.append(weighted_normals)
            
            if len(candidate_lights) > 0:
                # 取所有候选光源的中位数方向
                candidate_lights = torch.stack(candidate_lights, dim=0)  # [num_samples, 3]
                light_dir = candidate_lights.median(dim=0)[0]
                light_dir = F.normalize(light_dir.unsqueeze(0), dim=1).squeeze(0)
            else:
                # 备选方案：使用全局最亮区域
                top_k = min(100, lum.numel() // 10)
                _, top_indices = torch.topk(lum.view(-1), top_k)
                y_coords = top_indices // W
                x_coords = top_indices % W
                
                bright_normals = norm[:, y_coords, x_coords]
                light_dir = bright_normals.mean(dim=1)
                light_dir = F.normalize(light_dir.unsqueeze(0), dim=1).squeeze(0)
            
            light_dirs.append(light_dir)
        
        return torch.stack(light_dirs, dim=0)
    
    def estimate_light_simple(
        self, 
        image: torch.Tensor,
        normals: torch.Tensor
    ) -> torch.Tensor:
        """简单的光源估计（备用方案）"""
        B, _, H, W = image.shape
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        
        light_dirs = []
        
        for b in range(B):
            lum = luminance[b]
            norm = normals[b]
            
            # 自适应阈值
            threshold = torch.quantile(lum, 0.9)
            bright_mask = (lum > threshold)
            
            if bright_mask.sum() > 0:
                bright_normals = norm[:, bright_mask]
                light_dir = bright_normals.mean(dim=1)
                light_dir = F.normalize(light_dir.unsqueeze(0), dim=1).squeeze(0)
            else:
                # 默认从斜上方照射（而不是正上方）
                light_dir = torch.tensor([0.3, -0.5, 1.0], device=image.device)
                light_dir = F.normalize(light_dir.unsqueeze(0), dim=1).squeeze(0)
            
            light_dirs.append(light_dir)
        
        return torch.stack(light_dirs, dim=0)
    
    def compute_physics_features(
        self,
        image: torch.Tensor,
        normals: torch.Tensor,
        light_dir: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """计算物理一致性特征（使用自适应阈值）"""
        B = image.shape[0]
        
        # 计算亮度
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        luminance = luminance.unsqueeze(1)
        
        # 预测的着色（Lambertian模型）
        predicted_shading = torch.clamp(
            torch.sum(
                normals * light_dir.view(B, 3, 1, 1),
                dim=1,
                keepdim=True
            ),
            min=0,
            max=1
        )
        
        features = {}
        
        # 1. 单光源拟合误差
        shading_error = torch.abs(predicted_shading - luminance)
        features['single_light_error'] = shading_error.mean(dim=[1, 2, 3])
        
        # 2. 亮度方差（AI图片通常更均匀）
        features['luminance_variance'] = luminance.var(dim=[1, 2, 3])
        
        # 3. 着色-亮度相关性
        pred_flat = predicted_shading.view(B, -1)
        lum_flat = luminance.view(B, -1)
        
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        lum_centered = lum_flat - lum_flat.mean(dim=1, keepdim=True)
        
        correlation = (pred_centered * lum_centered).mean(dim=1) / (
            pred_centered.std(dim=1) * lum_centered.std(dim=1) + 1e-8
        )
        features['shading_correlation'] = correlation
        
        # *** 4. 自适应阴影一致性 ***
        for i in range(B):
            shadow_threshold = torch.quantile(luminance[i], 0.3)  # 最暗30%
            shadow_pred_threshold = torch.quantile(predicted_shading[i], 0.3)
        
        shadow_pred = (predicted_shading < 0.3).float()
        shadow_actual = (luminance < 0.3).float()
        shadow_iou = (shadow_pred * shadow_actual).sum(dim=[1,2,3]) / (
            (shadow_pred + shadow_actual).clamp(min=1).sum(dim=[1,2,3]) + 1e-8
        )
        features['shadow_consistency'] = shadow_iou
        
        # *** 5. 自适应高光一致性 ***
        highlight_pred = (predicted_shading > 0.7).float()
        highlight_actual = (luminance > 0.7).float()
        highlight_iou = (highlight_pred * highlight_actual).sum(dim=[1,2,3]) / (
            (highlight_pred + highlight_actual).clamp(min=1).sum(dim=[1,2,3]) + 1e-8
        )
        features['highlight_consistency'] = highlight_iou
        
        return features, predicted_shading
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """完整的物理特征提取流程"""
        # 1. 估计深度
        depth = self.estimate_depth(image)
        
        # 2. 计算法线
        normals = self.compute_normals(depth)
        
        # 3. 估计光源（使用配置的方法）
        if self.config.light_estimation_method == "multi_sample":
            light_dir = self.estimate_light_multi_sample(
                image, normals, self.config.num_light_samples
            )
        else:
            light_dir = self.estimate_light_simple(image, normals)
        
        # 4. 计算物理特征
        physics_features, predicted_shading = self.compute_physics_features(
            image, normals, light_dir
        )
        
        return depth, predicted_shading, physics_features


# ==================== 2. ViT分类器（含转换层)===============
class PhysicsAwareViT(nn.Module):

    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 转换层：5通道 → 3通道
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3)
        )
        
        # 加载标准的预训练ViT
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 物理特征融合层
        num_physics_features = 5
        self.physics_fusion = nn.Sequential(
            nn.Linear(num_physics_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256)
        )
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.vit_hidden_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        predicted_shading: torch.Tensor,
        physics_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = image.shape[0]
        
        # 调整所有输入到224x224
        resize = transforms.Resize((224, 224), antialias=True)
        image_resized = resize(image)
        depth_resized = resize(depth)
        shading_resized = resize(predicted_shading)
        
        # 拼接5通道输入
        five_channel_input = torch.cat([
            image_resized,
            depth_resized,
            shading_resized
        ], dim=1)
        
        # 转换为3通道
        adapted_input = self.channel_adapter(five_channel_input)
        
        # ViT编码
        vit_outputs = self.vit(pixel_values=adapted_input)
        visual_embedding = vit_outputs.last_hidden_state[:, 0]
        
        # 物理特征融合
        physics_feature_vector = torch.stack([
            physics_features['single_light_error'],
            physics_features['luminance_variance'],
            physics_features['shading_correlation'],
            physics_features['shadow_consistency'],
            physics_features['highlight_consistency']
        ], dim=1)
        
        physics_embedding = self.physics_fusion(physics_feature_vector)
        
        # 融合所有特征
        combined_embedding = torch.cat([
            visual_embedding,
            physics_embedding
        ], dim=1)
        
        # 分类
        logits = self.classifier(combined_embedding)
        
        return logits, combined_embedding


# ==================== 3. 完整分类器 ====================
class RealVsAIClassifier(nn.Module):
    """完整的真实/AI生成图片分类器"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.physics_extractor = PhysicsFeatureExtractor(config)
        self.classifier = PhysicsAwareViT(config)
        
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 提取物理特征
        depth, predicted_shading, physics_features = self.physics_extractor(image)
        
        # 分类
        logits, embedding = self.classifier(
            image, depth, predicted_shading, physics_features
        )
        
        probs = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probability': probs,
            'prediction': (probs > 0.5).long(),
            'depth': depth,
            'predicted_shading': predicted_shading,
            'embedding': embedding,
            'physics_features': physics_features
        }


# ==================== 4. 数据集 ====================
class PairedImageDataset(torch.utils.data.Dataset):
    """成对的真实/生成图片数据集"""
    
    def __init__(self, real_dir: str, generated_dir: str, transform=None, mode='train'):
        self.real_dir = Path(real_dir)
        self.generated_dir = Path(generated_dir)
        self.transform = transform
        self.mode = mode
        
        self.pairs = self._find_pairs()
        print(f"Found {len(self.pairs)} image pairs")
        
    def _find_pairs(self):
        """找到所有配对的图片"""
        pairs = []
        
        for real_path in sorted(self.real_dir.glob("*.jpg")) + sorted(self.real_dir.glob("*.png")):
            filename = real_path.stem
            if '_real' in filename:
                img_id = filename.replace('_real', '').replace('img_', '')
            else:
                continue
            
            gen_candidates = list(self.generated_dir.glob(f"*{img_id}*.jpg")) + \
                           list(self.generated_dir.glob(f"*{img_id}*.png"))
            
            if gen_candidates:
                for gen_path in gen_candidates:
                    if 'real' not in gen_path.stem:
                        pairs.append({
                            'id': img_id,
                            'real_path': real_path,
                            'generated_path': gen_path
                        })
        
        return pairs
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.pairs) * 2
        else:
            return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            pair_idx = idx // 2
            is_real = (idx % 2 == 0)
            pair = self.pairs[pair_idx]
            
            if is_real:
                image_path = pair['real_path']
                label = 0
            else:
                image_path = pair['generated_path']
                label = 1
            
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return {'image': image, 'label': label, 'pair_id': pair['id']}
        else:
            pair = self.pairs[idx]
            real_image = Image.open(pair['real_path']).convert('RGB')
            gen_image = Image.open(pair['generated_path']).convert('RGB')
            
            if self.transform:
                real_image = self.transform(real_image)
                gen_image = self.transform(gen_image)
            
            return {
                'real_image': real_image,
                'generated_image': gen_image,
                'pair_id': pair['id']
            }


# ==================== 5. 训练器 ====================
class Trainer:
    """训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.model = RealVsAIClassifier(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dpo_loss(
        self,
        real_logits: torch.Tensor,
        gen_logits: torch.Tensor,
        beta: float = 0.1
    ) -> torch.Tensor:
        """DPO损失：偏好生成图片得分高于真实图片"""
        diff = beta * (gen_logits - real_logits)
        loss = -F.logsigmoid(diff).mean()
        return loss
    
    def train_epoch(self, dataloader, use_dpo=True):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            if use_dpo:
                real_images = batch['real_image'].to(self.device)
                gen_images = batch['generated_image'].to(self.device)
                
                real_outputs = self.model(real_images)
                gen_outputs = self.model(gen_images)
                
                loss = self.dpo_loss(real_outputs['logits'], gen_outputs['logits'])
                
                real_correct = (real_outputs['prediction'] == 0).sum()
                gen_correct = (gen_outputs['prediction'] == 1).sum()
                correct += (real_correct + gen_correct).item()
                total += real_images.size(0) * 2
            else:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device).float().unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.bce_loss(outputs['logits'], labels)
                
                correct += (outputs['prediction'] == labels.long()).sum().item()
                total += images.size(0)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}' if total > 0 else '0'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / total if total > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")


# ==================== 6. 主函数 ====================
def main():
    config = Config(
        real_dir="data/real",
        generated_dir="data/generated",
        output_dir="outputs",
        batch_size=8,
        num_epochs=2,
        light_estimation_method="multi_sample",  # 使用多采样光源估计
        num_light_samples=10
    )
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 创建数据集
    dpo_dataset = PairedImageDataset(
        config.real_dir,
        config.generated_dir,
        transform=transform,
        mode='pair'
    )
    standard_dataset = PairedImageDataset(
        config.real_dir,
        config.generated_dir,
        transform=transform,
        mode='train'
    )
    
    # 数据加载器
    dpo_loader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    standard_loader = torch.utils.data.DataLoader(
        standard_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    print("Starting training...")
    Path(config.output_dir).mkdir(exist_ok=True, parents=True)
    
    for epoch in range(config.num_epochs):
        # 前半程用DPO，后半程用BCE
        if epoch < config.num_epochs // 2:
            train_loss, train_acc = trainer.train_epoch(dpo_loader, use_dpo=True)
            mode = "DPO"
        else:
            train_loss, train_acc = trainer.train_epoch(standard_loader, use_dpo=False)
            mode = "BCE"
        
        print(f"Epoch {epoch+1}/{config.num_epochs} [{mode}] "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        trainer.scheduler.step()
        
        # 每5个epoch保存一次
        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(
                f"{config.output_dir}/checkpoint_epoch_{epoch+1}.pt"
            )
    
    # 保存最终模型
    trainer.save_checkpoint(f"{config.output_dir}/final_model.pt")
    print("Training completed!")


if __name__ == "__main__":
    main()