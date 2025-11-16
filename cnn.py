import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import re
from tqdm import tqdm
from ViT_train import PhysicsFeatureExtractor, Config


# ==================== 全局初始化ViT特征提取器 ====================
print("初始化深度和着色提取器...")
config = Config()
extractor = PhysicsFeatureExtractor(config).to(config.device)
extractor.eval()

# 用于ViT的transform
vit_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== 数据集类 ====================
class PairedImageDataset(Dataset):
    def __init__(self, generated_dir, real_dir, transform=None, image_size=224):
        """
        Args:
            generated_dir: AI生成图像文件夹路径
            real_dir: 真实图像文件夹路径
            transform: 数据增强
            image_size: 统一的图像大小
        """
        self.generated_dir = Path(generated_dir)
        self.real_dir = Path(real_dir)
        self.transform = transform
        self.image_size = image_size
        
        # 获取所有图像对
        self.image_pairs = self._match_image_pairs()
        print(f"找到 {len(self.image_pairs)} 对图像")
        
    def _match_image_pairs(self):
        """匹配生成图像和真实图像对"""
        pairs = []
        
        # 获取生成图像列表
        gen_images = sorted(list(self.generated_dir.glob("*.jpg")) + 
                          list(self.generated_dir.glob("*.png")))
        real_images = sorted(list(self.real_dir.glob("*.jpg")) + 
                           list(self.real_dir.glob("*.png")))
        
        # 简单的配对策略：假设按顺序配对或通过数字编号配对
        for i, (gen_img, real_img) in enumerate(zip(gen_images, real_images)):
            pairs.append({
                'generated': gen_img,
                'real': real_img,
                'label': 0 if np.random.rand() > 0.5 else 1
            })
            
        return pairs
    
    def __len__(self):
        return len(self.image_pairs) * 2  # 每对图像产生2个样本
    
    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_generated = idx % 2 == 0  # 偶数索引为生成图像，奇数为真实图像
        
        pair = self.image_pairs[pair_idx]
        img_path = pair['generated'] if is_generated else pair['real']
        label = 1 if is_generated else 0  # 1=AI生成, 0=真实
        
        # 加载图像
        image_pil = Image.open(str(img_path)).convert('RGB')
        
        # 生成6通道特征
        six_channel = self._generate_six_channels(image_pil)
        
        if self.transform:
            six_channel = self.transform(six_channel)
        else:
            six_channel = torch.from_numpy(six_channel).float().permute(2, 0, 1)
        
        return six_channel, label
    
    def _generate_six_channels(self, image_pil):
        """
        生成6通道特征图（使用ViT提取深度和着色）
        Args:
            image_pil: PIL Image对象
        Returns:
            six_channel: numpy array (H, W, 6)
        """
        # 1. 转换为numpy array用于RGB通道
        rgb = np.array(image_pil.resize((self.image_size, self.image_size)))
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # 2. 使用ViT提取器获取深度图和着色图
        with torch.no_grad():
            # 准备输入
            image_tensor = vit_transform(image_pil).unsqueeze(0).to(config.device)
            
            # 调用forward获取depth和shading
            depth_tensor, predicted_shading_tensor, physics_features = extractor(image_tensor)
            
            # 转换为numpy并调整大小
            depth = depth_tensor.squeeze().cpu().numpy()  # [H, W]
            predicted_shading = predicted_shading_tensor.squeeze().cpu().numpy()  # [H, W]
            
            # 如果尺寸不匹配，resize到目标大小
            if depth.shape != (self.image_size, self.image_size):
                depth = cv2.resize(depth, (self.image_size, self.image_size))
            if predicted_shading.shape != (self.image_size, self.image_size):
                predicted_shading = cv2.resize(predicted_shading, (self.image_size, self.image_size))
        
        # 3. 计算残差图
        residual = self._compute_residual(rgb, predicted_shading)
        
        # 4. 拼接所有通道
        six_channel = np.concatenate([
            rgb_norm,                          # (H, W, 3)
            depth[..., np.newaxis],           # (H, W, 1)
            predicted_shading[..., np.newaxis],    # (H, W, 1)
            residual[..., np.newaxis]         # (H, W, 1)
        ], axis=2)
        
        return six_channel
    
    def _compute_residual(self, rgb, pred_shading):
        """
        计算残差图：实际亮度 vs 预测亮度
        Args:
            rgb: numpy array (H, W, 3), 范围[0, 1]
            pred_shading: numpy array (H, W), 范围[0, 1]
        """
        # 实际亮度 (归一化的灰度图)
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # 残差
        residual = np.abs(gray - pred_shading)
        
        # 归一化
        residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        
        return residual


# ==================== 修改的ResNet18模型 ====================
class SixChannelResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SixChannelResNet18, self).__init__()
        
        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 修改第一层卷积以接受6通道输入
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            6,  # 输入通道改为6
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # 初始化新卷积层的权重
        with torch.no_grad():
            # RGB通道使用预训练权重
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            # 额外3个通道随机初始化
            self.conv1.weight[:, 3:, :, :] = torch.randn(64, 3, 7, 7) * 0.01
        
        # 保留ResNet的其余部分
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # 修改全连接层
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.4f}')
        
        scheduler.step()
    
    print(f'\nBest Validation Accuracy: {best_acc:.4f}')
    return model


# ==================== 主函数 ====================
if __name__ == '__main__':
    # 设置路径
    generated_dir = r'C:\Users\zimol\Desktop\Fake-Image-Recognition-main\data_SD_50test\generated_image'
    real_dir = r'C:\Users\zimol\Desktop\Fake-Image-Recognition-main\data_SD_50test\real_images'
    
    # 检查路径
    if not os.path.exists(generated_dir) or not os.path.exists(real_dir):
        print("错误：请确认数据集路径是否正确")
        exit(1)
    
    # 创建数据集
    dataset = PairedImageDataset(
        generated_dir=generated_dir,
        real_dir=real_dir,
        image_size=224
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0  # Windows上建议设为0避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    model = SixChannelResNet18(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # 训练模型
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=20, 
        device=device
    )
    
    print("\n训练完成！模型已保存为 'best_model.pth'")