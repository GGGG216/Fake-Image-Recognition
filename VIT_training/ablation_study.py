"""
Model Comparison Benchmark - 修复标签检测（增强版）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTModel
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ⭐ 关键：从训练脚本导入Config和模型类
from ViT_train import RealVsAIClassifier, Config

# ==================== 配置 ====================
@dataclass
class BenchmarkConfig:
    """评测配置"""
    our_test_dir: str = "test"
    output_dir: str = "benchmark_results"
    our_model_path: str = "outputs/final_model.pt"
    batch_size: int = 16
    image_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # HuggingFace数据集配置
    use_hf_datasets: bool = True
    cifake_sample_size: int = 2000
    fake_or_real_sample_size: int = 1000


# ==================== 辅助函数 ====================
def detect_label_format_smart(dataset, dataset_name: str):
    """
    智能检测标签格式 - 改进版
    会扫描更多样本直到找到两种标签
    """
    print(f"\n{'='*60}")
    print(f"Smart Label Detection for {dataset_name}")
    print(f"{'='*60}")
    
    if len(dataset) == 0:
        print("⚠️  Dataset is empty!")
        return None, None, None, None
    
    # 检查第一个样本结构
    first_item = dataset[0]
    
    if not isinstance(first_item, dict):
        print(f"⚠️  First item is not a dictionary: {type(first_item)}")
        return None, None, None, None
    
    print(f"Available keys: {list(first_item.keys())}")
    
    # 1. 找到标签字段
    possible_label_keys = ['label', 'labels', 'class', 'category', 'target', 'gt']
    label_key = None
    
    for key in possible_label_keys:
        if key in first_item:
            label_key = key
            print(f"✓ Found label key: '{label_key}'")
            break
    
    if label_key is None:
        # 尝试使用第一个非图片字段
        for key, value in first_item.items():
            if not isinstance(value, Image.Image):
                label_key = key
                print(f"  Using '{label_key}' as label key")
                break
    
    if label_key is None:
        print("❌ Could not find label key")
        return None, None, None, None
    
    # 2. 找到图片字段
    possible_image_keys = ['image', 'img', 'picture', 'photo', 'pixel_values']
    image_key = None
    
    for key in possible_image_keys:
        if key in first_item:
            if isinstance(first_item[key], Image.Image):
                image_key = key
                print(f"✓ Found image key: '{image_key}'")
                break
    
    if image_key is None:
        print("❌ Could not find image key")
        return None, None, None, None
    
    # 3. 智能采样标签值（确保找到两种标签）
    print(f"\nScanning dataset for unique labels...")
    
    label_values = []
    unique_labels = set()
    max_scan = min(10000, len(dataset))  # 最多扫描10000个样本
    scan_step = max(1, len(dataset) // 1000)  # 采样步长
    
    # 使用采样策略：从头、中、尾均匀采样
    sample_indices = []
    
    # 从开头采样
    sample_indices.extend(range(0, min(500, len(dataset))))
    
    # 从中间采样
    mid = len(dataset) // 2
    sample_indices.extend(range(mid - 250, min(mid + 250, len(dataset))))
    
    # 从末尾采样
    sample_indices.extend(range(max(0, len(dataset) - 500), len(dataset)))
    
    # 随机采样
    if len(dataset) > 1000:
        random_indices = np.random.choice(
            len(dataset), 
            min(1000, len(dataset)), 
            replace=False
        )
        sample_indices.extend(random_indices)
    
    sample_indices = list(set(sample_indices))  # 去重
    
    print(f"Sampling {len(sample_indices)} items from dataset...")
    
    for idx in tqdm(sample_indices[:5000], desc="Scanning labels"):
        try:
            item = dataset[int(idx)]
            if isinstance(item, dict) and label_key in item:
                label_val = item[label_key]
                label_values.append(label_val)
                unique_labels.add(label_val)
                
                # 如果已经找到2种标签，提前退出
                if len(unique_labels) >= 2:
                    break
        except Exception as e:
            continue
    
    unique_labels = sorted(list(unique_labels))
    print(f"\n✓ Found {len(unique_labels)} unique labels: {unique_labels}")
    
    if len(unique_labels) == 0:
        print("❌ No labels found")
        return None, None, None, None
    
    # 统计完整分布
    label_counts = {label: label_values.count(label) for label in unique_labels}
    print(f"Label distribution in sample: {label_counts}")
    
    # 4. 确定真假标签值
    real_value = None
    fake_value = None
    
    if len(unique_labels) == 1:
        # 只有一种标签 - 假设是真实图片，假图片标签为1
        if unique_labels[0] == 0:
            real_value = 0
            fake_value = 1
            print(f"⚠️  Only found one label type: {unique_labels[0]}")
            print(f"   Assuming: Real={real_value}, Fake={fake_value}")
        else:
            real_value = unique_labels[0]
            fake_value = 0 if unique_labels[0] != 0 else 1
    
    elif len(unique_labels) == 2:
        label_0, label_1 = unique_labels[0], unique_labels[1]
        
        # 常见的真实标签
        real_indicators = ['REAL', 'real', 'Real', 'true', 'True', 0, '0', 'genuine']
        # 常见的假标签
        fake_indicators = ['FAKE', 'fake', 'Fake', 'false', 'False', 1, '1', 'generated', 'synthetic']
        
        # 检查标签0
        if label_0 in real_indicators or str(label_0).upper() == 'REAL':
            real_value = label_0
            fake_value = label_1
        elif label_1 in real_indicators or str(label_1).upper() == 'REAL':
            real_value = label_1
            fake_value = label_0
        elif label_0 in fake_indicators or str(label_0).upper() == 'FAKE':
            fake_value = label_0
            real_value = label_1
        elif label_1 in fake_indicators or str(label_1).upper() == 'FAKE':
            fake_value = label_1
            real_value = label_0
        # 默认：0是真，1是假
        elif 0 in unique_labels and 1 in unique_labels:
            real_value = 0
            fake_value = 1
        else:
            # 使用第一个作为真，第二个作为假
            real_value = label_0
            fake_value = label_1
    
    else:
        # 多分类情况
        print(f"⚠️  Found {len(unique_labels)} classes - using binary mapping")
        real_value = unique_labels[0]
        fake_value = unique_labels[1]
    
    print(f"\n✓ Final mapping:")
    print(f"  Label key: '{label_key}'")
    print(f"  Image key: '{image_key}'")
    print(f"  Real value: {real_value}")
    print(f"  Fake value: {fake_value}")
    
    return label_key, image_key, real_value, fake_value


# ==================== 数据集加载器 ====================
class LocalTestDataset(Dataset):
    """本地测试数据集"""
    
    def __init__(self, test_dir: str, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        if not self.test_dir.exists():
            print(f"⚠️  Test directory not found: {self.test_dir}")
            return samples
        
        real_dir = self.test_dir / "real"
        gen_dir = self.test_dir / "generated"
        
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in sorted(real_dir.glob(ext)):
                    samples.append({'path': img_path, 'label': 0, 'type': 'real'})
        
        if gen_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in sorted(gen_dir.glob(ext)):
                    samples.append({'path': img_path, 'label': 1, 'type': 'generated'})
        
        if samples:
            print(f"✓ Loaded {len(samples)} local samples")
            print(f"  Real: {sum(1 for s in samples if s['label'] == 0)}")
            print(f"  Generated: {sum(1 for s in samples if s['label'] == 1)}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label'],
            'path': str(sample['path'])
        }


class CIFAKEDataset(Dataset):
    """CIFAKE数据集 - 智能标签检测"""
    
    def __init__(self, sample_size: int = 2000, transform=None, split: str = "train"):
        self.transform = transform
        self.samples = []
        
        try:
            from datasets import load_dataset
            print(f"\nLoading CIFAKE dataset (split={split})...")
            
            dataset = load_dataset(
                "dragonintelligence/CIFAKE-image-dataset",
                split=split,
                trust_remote_code=True
            )
            
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # ⭐ 使用智能检测
            label_key, image_key, real_value, fake_value = detect_label_format_smart(
                dataset, "CIFAKE"
            )
            
            if label_key is None or image_key is None:
                print("❌ Could not detect dataset format")
                return
            
            # 统计完整数据集的真假分布
            print(f"\nCounting full dataset distribution...")
            real_count = 0
            fake_count = 0
            
            # 采样统计（避免扫描全部数据）
            sample_for_count = min(5000, len(dataset))
            step = max(1, len(dataset) // sample_for_count)
            
            for i in tqdm(range(0, len(dataset), step), desc="Counting labels"):
                try:
                    item = dataset[i]
                    if item[label_key] == real_value:
                        real_count += 1
                    elif item[label_key] == fake_value:
                        fake_count += 1
                except:
                    continue
            
            # 按比例估算总数
            total_sampled = real_count + fake_count
            if total_sampled > 0:
                real_ratio = real_count / total_sampled
                fake_ratio = fake_count / total_sampled
                estimated_real = int(len(dataset) * real_ratio)
                estimated_fake = int(len(dataset) * fake_ratio)
                print(f"  Estimated Real: ~{estimated_real}, Fake: ~{estimated_fake}")
            
            # 随机采样（确保平衡）
            print(f"\nSampling {sample_size} images...")
            
            real_indices = []
            fake_indices = []
            
            # 扫描数据集收集索引
            for i in tqdm(range(len(dataset)), desc="Collecting indices"):
                try:
                    item = dataset[i]
                    if item[label_key] == real_value:
                        real_indices.append(i)
                    elif item[label_key] == fake_value:
                        fake_indices.append(i)
                    
                    # 提前退出条件：收集足够的样本
                    if len(real_indices) >= sample_size and len(fake_indices) >= sample_size:
                        break
                except:
                    continue
            
            print(f"✓ Found {len(real_indices)} real and {len(fake_indices)} fake images")
            
            # 平衡采样
            real_sample_size = min(sample_size // 2, len(real_indices))
            fake_sample_size = min(sample_size // 2, len(fake_indices))
            
            if real_sample_size > 0:
                selected_real = np.random.choice(real_indices, real_sample_size, replace=False)
            else:
                selected_real = []
            
            if fake_sample_size > 0:
                selected_fake = np.random.choice(fake_indices, fake_sample_size, replace=False)
            else:
                selected_fake = []
            
            indices = np.concatenate([selected_real, selected_fake])
            np.random.shuffle(indices)
            
            print(f"Loading {len(indices)} samples...")
            for idx in tqdm(indices, desc="Loading images"):
                item = dataset[int(idx)]
                label = 0 if item[label_key] == real_value else 1
                self.samples.append({
                    'image': item[image_key],
                    'label': label,
                })
            
            real_in_sample = sum(1 for s in self.samples if s['label'] == 0)
            fake_in_sample = sum(1 for s in self.samples if s['label'] == 1)
            
            print(f"\n✓ Successfully loaded {len(self.samples)} CIFAKE images")
            print(f"  Real: {real_in_sample}, Fake: {fake_in_sample}")
            
        except Exception as e:
            print(f"⚠️  Failed to load CIFAKE: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        
        if not isinstance(image, Image.Image):
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label']
        }


class FakeOrRealDataset(Dataset):
    """Fake or Real竞赛数据集 - 智能标签检测"""
    
    def __init__(self, sample_size: int = 1000, transform=None, split: str = "train"):
        self.transform = transform
        self.samples = []
        
        try:
            from datasets import load_dataset
            print(f"\nLoading Fake or Real dataset (split={split})...")
            
            dataset = load_dataset(
                "mncai/Fake_or_Real_Competition_Dataset",
                split=split,
                trust_remote_code=True
            )
            
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            # ⭐ 使用智能检测
            label_key, image_key, real_value, fake_value = detect_label_format_smart(
                dataset, "Fake_or_Real"
            )
            
            if label_key is None or image_key is None:
                print("❌ Could not detect dataset format")
                return
            
            # 收集索引
            print(f"\nCollecting indices...")
            real_indices = []
            fake_indices = []
            
            for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
                try:
                    item = dataset[i]
                    if item[label_key] == real_value:
                        real_indices.append(i)
                    elif item[label_key] == fake_value:
                        fake_indices.append(i)
                    
                    if len(real_indices) >= sample_size and len(fake_indices) >= sample_size:
                        break
                except:
                    continue
            
            print(f"✓ Found {len(real_indices)} real and {len(fake_indices)} fake images")
            
            # 平衡采样
            real_sample_size = min(sample_size // 2, len(real_indices))
            fake_sample_size = min(sample_size // 2, len(fake_indices))
            
            if real_sample_size > 0:
                selected_real = np.random.choice(real_indices, real_sample_size, replace=False)
            else:
                selected_real = []
            
            if fake_sample_size > 0:
                selected_fake = np.random.choice(fake_indices, fake_sample_size, replace=False)
            else:
                selected_fake = []
            
            indices = np.concatenate([selected_real, selected_fake])
            np.random.shuffle(indices)
            
            print(f"Loading {len(indices)} samples...")
            for idx in tqdm(indices, desc="Loading images"):
                item = dataset[int(idx)]
                label = 0 if item[label_key] == real_value else 1
                self.samples.append({
                    'image': item[image_key],
                    'label': label,
                })
            
            real_in_sample = sum(1 for s in self.samples if s['label'] == 0)
            fake_in_sample = sum(1 for s in self.samples if s['label'] == 1)
            
            print(f"\n✓ Successfully loaded {len(self.samples)} Fake or Real images")
            print(f"  Real: {real_in_sample}, Fake: {fake_in_sample}")
            
        except Exception as e:
            print(f"⚠️  Failed to load Fake or Real: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        
        if not isinstance(image, Image.Image):
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['label']
        }


# ==================== 基线模型 ====================
class BaselineResNet(nn.Module):
    """ResNet50基线"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.resnet(image)
        probs = torch.sigmoid(logits)
        return {
            'logits': logits,
            'probability': probs,
            'prediction': (probs > 0.5).long()
        }


class VanillaViT(nn.Module):
    """原始ViT（无物理特征）"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            from transformers import ViTConfig
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.vit(pixel_values=image)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        probs = torch.sigmoid(logits)
        return {
            'logits': logits,
            'probability': probs,
            'prediction': (probs > 0.5).long()
        }


class EfficientNetBaseline(nn.Module):
    """EfficientNet-B0基线"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.efficientnet(image)
        probs = torch.sigmoid(logits)
        return {
            'logits': logits,
            'probability': probs,
            'prediction': (probs > 0.5).long()
        }


# ==================== 评测器（保持不变）====================
class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = config.device
        self.results = {}
    
    def load_our_model(self) -> Optional[RealVsAIClassifier]:
        """加载我们的模型"""
        try:
            print("Loading our Physics-Aware ViT model...")
            
            if not Path(self.config.our_model_path).exists():
                print(f"⚠️  Model checkpoint not found: {self.config.our_model_path}")
                return None
            
            try:
                checkpoint = torch.load(
                    self.config.our_model_path,
                    map_location=self.device,
                    weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(
                    self.config.our_model_path,
                    map_location=self.device
                )
            
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
                print("⚠️  No config in checkpoint, using default")
                model_config = Config()
            
            model = RealVsAIClassifier(model_config).to(self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            print("✓ Our model loaded successfully")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading our model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_baseline_models(self) -> Dict[str, nn.Module]:
        """加载基线模型"""
        models_dict = {}
        
        print("\nLoading baseline models...")
        
        try:
            print("  Loading ResNet50...")
            models_dict['ResNet50'] = BaselineResNet(pretrained=True).to(self.device)
            models_dict['ResNet50'].eval()
            print("  ✓ ResNet50 loaded")
        except Exception as e:
            print(f"  ⚠️  ResNet50 failed: {e}")
        
        try:
            print("  Loading Vanilla ViT...")
            models_dict['Vanilla_ViT'] = VanillaViT(pretrained=True).to(self.device)
            models_dict['Vanilla_ViT'].eval()
            print("  ✓ Vanilla ViT loaded")
        except Exception as e:
            print(f"  ⚠️  Vanilla ViT failed: {e}")
        
        try:
            print("  Loading EfficientNet-B0...")
            models_dict['EfficientNet_B0'] = EfficientNetBaseline(pretrained=True).to(self.device)
            models_dict['EfficientNet_B0'].eval()
            print("  ✓ EfficientNet-B0 loaded")
        except Exception as e:
            print(f"  ⚠️  EfficientNet-B0 failed: {e}")
        
        return models_dict
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """创建所有数据加载器"""
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        dataloaders = {}
        
        print("\n" + "="*80)
        print("Loading Datasets")
        print("="*80)
        
        # 1. 本地测试集
        local_dataset = LocalTestDataset(self.config.our_test_dir, transform=transform)
        if len(local_dataset) > 0:
            dataloaders['Local_Test'] = DataLoader(
                local_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        if self.config.use_hf_datasets:
            # 2. CIFAKE数据集
            cifake_dataset = CIFAKEDataset(
                sample_size=self.config.cifake_sample_size,
                transform=transform,
                split="train"
            )
            
            if len(cifake_dataset) > 0:
                dataloaders['CIFAKE'] = DataLoader(
                    cifake_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                )
            
            # 3. Fake or Real竞赛数据集
            fake_real_dataset = FakeOrRealDataset(
                sample_size=self.config.fake_or_real_sample_size,
                transform=transform,
                split="train"
            )
            
            if len(fake_real_dataset) > 0:
                dataloaders['Fake_or_Real'] = DataLoader(
                    fake_real_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                )
        
        if not dataloaders:
            print("\n❌ No datasets loaded!")
        
        return dataloaders
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """评测单个模型"""
        print(f"\nEvaluating {model_name}...")
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Testing {model_name}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                try:
                    outputs = model(images)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs['prediction'].squeeze().cpu().numpy())
                    all_probs.extend(outputs['probability'].squeeze().cpu().numpy())
                except Exception as e:
                    print(f"  Error in batch: {e}")
                    continue
        
        if len(all_labels) == 0:
            return None
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }
        
        print(f"  Acc: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return metrics
    
    def run_benchmark(self):
        """运行完整评测"""
        print("="*80)
        print("Multi-Dataset Model Benchmark")
        print("="*80)
        
        dataloaders = self.create_dataloaders()
        
        if not dataloaders:
            return {}
        
        models_dict = {}
        
        our_model = self.load_our_model()
        if our_model is not None:
            models_dict['Our_Physics_ViT'] = our_model
        
        baseline_models = self.load_baseline_models()
        models_dict.update(baseline_models)
        
        if not models_dict:
            print("\n❌ No models loaded!")
            return {}
        
        print(f"\n{'='*80}")
        print(f"Running Evaluation on {len(dataloaders)} datasets")
        print(f"{'='*80}")
        
        results = {}
        
        for dataset_name, dataloader in dataloaders.items():
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            results[dataset_name] = {}
            
            for model_name, model in models_dict.items():
                try:
                    metrics = self.evaluate_model(model, dataloader, model_name)
                    if metrics:
                        results[dataset_name][model_name] = metrics
                except Exception as e:
                    print(f"❌ Error evaluating {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        self.results = results
        
        if results:
            self.save_results()
            self.visualize_results()
        
        return results
    
    def save_results(self):
        """保存结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        json_path = output_dir / "results_full.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Full results saved to {json_path}")
        
        for dataset_name, dataset_results in self.results.items():
            if dataset_results:
                rows = []
                for model_name, metrics in dataset_results.items():
                    row = {'Model': model_name, **metrics}
                    rows.append(row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    csv_path = output_dir / f"results_{dataset_name}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"✓ {dataset_name} results saved to {csv_path}")
        
        summary_rows = []
        for dataset_name, dataset_results in self.results.items():
            for model_name, metrics in dataset_results.items():
                row = {
                    'Dataset': dataset_name,
                    'Model': model_name,
                    **metrics
                }
                summary_rows.append(row)
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            summary_path = output_dir / "results_summary.csv"
            df_summary.to_csv(summary_path, index=False)
            print(f"✓ Summary saved to {summary_path}")
    
    def visualize_results(self):
        """可视化结果"""
        output_dir = Path(self.config.output_dir)
        
        if not self.results:
            return
        
        for dataset_name, dataset_results in self.results.items():
            if not dataset_results:
                continue
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            
            data = []
            for model_name, result in dataset_results.items():
                for metric in metrics:
                    if metric in result:
                        data.append({
                            'Model': model_name,
                            'Metric': metric.capitalize(),
                            'Value': result[metric]
                        })
            
            if data:
                df = pd.DataFrame(data)
                pivot = df.pivot(index='Model', columns='Metric', values='Value')
                
                plt.figure(figsize=(12, 6))
                pivot.plot(kind='bar', figsize=(12, 6))
                plt.title(f'Performance on {dataset_name}', fontsize=14, fontweight='bold')
                plt.xlabel('Model', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.1)
                plt.tight_layout()
                
                plot_path = output_dir / f'comparison_{dataset_name}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ {dataset_name} plot saved to {plot_path}")
        
        all_models = set()
        all_datasets = list(self.results.keys())
        
        for dataset_results in self.results.values():
            all_models.update(dataset_results.keys())
        
        all_models = sorted(list(all_models))
        
        for metric in ['accuracy', 'f1', 'auc']:
            heatmap_data = []
            
            for model_name in all_models:
                row = []
                for dataset_name in all_datasets:
                    if model_name in self.results[dataset_name]:
                        value = self.results[dataset_name][model_name].get(metric, 0)
                    else:
                        value = 0
                    row.append(value)
                heatmap_data.append(row)
            
            df_heatmap = pd.DataFrame(
                heatmap_data,
                index=all_models,
                columns=all_datasets
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='YlGnBu',
                       cbar_kws={'label': metric.capitalize()}, vmin=0, vmax=1)
            plt.title(f'{metric.capitalize()} Across Datasets', fontsize=14, fontweight='bold')
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Model', fontsize=12)
            plt.tight_layout()
            
            heatmap_path = output_dir / f'heatmap_{metric}.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {metric} heatmap saved to {heatmap_path}")
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to display")
            return
        
        for dataset_name, dataset_results in self.results.items():
            print(f"\n{dataset_name}:")
            print("-" * 60)
            
            best_acc = 0
            best_model = None
            
            for model_name, metrics in dataset_results.items():
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)
                auc = metrics.get('auc', 0)
                print(f"  {model_name:25s}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name
            
            if best_model:
                print(f"  ★ Best: {best_model} (Accuracy: {best_acc:.4f})")


# ==================== 主函数 ====================
def main():
    config = BenchmarkConfig(
        our_test_dir="data/test",
        our_model_path="outputs/final_model.pt",
        output_dir="benchmark_results",
        batch_size=16,
        use_hf_datasets=True,
        cifake_sample_size=2000,
        fake_or_real_sample_size=1000,
    )
    
    evaluator = ModelEvaluator(config)
    results = evaluator.run_benchmark()
    
    if results:
        evaluator.print_summary()
        print(f"\n✓ Benchmark completed! Results in {config.output_dir}/")
    else:
        print("\n❌ Benchmark failed")


if __name__ == "__main__":
    main()