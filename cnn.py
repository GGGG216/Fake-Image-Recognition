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
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ViT_train import PhysicsFeatureExtractor, Config


# ==================== Preprocessing Function ====================
def preprocess_dataset(generated_dir, real_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    config = Config()
    extractor = PhysicsFeatureExtractor(config).to(config.device)
    extractor.eval()
    
    vit_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def process_single_image(img_path, label, save_name):
        image_pil = Image.open(str(img_path)).convert('RGB')
        rgb = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0
        
        with torch.no_grad():
            image_tensor = vit_transform(image_pil).unsqueeze(0).to(config.device)
            depth_tensor, predicted_shading_tensor, _ = extractor(image_tensor)
            
            depth = depth_tensor.squeeze().cpu().numpy()
            predicted_shading = predicted_shading_tensor.squeeze().cpu().numpy()
            
            if depth.shape != (224, 224):
                depth = cv2.resize(depth, (224, 224))
            if predicted_shading.shape != (224, 224):
                predicted_shading = cv2.resize(predicted_shading, (224, 224))
        
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        residual = np.abs(gray - predicted_shading)
        residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
        
        six_channel = np.concatenate([
            rgb,
            depth[..., np.newaxis],
            predicted_shading[..., np.newaxis],
            residual[..., np.newaxis]
        ], axis=2).astype(np.float32)
        
        save_path = os.path.join(output_dir, f"{save_name}.npy")
        np.save(save_path, six_channel)
        
        return save_path, label
    
    generated_dir = Path(generated_dir)
    real_dir = Path(real_dir)
    
    gen_images = sorted(list(generated_dir.glob("*.jpg")) + list(generated_dir.glob("*.png")))
    real_images = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    
    metadata = []
    
    print(f"Preprocessing generated images: {len(gen_images)} images")
    for i, img_path in enumerate(tqdm(gen_images)):
        save_name = f"gen_{i:05d}"
        try:
            path, label = process_single_image(img_path, 1, save_name)
            metadata.append({'path': path, 'label': label})
        except Exception as e:
            print(f"Failed: {img_path}, {e}")
    
    print(f"Preprocessing real images: {len(real_images)} images")
    for i, img_path in enumerate(tqdm(real_images)):
        save_name = f"real_{i:05d}"
        try:
            path, label = process_single_image(img_path, 0, save_name)
            metadata.append({'path': path, 'label': label})
        except Exception as e:
            print(f"Failed: {img_path}, {e}")
    
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Preprocessing complete: {len(metadata)} images")
    return metadata


# ==================== Dataset Class (with Data Augmentation) ====================
class PreprocessedDataset(Dataset):
    def __init__(self, metadata_path, is_train=True):
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.is_train = is_train
        
        # Data augmentation (only for training)
        if is_train:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
        else:
            self.augment = None
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        six_channel = np.load(item['path'])
        label = item['label']
        six_channel = torch.from_numpy(six_channel).float().permute(2, 0, 1)
        
        # Augment RGB channels during training
        if self.is_train and self.augment:
            rgb = six_channel[:3]
            other = six_channel[3:]
            rgb = self.augment(rgb)
            six_channel = torch.cat([rgb, other], dim=0)
        
        return six_channel, label


# ==================== Model (with Dropout) ====================
class SixChannelResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.6):
        super(SixChannelResNet18, self).__init__()
        
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            self.conv1.weight[:, 3:, :, :] = torch.randn(64, 3, 7, 7) * 0.01
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.dropout = nn.Dropout(p=dropout)
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
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ==================== Training Function with ACC, F1, AUC ====================
def train_model(model, train_loader, val_loader, num_epochs=6, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    best_epoch = 0
    best_f1 = 0.0
    best_auc = 0.0
    patience_counter = 0
    patience = 5
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': [],
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*60}')
        
        # Training
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_train_labels, all_train_preds)
        epoch_f1 = f1_score(all_train_labels, all_train_preds)
        
        print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (fake)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        
        print(f'Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        
        # 记录历史
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_f1'].append(epoch_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch + 1)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1,
                'best_auc': best_auc,
                'train_acc': epoch_acc,
                'train_f1': epoch_f1,
                'train_loss': epoch_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f'✓ Saved best model - Epoch {epoch+1} | Val Acc: {best_acc:.4f} | F1: {best_f1:.4f} | AUC: {best_auc:.4f}')
        else:
            patience_counter += 1
            print(f'✗ No improvement ({patience_counter}/{patience})')
            
        # Early stopping
        if patience_counter >= patience:
            print(f'\n Early stopping at epoch {epoch+1}')
            break
    
    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'{"="*60}')
    print(f'Best model: Epoch {best_epoch}')
    print(f'  Validation Accuracy: {best_acc:.4f}')
    print(f'  Validation F1 Score: {best_f1:.4f}')
    print(f'  Validation AUC:      {best_auc:.4f}')
    print(f'{"="*60}')
    
    return model, history


# ==================== Plot Training History ====================
def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Loss
    axes[0, 0].plot(history['epochs'], history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=8)
    axes[0, 0].plot(history['epochs'], history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(history['epochs'])
    
    # Plot Accuracy
    axes[0, 1].plot(history['epochs'], [acc*100 for acc in history['train_acc']], 'o-', label='Train Acc', linewidth=2, markersize=8)
    axes[0, 1].plot(history['epochs'], [acc*100 for acc in history['val_acc']], 's-', label='Val Acc', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(history['epochs'])
    axes[0, 1].set_ylim([0, 105])
    
    # Plot F1 Score
    axes[1, 0].plot(history['epochs'], [f1*100 for f1 in history['train_f1']], 'o-', label='Train F1', linewidth=2, markersize=8)
    axes[1, 0].plot(history['epochs'], [f1*100 for f1 in history['val_f1']], 's-', label='Val F1', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(history['epochs'])
    axes[1, 0].set_ylim([0, 105])
    
    # Plot AUC Score (validation only)
    axes[1, 1].plot(history['epochs'], [auc*100 for auc in history['val_auc']], 's-', label='Val AUC', linewidth=2, markersize=8, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('AUC Score (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation AUC Score', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(history['epochs'])
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n📊 Training history plot saved to: {save_path}')
    plt.show()


# ==================== Main Function ====================
if __name__ == '__main__':
    generated_dir = r'C:\Users\zimol\Desktop\Fake-Image-Recognition\data_SD_50test\generated_image'
    real_dir = r'C:\Users\zimol\Desktop\Fake-Image-Recognition\data_SD_50test\real_images'
    preprocessed_dir = r'C:\Users\zimol\Desktop\Fake-Image-Recognition\preprocessed_data'
    
    metadata_path = os.path.join(preprocessed_dir, 'metadata.pkl')
    
    # Preprocessing
    if not os.path.exists(metadata_path):
        print("Starting preprocessing...")
        preprocess_dataset(generated_dir, real_dir, preprocessed_dir)
    
    # Load data and split - 验证集固定为500
    with open(metadata_path, 'rb') as f:
        all_metadata = pickle.load(f)
    
    import random
    random.seed(42)
    
    # 按标签分组
    label_0 = [item for item in all_metadata if item['label'] == 0]
    label_1 = [item for item in all_metadata if item['label'] == 1]
    
    # 打乱
    random.shuffle(label_0)
    random.shuffle(label_1)
    
    # 验证集：每类各250张，总共500张
    val_size_per_class = 250
    val_metadata = label_0[:val_size_per_class] + label_1[:val_size_per_class]
    train_metadata = label_0[val_size_per_class:] + label_1[val_size_per_class:]
    
    # 再次打乱
    random.shuffle(train_metadata)
    random.shuffle(val_metadata)
    
    print(f"\n{'='*60}")
    print(f"Dataset Split:")
    print(f"  Training:   {len(train_metadata)} samples")
    print(f"  Validation: {len(val_metadata)} samples")
    print(f"{'='*60}\n")
    
    # Save temporary metadata
    with open('train_temp.pkl', 'wb') as f:
        pickle.dump(train_metadata, f)
    with open('val_temp.pkl', 'wb') as f:
        pickle.dump(val_metadata, f)
    
    # Create datasets
    train_dataset = PreprocessedDataset('train_temp.pkl', is_train=True)
    val_dataset = PreprocessedDataset('val_temp.pkl', is_train=False)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Anti-overfitting strategies:")
    print("  - Data Augmentation")
    print("  - Dropout: 0.6")
    print("  - Weight Decay: 5e-4")
    print("  - Learning Rate: 0.0001")
    print("  - Early Stopping (patience=5)")
    print("  - Epochs: 6\n")
    
    model = SixChannelResNet18(num_classes=2, pretrained=True, dropout=0.6)
    model = model.to(device)
    
    # Train
    model, history = train_model(model, train_loader, val_loader, num_epochs=6, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n Training finished!")