import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_images_dir = 'data/images_train'
val_images_dir = 'data/images_val'
train_labels_path = 'data/labels_train.csv'
val_labels_path = 'data/labels_val.csv'
output_dir = 'output/direction'
output_name = '2022101015_2.csv'
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_labels_path)
print(f"Original train data shape: {train_df.shape}")

# Remove rows with Region_ID not in range [1, 15] or angle > 360
train_df = train_df[(train_df['Region_ID'] >= 1) & (train_df['Region_ID'] <= 15) & (train_df['angle'] <= 360)]
print(f"Cleaned train data shape: {train_df.shape}")

val_df = pd.read_csv(val_labels_path)
print(f"Validation data shape: {val_df.shape}")

print("\nAngle distribution statistics in training data:")
print(train_df['angle'].describe())

# Custom function to calculate angle MAE (Mean Absolute Error)
def angle_mae(y_true, y_pred):
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    errors = np.minimum(errors, 360 - errors)
    return np.mean(errors)

# Image transformation for EfficientNet
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms without augmentation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class AngleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # Pre-compute min-max values for normalization
        self.lat_min, self.lat_max = df['latitude'].min(), df['latitude'].max()
        self.lon_min, self.lon_max = df['longitude'].min(), df['longitude'].max()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Label: Angle value
        label = self.df.iloc[idx]['angle']
        
        timestamp = self.df.iloc[idx]['timestamp']
        parts = timestamp.strip().split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours, minutes = map(int, parts)
            seconds = 0
        time_in_seconds = hours * 3600 + minutes * 60 + seconds
        total_seconds_in_day = 24 * 3600
        
        # Cyclical encoding for time (sin and cos components)
        time_sin = np.sin(2 * np.pi * time_in_seconds / total_seconds_in_day)
        time_cos = np.cos(2 * np.pi * time_in_seconds / total_seconds_in_day)
        
        # Normalize coordinates to [0, 1]
        latitude = (self.df.iloc[idx]['latitude'] - self.lat_min) / (self.lat_max - self.lat_min)
        longitude = (self.df.iloc[idx]['longitude'] - self.lon_min) / (self.lon_max - self.lon_min)
        
        # One-hot encoding for Region_ID
        region_id = self.df.iloc[idx]['Region_ID']
        
        metadata = torch.tensor([
            time_sin, time_cos,
            latitude, longitude,
            region_id
        ], dtype=torch.float32)
        
        return image, metadata, label

train_dataset = AngleDataset(train_df, train_images_dir, transform=train_transforms)
val_dataset = AngleDataset(val_df, val_images_dir, transform=val_transforms)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class AngleRegressor(nn.Module):
    def __init__(self, num_classes=2):
        super(AngleRegressor, self).__init__()
        # Pre-trained EfficientNet B3
        self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Fine-tuning
        layers = 3
        ct = 0
        for child in self.efficientnet.features.children():
            ct += 1
            if ct < layers:
                for param in child.parameters():
                    param.requires_grad = False
        num_ftrs = self.efficientnet.classifier[1].in_features
        
        # Replace the final classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Process metadata
        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 64),  # 5 features: sin/cos time, lat, lon, Region_ID
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined regressor
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights(self.metadata_encoder)
        self._initialize_weights(self.regressor)
        
    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, metadata):
        # Extract features
        features = self.efficientnet(x)
        metadata_features = self.metadata_encoder(metadata)
        
        # Concatenate features
        combined = torch.cat((features, metadata_features), dim=1)
        
        # Pass through the regressor
        x = self.regressor(combined)
        return x

class CircularMSELoss(nn.Module):
    def __init__(self):
        super(CircularMSELoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # Convert angles to radians
        y_true_rad = y_true * (np.pi / 180.0)
        
        # Get true sin and cos values
        true_sin = torch.sin(y_true_rad)
        true_cos = torch.cos(y_true_rad)
        true_sin_cos = torch.stack([true_sin, true_cos], dim=1)
        
        # Calculate MSE loss between predicted and true sin/cos values
        loss = torch.mean((y_pred - true_sin_cos) ** 2)
        return loss

# Convert predicted sin/cos to angle in degrees
def sin_cos_to_angle(sin_val, cos_val):
    angle_rad = torch.atan2(sin_val, cos_val)
    angle_deg = angle_rad * (180 / np.pi)
    angle_deg = (angle_deg + 360) % 360 # Ensure angle is between 0 and 360
    return angle_deg

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

model = AngleRegressor().to(device)
criterion = CircularMSELoss()
optimizer = optim.AdamW([
    {'params': model.efficientnet.parameters(), 'lr': 0.0001},
    {'params': model.metadata_encoder.parameters(), 'lr': 0.0005},
    {'params': model.regressor.parameters(), 'lr': 0.0005}
], weight_decay=1e-4)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_mae = float('inf')
    best_model_weights = None
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    patience = 10
    no_improve = 0
    total_steps = int(len(train_loader) * num_epochs * 1.1)
    
    # One Cycle learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0001, 0.0005, 0.0005], 
        total_steps=total_steps, 
        pct_start=0.3,
        div_factor=25, 
        final_div_factor=1000
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        all_train_true = []
        all_train_pred = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, metadata, angles in train_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            angles = torch.tensor(angles, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images, metadata)
                loss = criterion(outputs, angles)
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            
            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler every batch
            scheduler.step()
            
            # Statistics
            pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
            all_train_true.extend(angles.cpu().numpy())
            all_train_pred.extend(pred_angles.cpu().detach().numpy())
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar
            train_bar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_mae = angle_mae(all_train_true, all_train_pred)
        history['train_loss'].append(epoch_train_loss)
        history['train_mae'].append(epoch_train_mae)
        
        # Validation
        model.eval()
        running_loss = 0.0
        all_val_true = []
        all_val_pred = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, metadata, angles in val_bar:
                images = images.to(device)
                metadata = metadata.to(device)
                angles = torch.tensor(angles, dtype=torch.float32).to(device)
                
                # Forward pass
                outputs = model(images, metadata)
                loss = criterion(outputs, angles)
                
                # Statistics
                pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
                running_loss += loss.item() * images.size(0)

                all_val_true.extend(angles.cpu().numpy())
                all_val_pred.extend(pred_angles.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({'loss': loss.item()})
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_mae = angle_mae(all_val_true, all_val_pred)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")
        
        # Save the best model
        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            best_model_weights = model.state_dict().copy()
            print(f"New best model with MAE: {best_val_mae:.4f}")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping check
        if no_improve >= patience and epoch >= 10:
            print(f"Early stopping at epoch {epoch+1} as validation MAE hasn't improved for {patience} epochs")
            break
    
    print(f"Best validation MAE: {best_val_mae:.4f}")
    model.load_state_dict(best_model_weights)
    return model, history

# Train the model and save the best weights
num_epochs = 100
model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_mae'], label='Train MAE')
plt.plot(history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (degrees)')
plt.legend()
plt.title('Training and Validation MAE')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

def predict_angles():
    # For validation set
    model.eval()
    val_predictions = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Predicting validation set")
        for images, metadata, _ in val_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            
            outputs = model(images, metadata)
            pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
            pred_angles = torch.round(pred_angles).clamp(0, 360).int()
            
            val_predictions.extend(pred_angles.cpu().numpy().tolist())
    
    result_df = pd.DataFrame({
        'id': range(738),  # 0 to 737
        'angle': [0] * 738  # Initialize with zeros
    })
    result_df.loc[:368, 'angle'] = val_predictions
    
    # Save results to CSV
    result_df.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f"Results saved to {os.path.join(output_dir, output_name)}")

predict_angles()