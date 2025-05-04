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
from sklearn.metrics import mean_squared_error
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
output_dir = 'output/region'
output_name = '2022101015_2.csv'
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_labels_path)
print(f"Original train data shape: {train_df.shape}")

# Remove rows with Region_ID not in range [1, 15] or angle > 360
train_df = train_df[(train_df['Region_ID'] >= 1) & (train_df['Region_ID'] <= 15) & (train_df['angle'] <= 360)]
print(f"Cleaned train data shape: {train_df.shape}")

val_df = pd.read_csv(val_labels_path)
print(f"Validation data shape: {val_df.shape}")

print("\nLatitude range:", train_df['latitude'].min(), "-", train_df['latitude'].max())
print("Longitude range:", train_df['longitude'].min(), "-", train_df['longitude'].max())

# Global max/min for normalization
lat_min = min(train_df['latitude'].min(), val_df['latitude'].min())
lat_max = max(train_df['latitude'].max(), val_df['latitude'].max())
lon_min = min(train_df['longitude'].min(), val_df['longitude'].min())
lon_max = max(train_df['longitude'].max(), val_df['longitude'].max())

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

class LatLongDataset(Dataset):
    def __init__(self, df, img_dir, lat_min, lat_max, lon_min, lon_max, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.angle_min, self.angle_max = 0, 360
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Labels: Latitude and longitude
        latitude = self.df.iloc[idx]['latitude']
        longitude = self.df.iloc[idx]['longitude']
        
        # Normalize coordinates to [0, 1]
        latitude = (latitude - self.lat_min) / (self.lat_max - self.lat_min)
        longitude = (longitude - self.lon_min) / (self.lon_max - self.lon_min)

        label = torch.tensor([latitude, longitude], dtype=torch.float32)
        
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
        
        # Normalize angle with cyclical encoding
        angle = self.df.iloc[idx]['angle']
        angle_sin = np.sin(2 * np.pi * angle / 360)
        angle_cos = np.cos(2 * np.pi * angle / 360)

        # One-hot encoding for Region_ID
        region_id = self.df.iloc[idx]['Region_ID']
        
        metadata = torch.tensor([
            time_sin, time_cos,
            angle_sin, angle_cos,
            region_id
        ], dtype=torch.float32)

        # Original coordinates for evaluation
        coords = torch.tensor([latitude, longitude], dtype=torch.float32)
        
        return image, metadata, label, coords

train_dataset = LatLongDataset(train_df, train_images_dir, lat_min, lat_max, lon_min, lon_max, transform=train_transforms)
val_dataset = LatLongDataset(val_df, val_images_dir, lat_min, lat_max, lon_min, lon_max, transform=val_transforms)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class LatLongClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LatLongClassifier, self).__init__()
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
            nn.Linear(19, 64),  # 19 features: sin/cos time + sin/cos angle + 15 regions
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
    
class LinearMSELoss(nn.Module):
    def __init__(self):
        super(LinearMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

model = LatLongClassifier().to(device)
criterion = LinearMSELoss()
optimizer = optim.AdamW([
    {'params': model.efficientnet.parameters(), 'lr': 0.0001},
    {'params': model.metadata_encoder.parameters(), 'lr': 0.0005},
    {'params': model.regressor.parameters(), 'lr': 0.0005}
], weight_decay=1e-4)

def denormalize_coords(coords, lat_min, lat_max, lon_min, lon_max):
    lat = coords[:, 0] * (lat_max - lat_min) + lat_min
    lon = coords[:, 1] * (lon_max - lon_min) + lon_min
    return torch.stack((lat, lon), dim=1)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_val_mse = float('inf')
    best_model_weights = None
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
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
        for images, metadata, targets, coords in train_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images, metadata)
                loss = criterion(outputs, targets)
            
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
            pred_coords = denormalize_coords(outputs.detach().cpu(), lat_min, lat_max, lon_min, lon_max)
            all_train_true.extend(pred_coords)
            all_train_pred.extend(coords)
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar
            train_bar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        all_train_true = torch.cat(all_train_true)
        all_train_pred = torch.cat(all_train_pred)
        epoch_train_mse = mean_squared_error(all_train_true.numpy(), all_train_pred.numpy())

        history['train_loss'].append(epoch_train_loss)
        history['train_mse'].append(epoch_train_mse)
        
        # Validation
        model.eval()
        running_loss = 0.0
        all_val_true = []
        all_val_pred = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, metadata, targets, coords in val_bar:
                images = images.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images, metadata)
                loss = criterion(outputs, targets)
                
                # Statistics
                pred_coords = denormalize_coords(outputs.detach().cpu(), lat_min, lat_max, lon_min, lon_max)
                all_val_true.extend(pred_coords)
                all_val_pred.extend(coords)
                running_loss += loss.item() * images.size(0)
                
                # Update progress bar
                val_bar.set_postfix({'loss': loss.item()})
        
        epoch_val_loss = running_loss / len(train_loader.dataset)
        all_val_true = torch.cat(all_val_true)
        all_val_pred = torch.cat(all_val_pred)
        epoch_val_mse = mean_squared_error(all_val_true.numpy(), all_val_pred.numpy())
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train MSE: {epoch_train_mse:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val MSE: {epoch_val_mse:.4f}")
        
        # Save the best model
        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            best_model_weights = model.state_dict().copy()
            print(f"New best model with MSE: {best_val_mse:.4f}")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping check
        if no_improve >= patience and epoch >= 10:
            print(f"Early stopping at epoch {epoch+1} as validation MSE hasn't improved for {patience} epochs")
            break
    
    print(f"Best validation MSE: {best_val_mse:.4f}")
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
plt.plot(history['train_mse'], label='Train MSE')
plt.plot(history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training and Validation MSE')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

def predict_latlong():
    # For validation set
    model.eval()
    val_predictions = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Predicting validation set")
        for images, metadata, _ in val_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            
            # Original image
            outputs = model(images, metadata)
            
            # Horizontal flip
            flipped_images = torch.flip(images, dims=[3])
            outputs_flip = model(flipped_images, metadata)
            outputs = (outputs + outputs_flip) / 2
            
            # Denormalize predictions
            preds = denormalize_coords(outputs.cpu(), lat_min, lat_max, lon_min, lon_max)
            
            for lat, lon in preds:
                val_predictions.append((lat.item(), lon.item()))
    
    result_df = pd.DataFrame({
        'id': range(738),  # 0 to 737
        'latitude': [0] * 738,  # Initialize with zeros
        'longitude': [0] * 738  # Initialize with zeros
    })
    for i, (lat, lon) in enumerate(val_predictions):
        result_df.loc[i, 'latitude'] = lat
        result_df.loc[i, 'longitude'] = lon
    
    # Save results to CSV
    result_df.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f"Results saved to {os.path.join(output_dir, output_name)}")

predict_latlong()