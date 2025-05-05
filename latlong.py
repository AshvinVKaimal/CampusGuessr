import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
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
output_dir = 'output/latlong'
output_name = '2022101015_2.csv'
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_labels_path)
print(f"Original train data shape: {train_df.shape}")

# Remove rows with Region_ID not in range [1, 15] or angle > 360
train_df = train_df[(train_df['Region_ID'] >= 1) & (train_df['Region_ID'] <= 15) & (train_df['angle'] <= 360)]
print(f"Cleaned train data shape: {train_df.shape}")

val_df = pd.read_csv(val_labels_path)
print(f"Validation data shape: {val_df.shape}")

print("\nRegion_ID distribution in training data:")
print(train_df['Region_ID'].value_counts().sort_index())

region_stats = train_df.groupby('Region_ID')[['latitude', 'longitude']].agg(['mean', 'std', 'min', 'max'])
print("\nRegion statistics:")
print(region_stats)

region_boundaries = {}
for region_id in range(1, 16):
    if region_id in train_df['Region_ID'].values:
        region_data = train_df[train_df['Region_ID'] == region_id]
        region_boundaries[region_id] = {
            'lat_min': region_data['latitude'].min(),
            'lat_max': region_data['latitude'].max(),
            'lat_mean': region_data['latitude'].mean(),
            'lat_std': region_data['latitude'].std(),
            'lon_min': region_data['longitude'].min(),
            'lon_max': region_data['longitude'].max(),
            'lon_mean': region_data['longitude'].mean(),
            'lon_std': region_data['longitude'].std()
        }

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
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # Pre-compute min-max values for normalization
        self.lat_min, self.lat_max = df['latitude'].min(), df['latitude'].max()
        self.lon_min, self.lon_max = df['longitude'].min(), df['longitude'].max()
        self.angle_min, self.angle_max = 0, 360  # Angle is between 0 and 360 degrees
        
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
        label = torch.tensor([latitude, longitude], dtype=torch.float32)

        # Normalize latitude and longitude
        latitude = (latitude - self.lat_min) / (self.lat_max - self.lat_min)
        longitude = (longitude - self.lon_min) / (self.lon_max - self.lon_min)
        coords = torch.tensor([latitude, longitude], dtype=torch.float32)

        region_id = self.df.iloc[idx]['Region_ID'] - 1  # Convert to zero-based index
        
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
        
        metadata = torch.tensor([
            time_sin, time_cos,
            angle_sin, angle_cos
        ], dtype=torch.float32)
        
        return image, metadata, coords, region_id, label

train_dataset = LatLongDataset(train_df, train_images_dir, transform=train_transforms)
val_dataset = LatLongDataset(val_df, val_images_dir, transform=val_transforms)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class LatLongRegressor(nn.Module):
    def __init__(self, num_classes=2):
        super(LatLongRegressor, self).__init__()
        # Pre-trained ConvNeXt Large model
        self.convnext = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k', pretrained=True)
        
        # # Fine-tuning
        # layers = 5
        # ct = 0
        # for child in self.convnext.features.children():
        #     ct += 1
        #     if ct < layers:
        #         for param in child.parameters():
        #             param.requires_grad = False
        # num_ftrs = self.convnext.classifier[2].in_features
        
        # # Remove default classifier
        # self.convnext.classifier = nn.Identity()

        num_ftrs = self.convnext.num_features
        self.convnext.reset_classifier(0)  # Remove the classifier layer
        
        # Process metadata
        self.metadata_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 4 features: sin/cos time, sin/cos angle
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Shared features
        self.shared_features = nn.Sequential(
            nn.Linear(num_ftrs + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Coordinate regressor head
        self.coords_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Region classifier head
        self.region_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 15)  # 15 regions
        )

        # Combined regressor
        # self.regressor = nn.Sequential(
        #     nn.Linear(num_ftrs + 128, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, num_classes)
        # )
        
        self._initialize_weights(self.metadata_encoder)
        self._initialize_weights(self.shared_features)
        self._initialize_weights(self.coords_regressor)
        self._initialize_weights(self.region_classifier)
        
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
        features = self.convnext(x)
        metadata_features = self.metadata_encoder(metadata)
        
        # Concatenate features
        combined = torch.cat((features, metadata_features), dim=1)
        
        # Pass through the regressor
        x = self.shared_features(combined)
        coords = self.coords_regressor(x)
        region = self.region_classifier(x)
        return coords, region
    
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

class_counts = train_df['Region_ID'].value_counts().sort_index().values
class_weights = 1.0 / (class_counts / class_counts.sum())
class_weights = np.sqrt(class_weights)  # Square root smoothing
class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
class_weights = torch.FloatTensor(class_weights).to(device)

model = LatLongRegressor().to(device)
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW([
    {'params': model.convnext.parameters(), 'lr': 0.0001},
    {'params': model.metadata_encoder.parameters(), 'lr': 0.0005},
    {'params': model.shared_features.parameters(), 'lr': 0.0005},
    {'params': model.coords_regressor.parameters(), 'lr': 0.0005},
    {'params': model.region_classifier.parameters(), 'lr': 0.0005}
], weight_decay=1e-4)

def denormalize_coords(coords, stats):
    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)  # Add batch dimension if missing
    if coords.shape[1] != 2:
        raise ValueError(f"Expected coords to have shape [batch_size, 2], got {coords.shape}")
    
    lat = coords[:, 0] * (stats.lat_max - stats.lat_min) + stats.lat_min
    lon = coords[:, 1] * (stats.lon_max - stats.lon_min) + stats.lon_min

    return torch.stack([lat, lon], dim=1)

def train_model(model, train_loader, val_loader, optimizer, num_epochs=20):
    best_val_mse = float('inf')
    best_model_weights = None
    history = {
        'train_coord_loss': [], 'train_region_loss': [], 'train_total_loss': [],
        'val_coord_loss': [], 'val_region_loss': [], 'val_total_loss': [],
        'val_mse': []
    }
    patience = 10
    no_improve = 0
    aux_start, aux_end = 0.3, 0.1
    total_steps = int(len(train_loader) * num_epochs * 1.1)
    
    # One Cycle learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0001, 0.0005, 0.0005, 0.0005, 0.0005], 
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
        running_coord_loss = 0.0
        running_region_loss = 0.0
        running_total_loss = 0.0
        aux_wt = aux_start + (aux_end - aux_start) * (epoch / num_epochs)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, metadata, coords, regions, _ in train_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            coords = coords.to(device)
            regions = regions.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred_coords, pred_regions = model(images, metadata)
                coord_loss = mse_loss(pred_coords, coords)
                region_loss = ce_loss(pred_regions, regions)
                total_loss = coord_loss + region_loss * aux_wt
            
            # Scale loss and perform backward pass
            scaler.scale(total_loss).backward()
            
            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler every batch
            scheduler.step()
            
            # Statistics
            running_coord_loss += coord_loss.item() * images.size(0)
            running_region_loss += region_loss.item() * images.size(0)
            running_total_loss += total_loss.item() * images.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'coord_loss': coord_loss.item(),
                'region_loss': region_loss.item(),
                'total_loss': total_loss.item()
            })
        
        epoch_train_coord_loss = running_coord_loss / len(train_loader.dataset)
        epoch_train_region_loss = running_region_loss / len(train_loader.dataset)
        epoch_train_total_loss = running_total_loss / len(train_loader.dataset)
        history['train_coord_loss'].append(epoch_train_coord_loss)
        history['train_region_loss'].append(epoch_train_region_loss)
        history['train_total_loss'].append(epoch_train_total_loss)
        
        # Validation
        model.eval()
        running_coord_loss = 0.0
        running_region_loss = 0.0
        running_total_loss = 0.0
        all_val_true = []
        all_val_pred = []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, metadata, coords, regions, labels in val_bar:
                images = images.to(device)
                metadata = metadata.to(device)
                coords = coords.to(device)
                regions = regions.to(device)
                
                # Forward pass
                pred_coords, pred_regions = model(images, metadata)
                coord_loss = mse_loss(pred_coords, coords)
                region_loss = ce_loss(pred_regions, regions)
                total_loss = coord_loss + region_loss * aux_wt
                
                # Statistics
                running_coord_loss += coord_loss.item() * images.size(0)
                running_region_loss += region_loss.item() * images.size(0)
                running_total_loss += total_loss.item() * images.size(0)

                # Denormalize coordinates
                pred_coords = denormalize_coords(pred_coords, val_dataset)

                if epoch == 0 and len(all_val_true) == 0:
                    print(f"Debug - true_coords shape: {len(all_val_true)}")
                    print(f"Debug - pred_coords shape: {pred_coords.shape}")
                
                all_val_true.extend(labels.cpu().numpy())
                all_val_pred.extend(pred_coords.cpu().numpy())

                # Update progress bar
                val_bar.set_postfix({'loss': total_loss.item()})
        
        epoch_val_coord_loss = running_coord_loss / len(val_loader.dataset)
        epoch_val_region_loss = running_region_loss / len(val_loader.dataset)
        epoch_val_total_loss = running_total_loss / len(val_loader.dataset)
        epoch_val_mse = mean_squared_error(np.array(all_val_true), np.array(all_val_pred))
        history['val_coord_loss'].append(epoch_val_coord_loss)
        history['val_region_loss'].append(epoch_val_region_loss)
        history['val_total_loss'].append(epoch_val_total_loss)
        history['val_mse'].append(epoch_val_mse)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Coordinate Loss: {epoch_train_coord_loss:.6f}, Train Region Loss: {epoch_train_region_loss:.6f}, Train Total Loss: {epoch_train_total_loss:.6f}")
        print(f"Validation Coordinate Loss: {epoch_val_coord_loss:.6f}, Validation Region Loss: {epoch_val_region_loss:.6f}, Validation Total Loss: {epoch_val_total_loss:.6f}")
        print(f"Validation MSE: {epoch_val_mse:.6f}")
        
        # Save the best model
        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            best_model_weights = model.state_dict().copy()
            print(f"New best model with MSE: {best_val_mse:.6f}")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping check
        if no_improve >= patience and epoch >= 10:
            print(f"Early stopping at epoch {epoch+1} as validation MSE hasn't improved for {patience} epochs")
            break
    
    print(f"Best validation MSE: {best_val_mse:.6f}")
    model.load_state_dict(best_model_weights)
    return model, history

# Train the model and save the best weights
num_epochs = 100
model, history = train_model(model, train_loader, val_loader, optimizer, num_epochs)
torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

# Plot training history
plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)
plt.plot(history['train_coord_loss'], label='Train Coordinate Loss')
plt.plot(history['val_coord_loss'], label='Validation Coordinate Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Coordinate Loss')

plt.subplot(2, 2, 2)
plt.plot(history['train_region_loss'], label='Train Region Loss')
plt.plot(history['val_region_loss'], label='Validation Region Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Region Classification Loss')

plt.subplot(2, 2, 3)
plt.plot(history['train_total_loss'], label='Train Total Loss')
plt.plot(history['val_total_loss'], label='Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Total Loss')

plt.subplot(2, 2, 4)
plt.plot(history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Validation MSE (Denormalized Coordinates)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()

def predict_coords():
    # For validation set
    model.eval()
    val_pred = []
    val_true = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Predicting validation set")
        for images, metadata, _, _, coords in val_bar:
            images = images.to(device)
            metadata = metadata.to(device)
            
            # Original image
            pred_coords, _ = model(images, metadata)
            
            # Horizontal flip
            flipped_images = torch.flip(images, dims=[3])
            pred_coords_flip, _ = model(flipped_images, metadata)
            
            # Average predictions from both augmentations
            final_coords = (pred_coords + pred_coords_flip) / 2
            if len(val_pred) == 0:
                print(f"Debug - pred_coords shape: {pred_coords.shape}")
                print(f"Debug - pred_coords_flip shape: {pred_coords_flip.shape}")
                print(f"Debug - final_coords shape: {final_coords.shape}")

            final_coords = denormalize_coords(final_coords, val_dataset)
            if len(val_pred) == 0:
                print(f"Debug - final_coords shape: {final_coords.shape}")
                print(f"Debug - coords shape: {coords.shape}")
                print(f"Debug - pred: {final_coords[0]}, true: {coords[0]}")
            
            val_pred.extend((final_coords.cpu().numpy()).tolist())
            val_true.extend(coords.cpu().numpy().tolist())

    val_mse = mean_squared_error(np.array(val_true), np.array(val_pred))
    print(f"Validation MSE (Denormalized Coordinates): {val_mse:.6f}")
    
    result_df = pd.DataFrame({
        'id': range(738),  # 0 to 737
        'latitude': [0.0] * 738,  # Initialize with zeros
        'longitude': [0.0] * 738  # Initialize with zeros
    })
    for i, coords in enumerate(val_pred):
        result_df.at[i, 'latitude'] = coords[0]
        result_df.at[i, 'longitude'] = coords[1]
    
    # Save results to CSV
    result_df.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f"Results saved to {os.path.join(output_dir, output_name)}")

    # Scatter plot of true vs predicted coordinates
    plt.figure(figsize=(10, 10))
    plt.scatter([coord[1] for coord in val_true], [coord[0] for coord in val_true], 
                alpha=0.5, label='True Coordinates', color='blue')
    plt.scatter([coord[1] for coord in val_pred], [coord[0] for coord in val_pred], 
                alpha=0.5, label='Predicted Coordinates', color='red')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True vs Predicted Coordinates')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'coordinates_scatter.png'))
    plt.close()
    
    # Analyze prediction error by Region_ID
    val_region_ids = val_df['Region_ID'].values
    region_errors = {}
    
    for i, (true_coord, pred_coord) in enumerate(zip(val_true, val_pred)):
        region_id = val_region_ids[i]
        error = np.sqrt((true_coord[0] - pred_coord[0])**2 + (true_coord[1] - pred_coord[1])**2)
        
        if region_id not in region_errors:
            region_errors[region_id] = []
        
        region_errors[region_id].append(error)
    
    # Calculate mean error by region
    region_mean_errors = {region: np.mean(errors) for region, errors in region_errors.items()}
    
    # Plot mean error by region
    plt.figure(figsize=(12, 6))
    regions = list(region_mean_errors.keys())
    errors = list(region_mean_errors.values())
    
    plt.bar(regions, errors)
    plt.xlabel('Region ID')
    plt.ylabel('Mean Coordinate Error')
    plt.title('Mean Prediction Error by Region')
    plt.xticks(regions)
    plt.savefig(os.path.join(output_dir, 'region_errors.png'))
    plt.close()

predict_coords()