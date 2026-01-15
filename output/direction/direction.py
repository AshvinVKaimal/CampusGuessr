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
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_images_dir = 'data/images_train'
val_images_dir = 'data/images_val'
test_images_dir = 'data/images_test'
train_labels_path = 'data/labels_train.csv'
val_labels_path = 'data/labels_val.csv'
output_dir = 'output/direction'
output_name = 'results.csv'
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_labels_path)
print(f"Original train data shape: {train_df.shape}")

# Remove rows with Region_ID not in range [1, 15] or angle > 360 and < 0
train_df = train_df[(train_df["Region_ID"].between(1, 15)) & (train_df["angle"].between(0, 360))].reset_index(drop=True)
print(f"Cleaned train data shape: {train_df.shape}")

val_df = pd.read_csv(val_labels_path).reset_index(drop=True)
print(f"Validation data shape: {val_df.shape}")

print("\nAngle distribution statistics in training data:")
print(train_df['angle'].describe())

# Custom functions to calculate angle MAE (Mean Absolute Error)
def angle_to_sincos(angle):
    rad = torch.deg2rad(angle)
    return torch.stack([torch.sin(rad), torch.cos(rad)], dim=1)

def sincos_to_angle(pred):
    rad = torch.atan2(pred[:, 0], pred[:, 1])
    return torch.rad2deg(rad) % 360

def circular_mae(y_true, y_pred):
    delta = (y_pred - y_true + 180) % 360 - 180
    return torch.mean(torch.abs(delta)).item()
    # errors = np.abs(np.array(y_true) - np.array(y_pred))
    # errors = np.minimum(errors, 360 - errors)
    # return np.mean(errors)

# Image transformation
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
    def __init__(self, df, img_dir, transform=None, test=False):
        self.df = df                # Test set will use validation DataFrame
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

        if test:
            self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Pre-compute min-max values for normalization
        # self.lat_min, self.lat_max = df['latitude'].min(), df['latitude'].max()
        # self.lon_min, self.lon_max = df['longitude'].min(), df['longitude'].max()
        
    def __len__(self):
        if self.test:
            return len(self.imgs)
        else:
            return len(self.df)
    
    def __getitem__(self, idx):
        if self.test:
            img_name = self.imgs[idx]
            angle = None  # No label for test set
        else:
            img_name = self.df.iloc[idx]['filename']
            angle = torch.tensor(self.df.iloc[idx]['angle'], dtype=torch.float32)
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Label: Angle value
        # label = self.df.iloc[idx]['angle']
        
        if self.test:
            return image, torch.zeros(2)  # Dummy label for test set
        else:
            return image, angle_to_sincos(angle.unsqueeze(0)).squeeze(0)

train_dataset = AngleDataset(train_df, train_images_dir, train_transforms)
val_dataset = AngleDataset(val_df, val_images_dir, val_transforms)
test_dataset = AngleDataset(val_df, test_images_dir, val_transforms, test=True)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class AngleRegressor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Pre-trained ConvNeXt model
        self.backbone = timm.create_model('convnext_base', pretrained=True)
        fts = self.backbone.head.fc.in_features
        self.backbone.head.fc = torch.nn.Linear(fts, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# class CircularMSELoss(nn.Module):
#     def __init__(self):
#         super(CircularMSELoss, self).__init__()
        
#     def forward(self, y_pred, y_true):
#         # Convert angles to radians
#         y_true_rad = y_true * (np.pi / 180.0)
        
#         # Get true sin and cos values
#         true_sin = torch.sin(y_true_rad)
#         true_cos = torch.cos(y_true_rad)
#         true_sin_cos = torch.stack([true_sin, true_cos], dim=1)
        
#         # Calculate MSE loss between predicted and true sin/cos values
#         loss = torch.mean((y_pred - true_sin_cos) ** 2)
#         return loss

# # Convert predicted sin/cos to angle in degrees
# def sin_cos_to_angle(sin_val, cos_val):
#     angle_rad = torch.atan2(sin_val, cos_val)
#     angle_deg = angle_rad * (180 / np.pi)
#     angle_deg = (angle_deg + 360) % 360 # Ensure angle is between 0 and 360
#     return angle_deg

model = AngleRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_mae = float('inf')
    best_weights = None
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    patience = 7
    no_improve = 0
    
    # One Cycle learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-4, 
        total_steps=num_epochs * len(train_loader), 
        pct_start=0.3,
        div_factor=25, 
        final_div_factor=1000
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_true, train_pred = [], []
        
        for imgs, angles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            imgs = imgs.to(device)
            angles = torch.tensor(angles, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
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
            train_loss += loss.item() * imgs.size(0)
            train_true.append(sincos_to_angle(angles))
            train_pred.append(sincos_to_angle(outputs).detach())

            # pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
            # all_train_true.extend(angles.cpu().numpy())
            # all_train_pred.extend(pred_angles.cpu().detach().numpy())
            # running_loss += loss.item() * images.size(0)
            
        # epoch_train_loss = train_loss / len(train_loader.dataset)
        # epoch_train_mae = angle_mae(train_true, train_pred)

        train_true = torch.cat(train_true)
        train_pred = torch.cat(train_pred)
        train_mae = circular_mae(train_true, train_pred)
        train_loss /= len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        
        # Validation
        model.eval()
        val_loss = 0
        val_true, val_pred = [], []
        
        with torch.no_grad():
            for imgs, angles in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                imgs = imgs.to(device)
                angles = angles.to(device)
                
                # Forward pass
                outputs = model(imgs)
                loss = criterion(outputs, angles)
                
                # Statistics
                val_loss += loss.item() * imgs.size(0)
                val_true.append(sincos_to_angle(angles))
                val_pred.append(sincos_to_angle(outputs))

                # pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
                # running_loss += loss.item() * images.size(0)
                # all_val_true.extend(angles.cpu().numpy())
                # all_val_pred.extend(pred_angles.cpu().numpy())
        
        # epoch_val_loss = val_loss / len(val_loader.dataset)
        # epoch_val_mae = angle_mae(val_true, val_pred)
        
        val_true = torch.cat(val_true)
        val_pred = torch.cat(val_pred)
        val_mae = circular_mae(val_true, val_pred)
        val_loss /= len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Save the best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_weights = model.state_dict()
            print(f"New best model with MAE: {best_mae:.4f}")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping check
        if no_improve >= patience and epoch >= 10:
            print(f"Early stopping at epoch {epoch+1} as validation MAE hasn't improved for {patience} epochs")
            break
    
    print(f"Best validation MAE: {best_mae:.4f}")
    model.load_state_dict(best_weights)
    return model, history

# Train the model and save the best weights
model, history = train_model(model, train_loader, val_loader, criterion, optimizer)
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
    model.eval()
    preds = []

    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc="Predicting validation set"):
            imgs = imgs.to(device)
            preds.extend(torch.round(sincos_to_angle(model(imgs))).cpu().int())

        for imgs, _ in tqdm(test_loader, desc="Predicting test set"):
            imgs = imgs.to(device)
            preds.extend(torch.round(sincos_to_angle(model(imgs))).cpu().int())
    
    result_df = pd.DataFrame({
        'id': range(len(preds)),  # 0 to 737
        'angle': preds  # Initialize with zeros
    })
    result_df.to_csv(os.path.join(output_dir, output_name), index=False)
    print(f"Results saved to {os.path.join(output_dir, output_name)}")
    
    # For validation set
    # val_predictions = []
    # with torch.no_grad():
    #     val_bar = tqdm(val_loader, desc="Predicting validation set")
    #     for images, _ in val_bar:
    #         images = images.to(device)
            
    #         outputs = model(images)
    #         pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
    #         pred_angles = torch.round(pred_angles).clamp(0, 360).int()
            
    #         val_predictions.extend(pred_angles.cpu().numpy().tolist())

    # For test set
    # test_predictions = []
    # with torch.no_grad():
        # test_bar = tqdm(test_loader, desc="Predicting test set")
        # for images, _ in test_bar:
            # images = images.to(device)
            
            # outputs = model(images)
            # pred_angles = sin_cos_to_angle(outputs[:, 0], outputs[:, 1])
            # pred_angles = torch.round(pred_angles).clamp(0, 360).int()
            
            # test_predictions.extend(pred_angles.cpu().numpy().tolist())
    
    # result_df = pd.DataFrame({
    #     'id': range(738),  # 0 to 737
    #     'angle': [0] * 738  # Initialize with zeros
    # })
    # num_val = min(len(val_predictions), 369)
    # num_test = min(len(test_predictions), 369)
    # for i in range(num_val):
    #     result_df.loc[i, 'angle'] = val_predictions[i]
    # for i in range(num_test):
    #     result_df.loc[369 + i, 'angle'] = test_predictions[i]
    
    # # Save results to CSV
    # result_df.to_csv(os.path.join(output_dir, output_name), index=False)
    # print(f"Results saved to {os.path.join(output_dir, output_name)}")

predict_angles()