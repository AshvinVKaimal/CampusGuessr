import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

train_dir = 'data/images_train'
val_dir = 'data/images_val'
train_labels = 'data/labels_train.csv'
val_labels = 'data/labels_val.csv'
output_dir = 'output'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 10
lr = 0.001

class ImageData(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = self.labels.iloc[i]
        img_name = os.path.join(self.img_dir, label['filename'])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        latitude = torch.tensor(label['latitude'], dtype=torch.float32)
        longitude = torch.tensor(label['longitude'], dtype=torch.float32)
        angle = torch.tensor(label['angle'], dtype=torch.float32)
        region = torch.tensor(label['Region_ID'] - 1, dtype=torch.long)

        return image, latitude, longitude, angle, region
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.base_model.fc = nn.Identity()
        self.fc_latlong = nn.Linear(2048, 2)
        self.fc_direction = nn.Linear(2048, 1)
        self.fc_region = nn.Linear(2048, 15)

    def forward(self, x):
        features = self.base_model(x)
        latlong = self.fc_latlong(features)
        direction = self.fc_direction(features)
        region = self.fc_region(features)

        return latlong, direction, region
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = ImageData(train_dir, train_labels, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
val_data = ImageData(val_dir, val_labels, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

model = Model().to(device)
criterion_latlong = nn.MSELoss()
criterion_direction = nn.L1Loss()
criterion_region = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for e in range(epochs):
    model.train()
    running_loss = 0.0

    for img, lat, long, angle, region in train_loader:
        img = img.to(device)
        lat = lat.to(device)
        long = long.to(device)
        angle = angle.to(device)
        region = region.to(device)

        optimizer.zero_grad()
        latlong_pred, direction_pred, region_pred = model(img)

        loss_latlong = criterion_latlong(latlong_pred, torch.stack((lat, long), dim=1))
        loss_direction = criterion_direction(direction_pred.squeeze(), angle)
        loss_region = criterion_region(region_pred, region)

        loss = loss_latlong + loss_direction + loss_region
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {e+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

model.eval()

lat_preds = []
long_preds = []
angle_preds = []
region_preds = []

with torch.no_grad():
    for img, _, _, _, _ in val_loader:
        img = img.to(device)
        latlong_pred, direction_pred, region_pred = model(img)

        lat_preds.append(latlong_pred[:, 0].cpu().numpy())
        long_preds.append(latlong_pred[:, 1].cpu().numpy())
        angle_preds.append(direction_pred.squeeze().cpu().numpy())
        region_preds.append(region_pred.argmax(dim=1).cpu().numpy())

lat_preds = np.concatenate(lat_preds).round().astype(int)
long_preds = np.concatenate(long_preds).round().astype(int)
angle_preds = np.concatenate(angle_preds).round().clip(0, 360).astype(int)
region_preds = (np.concatenate(region_preds) + 1).astype(int)

os.makedirs(output_dir, exist_ok=True)

n_val = len(val_data)
n_total = 738

latlong_data = {
    'id': np.arange(n_total),
    'latitude': np.concatenate([lat_preds, np.zeros(n_total - n_val, dtype=int)]),
    'longitude': np.concatenate([long_preds, np.zeros(n_total - n_val, dtype=int)])
}
direction_data = {
    'id': np.arange(n_total),
    'angle': np.concatenate([angle_preds, np.zeros(n_total - n_val, dtype=int)])
}
region_data = {
    'id': np.arange(n_total),
    'Region_ID': np.concatenate([region_preds, np.zeros(n_total - n_val, dtype=int)])
}

pd.DataFrame(latlong_data).to_csv(os.path.join(output_dir, 'latlong.csv'), index=False)
pd.DataFrame(direction_data).to_csv(os.path.join(output_dir, 'direction.csv'), index=False)
pd.DataFrame(region_data).to_csv(os.path.join(output_dir, 'region.csv'), index=False)

print("Predictions saved to output directory.")