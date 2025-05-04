import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm
from PIL import Image
from tqdm import tqdm

train_dir = 'data/images_train'
val_dir = 'data/images_val'
train_labels = 'data/labels_train.csv'
val_labels = 'data/labels_val.csv'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 10
lr = 1e-4

def clean_labels(labels):
    labels = labels[labels['angle'].between(0, 360)]
    labels = labels[labels['Region_ID'].between(1, 15)]
    return labels

class ImageData(Dataset):
    def __init__(self, img_dir, labels, target, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        label = self.labels.iloc[i]
        img_name = os.path.join(self.img_dir, label['filename'])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.target == 'latlong':
            target = torch.tensor([label['latitude'], label['longitude']], dtype=torch.float32)
        elif self.target == 'direction':
            target = torch.tensor(label['angle'], dtype=torch.float32)
        elif self.target == 'region':
            target = torch.tensor(label['Region_ID'] - 1, dtype=torch.long)

        return image, target
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_model(dim):
    model = timm.create_model("convnext_xlarge.fb_in22k_ft_in1k", pretrained=True, num_classes=dim, global_pool='avg')
    return model.to(device)

def train_model(model, dl, criterion, optimizer):
    model.train()
    for e in range(epochs):
        running_loss = 0.0
        for img, labels in tqdm(dl, desc=f'Epoch {e+1}/{epochs}'):
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Loss: {running_loss/len(dl)}')

def predict_model(model, dl):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for img, _ in tqdm(dl, desc='Predicting'):
            img = img.to(device)
            outputs = model(img).cpu()
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            preds.extend(outputs.numpy())

    return preds

train_labels = pd.read_csv(train_labels)
train_labels = clean_labels(train_labels)
val_labels = pd.read_csv(val_labels)
val_labels = clean_labels(val_labels)

latlong_train = ImageData(train_dir, train_labels, target='latlong', transform=transform)
angle_train = ImageData(train_dir, train_labels, target='direction', transform=transform)
region_train = ImageData(train_dir, train_labels, target='region', transform=transform)
latlong_val = ImageData(val_dir, val_labels, target='latlong', transform=transform)
angle_val = ImageData(val_dir, val_labels, target='direction', transform=transform)
region_val = ImageData(val_dir, val_labels, target='region', transform=transform)

latlong_train_loader = DataLoader(latlong_train, batch_size=batch_size, shuffle=True, num_workers=2)
angle_train_loader = DataLoader(angle_train, batch_size=batch_size, shuffle=True, num_workers=2)
region_train_loader = DataLoader(region_train, batch_size=batch_size, shuffle=True, num_workers=2)
latlong_val_loader = DataLoader(latlong_val, batch_size=batch_size, shuffle=False, num_workers=2)
angle_val_loader = DataLoader(angle_val, batch_size=batch_size, shuffle=False, num_workers=2)
region_val_loader = DataLoader(region_val, batch_size=batch_size, shuffle=False, num_workers=2)

latlong_model = build_model(2)
train_model(latlong_model, latlong_train_loader, nn.MSELoss(), optim.Adam(latlong_model.parameters(), lr=lr))
latlong_preds = predict_model(latlong_model, latlong_val_loader)
latlong_data = pd.DataFrame({
    'id': list(range(738)),
    'latitude': [int(pred[0]) for pred in latlong_preds] + [0] * (738 - len(val_labels)),
    'longitude': [int(pred[1]) for pred in latlong_preds] + [0] * (738 - len(val_labels))
})
latlong_data.to_csv(os.path.join(output_dir, 'latlong/2022101015_1.csv'), index=False)
print("Latlong predictions:", latlong_data.head())

angle_model = build_model(1)
train_model(angle_model, angle_train_loader, nn.L1Loss(), optim.Adam(angle_model.parameters(), lr=lr))
angle_preds = predict_model(angle_model, angle_val_loader)
direction_data = pd.DataFrame({
    'id': list(range(738)),
    'angle': [int(pred) % 360 for pred in angle_preds] + [0] * (738 - len(val_labels))
})
direction_data.to_csv(os.path.join(output_dir, 'direction/2022101015_1.csv'), index=False)
print("Direction predictions:", direction_data.head())

region_model = build_model(15)
train_model(region_model, region_train_loader, nn.CrossEntropyLoss(), optim.Adam(region_model.parameters(), lr=lr))
region_preds = predict_model(region_model, region_val_loader)
region_data = pd.DataFrame({
    'id': list(range(738)),
    'Region_ID': [int(np.argmax(pred)) + 1 for pred in region_preds] + [0] * (738 - len(val_labels))
})
region_data.to_csv(os.path.join(output_dir, 'region/2022101015_1.csv'), index=False)
print("Region predictions:", region_data.head())

print("Predictions saved to output directory.")