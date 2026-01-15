import os
import shutil
import pandas as pd

def organize_dataset(image_dir, csv_path, output_base_dir, split):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_filename = row['filename']
        region_id = str(row['Region_ID'])
        
        # Create directory for region if it doesn't exist
        dest_dir = os.path.join(output_base_dir, split, region_id)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Source and destination paths
        src_path = os.path.join(image_dir, img_filename)
        dest_path = os.path.join(dest_dir, img_filename)
        
        # Copy image
        shutil.copy(src_path, dest_path)

# Define paths
output_dir = 'dataset'

# Organize training data
organize_dataset(
    image_dir='data/images_train',
    csv_path='data/labels_train.csv',
    output_base_dir=output_dir,
    split='train'
)

# Organize validation data
organize_dataset(
    image_dir='data/images_val',
    csv_path='data/labels_val.csv',
    output_base_dir=output_dir,
    split='val'
)