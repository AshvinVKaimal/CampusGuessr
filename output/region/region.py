import pandas as pd
import os
from ultralytics import YOLO
from pathlib import Path

model = YOLO('yolo11n-cls.pt')

model.train(
    data='dataset',
    task='classify',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project='output/region',
    name='run',
    exist_ok=False,
    save_period=0,
)

val_imgs_dir = 'data/images_val'
val_labels_path = 'data/labels_val.csv'
test_imgs_dir = 'data/images_test'

df = pd.DataFrame({
    'id': range(738),  # 0 to 737
    'Region_ID': [0] * 738  # Initialize with zeros
})

def remap_region_id(r_id):
    pred = r_id + 1
    if pred >= 2 and pred <= 7:
        pred += 8
    elif pred >= 8:
        pred -= 6
    return pred

val_df = pd.read_csv(val_labels_path)
val_imgs = val_df['filename'].tolist()
val_paths = [os.path.join(val_imgs_dir, img) for img in val_imgs]
val_preds = model(val_paths, verbose=False)

for i, res in enumerate(val_preds):
    pred = remap_region_id(int(res.probs.top1))
    df.at[i, 'Region_ID'] = pred
    # print(f"{i} - {val_imgs[i]}")
    # print(f"Predicted Region_ID = {pred}")
    # print(f"Real Region_ID = {val_df['Region_ID'][i]}")

test_dir = Path(test_imgs_dir)
test_imgs = sorted([f.name for f in test_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg', '.JPEG']])
test_paths = [os.path.join(test_imgs_dir, img) for img in test_imgs]
test_preds = model(test_paths, verbose=False)

for i, res in enumerate(test_preds):
    pred = remap_region_id(int(res.probs.top1))
    df.at[i + 369, 'Region_ID'] = pred

df.to_csv('output/region/results.csv', index=False)