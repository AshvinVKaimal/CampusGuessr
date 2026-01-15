# Region ID Prediction

This code performs image classification using a fine-tuned YOLO11n classification model, which is a lightweight convolutional neural network (CNN) pre-trained on ImageNet. The training phase sets hyperparameters such as image size (640) and batch size (16), and saves model checkpoints every epoch.

After training, it evaluates the model on the validation set and makes predictions for the test set. The validation labels are read from a CSV file and compared to predicted labels for performance checking. Predictions are adjusted by remapping YOLO’s class indices (e.g., classes 2–7 shifted to 10–15) to align with the Region_ID format.

There’s no explicit image pre-processing in the script as YOLO internally handles resizing and normalization during inference. 