# Angle Prediction

This code implements an angle regression model using a fine-tuned ConvNeXt CNN model. It uses a circular loss function to handle angular data, and predicts the sin and cos values of the angles rather than the angles directly.

The data is preprocessed by outlier removal, image resizing, data augmentation, etc. The model is trained using the Adam optimizer and a learning rate scheduler. The training process includes early stopping based on validation loss.

The model is evaluated using the mean absolute error (MAE) metric, and the results are saved in a CSV file. The code also includes functions for visualizing the training process and saving the model checkpoints.

Better performance can be achieved by running the model with more epochs, but due to time and resource constraints, the performance is not optimal.