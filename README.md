# -Product-Classification-Verification-Computer-Vision-Project

# Part A) Product Classification Data
# Dataset Description:

20 different products with training and validation folders.
Training images per product: 6-11.
Validation images per product: 1-3.
# CNN (with augmentation and data generation):

Preprocessing:
Loading Data
Data Augmentation
Combining Original and Augmented Data
Generating new data
Model Description:
Convolutional Neural Network (CNN) using Keras.
Layers: Conv2D, MaxPooling, Flatten, Dense, Dropout, SoftMax.
Techniques: Sparse Categorical Cross-entropy Loss, ReduceLROnPlateau callback.
Training Time: 15m 53.4s
Test Time: 2.6s
Training Accuracy: 98.9%
Validation Accuracy: 94.12%
# ResNet50:

Preprocessing:
Loading Data
Data Augmentation
Train-Test Split
Model Description:
Base Model: ResNet50 (pre-trained on ImageNet).
Custom Layers: Global Average Pooling, Dense, Dropout.
Techniques: Adam Optimizer, Sparse Categorical Crossentropy Loss.
Training Time: 28m 50.5s
Test Time: 3.2s
Training Accuracy: 66.4%
Validation Accuracy: 61.76%
# Custom CNN and ANN:

Preprocessing:
Image Loading
Data Augmentation
Data Split
Class Labeling
Normalization
Model Description:
ANN: Two Dense layers (ReLU), output layer (softmax).
CNN: Conv2D, MaxPooling, Flatten, Dense, Loss Function: Sparse Categorical Cross-entropy.
Training Time: 29.0s
Test Time: 1.9s
Training Accuracy (CNN): 76.4%
Validation Accuracy (CNN): 61.76%
# MobileNet:

Preprocessing:
Image Loading
Data Split
Data Augmentation
Normalization
Model Description:
MobileNet base with additional layers.
Techniques: Transfer Learning, Global Average Pooling, Dropout.
Training Time: 10.1s
Test Time: 2.4s
Training Accuracy: 35.29%
Validation Accuracy: 85.2%
# VGG19:

Preprocessing:
Image Loading
Data Split
Data Augmentation
Normalization
Model Description:
VGG19 base for transfer learning.
Techniques: Transfer Learning, Global Average Pooling, Dropout.
Training Time: 1m 34.7s
Test Time: 2.3s
Training Accuracy: 35.29%
Validation Accuracy: 88.2%
##################################################33
# Part B) Product Verification/Recognition
# Dataset Description:

60 different products with varying images per product.
First 40 products used as training data, remaining 20 for validation.
# Siamese:

Preprocessing:
preprocess_input() method from tensorflow.keras.applications.resnet for ResNet-specific preprocessing.
Model Description:
Siamese network with two identical neural networks.
Training Time: 31m
Test Time: 20s
Training Accuracy: 97.35%
Validation Accuracy: 81.38%
