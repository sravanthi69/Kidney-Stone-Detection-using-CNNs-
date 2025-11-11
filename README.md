# Kidney-Stone-Detection-using-CNNs-
This project focuses on detecting Kidney (Renal) Stones from abdominal CT scan images using Deep Learning based Convolutional Neural Networks (CNNs). This is an image classification problem where images are categorized into two classes – **Kidney_Stone** and **Normal**.

---

## 1. Gathering Data
As this project deals with medical image classification, the dataset was obtained from an open-source GitHub repository.

**Dataset Source:**  
https://github.com/yildirimozal/Kidney_stone_detection/tree/main/Dataset

The dataset contains two folders:
- `Kidney_Stone/`
- `Normal/`

These folder names act as the class labels for model training.

---

## 2. Data Pre-Processing
Since CT scan images come in different resolutions and formats, we standardize them before training.

Steps performed:
- Resized all CT images to a fixed dimension (128 × 128 pixels for consistency)
- Converted images to grayscale or stacked them into 3-channel arrays for CNN input
- Converted each processed image into a NumPy array and assigned its corresponding class label
- Split the dataset into 90 % training and 10 % testing subsets
---

## 3. Image Processing
Medical CT images may contain noise. Improper quality affects feature extraction and model understanding. To enhance image quality, below operations were applied:

| Technique | Explanation |
|----------|-------------|
| Median Blur | Removes salt & pepper noise using median filtering |
| Histogram Equalization (CLAHE) | Improves contrast for better feature extraction |
| Power Law / Gamma Correction | Enhances brightness/contrast adaptively |

## 4. Applied Models

### Model 1 – Custom CNN
A custom CNN was designed using 2D Convolution layers with ReLU activation, pooling layers, fully connected dense layers and sigmoid output neuron for binary classification.

### Model 2 – Transfer Learning (VGG16)
A VGG16 model pre-trained on ImageNet was used with frozen base layers, and a custom top classifier was added. This improves feature representation and supports faster convergence with better generalization.

Both models were trained separately and their performances compared.

---

## 5. Evaluation Output
Both notebooks generate the following:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Training Accuracy vs Validation Accuracy comparison


