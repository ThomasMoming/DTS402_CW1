# DTS402 – Machine Learning
## Individual Coursework – Part B: Ensemble Learning with VGG Features

### Student Name: Fanqi Sun
### Student ID: 2575821

## 1. Introduction

This project implements the complete workflow required for **Part B** of the DTS402TC Individual Coursework.  
The tasks include:

- Downloading and preprocessing Tiny ImageNet
- Randomly selecting 10 classes and sampling exactly 1000 training and 200 validation images
- Resizing and normalising images using VGG ImageNet preprocessing
- Extracting three VGG16 feature views (Conv3, Conv4, AvgPool)
- Applying **PCA** dimensionality reduction to reduce high-dimensional features
- Training Random Forest classifiers on both original and PCA-reduced features
- Building two ensembles:
  - **Ensemble A** — same data, different feature views
  - **Ensemble B** — bootstrap + random feature views
- Running ablation experiments for number of trees
- Reporting accuracy, F1 scores, confusion matrices, and analysis

All required outputs are included in the accompanying **report.pdf**.

## 2. File Structure

- code
  - PartB.ipynb 
  - features/ # Cached VGG features (.pt)
    - train_conv3.pt
    - train_conv4.pt
    - train_avgpool.pt
    - val_conv3.pt
    - val_conv4.pt
    - val_avgpool.pt
- report.pdf 
- README.md 

## 3. Environment Requirements

### Python version: Python >= 3.8

### Required libraries:
- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- Pillow 

## 4. Running Instructions

### Google Colab

1. Upload the entire folder to Google Drive.
2. Open **PartB.ipynb** in Google Colab.
3. Ensure that the working directory paths match on your own drive
4. Run ALL cells from top to bottom.
5. All outputs (plots, confusion matrices, tables) will be generated automatically.

### google drive data folder link: https://drive.google.com/drive/folders/1izvQ5QV2eESCwn9DRgve5d803JGM4wwp?usp=sharing

## 5. Workflow 

### **B.1 – Data Loading & Curation**
- Random seed = 42
- Random selection of 10 classes
- Sampling 1000 train + 200 val images
- Apply resize + VGG normalization
- Visualize sample images and class counts

### **B.2 – VGG Feature Extraction**
- Load pretrained VGG16 (ImageNet)
- Freeze all parameters
- Extract:
  - Conv3 → 256 dimensions  
  - Conv4 → 512 dimensions  
  - AvgPool → 25,088 dimensions

### **PCA Dimensionality Reduction**
- Conv3 → reduced to 64 dims
- Conv4 → reduced to 128 dims
- AvgPool → reduced to 512 dims
- PCA applied using scikit-learn

### **B.3 – Ensemble A**
- Train RF on Conv3, Conv4, AvgPool
- Probability averaging
- Report accuracy + confusion matrix

### **B.4 – Ensemble B**
- Bootstrap sampling for each tree
- Random feature view selection per tree
- Probability averaging
- Report accuracy and per-class F1

### **B.5 – Comparative Analysis**
- A vs B comparison (accuracy/F1)
- Ablation study for different number of trees
- Compact results table


## 5. Github backup
- https://github.com/ThomasMoming/DTS402_CW1





















