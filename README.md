# CNN/Deep Learning Based Automated Class Attendance System

**Team:** Quoc Hung Le, Hassan, Khoa

## Quick Start - Classification (Milestone 1)

### Prerequisites

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Data Preprocessing

```bash
# Launch Jupyter notebook
jupyter notebook

# Download data
Download zip file for all via clicking Download from https://drive.google.com/drive/folders/1eZE1CZG9d6RfkAMmHSeuF4QSLXk4HpvZ
Unzip file at outside of workspace, it should be named 'Discriminative Project Milstone_1'

# Open and run:
notebooks/01_data_preprocessing.ipynb
```

**Outputs:**
- `data/processed/single_objects/train/` - 2,871 images
- `data/processed/single_objects/val/` - 611 images
- `data/processed/single_objects/test/` - 626 images
- `data/class_mapping.json` - Class to index mapping

### Step 2: Train Classification Models

```bash
# Open and run training notebook
notebooks/03_train_classification.ipynb
```

**Models trained:**
- Custom CNN (from scratch)
- ResNet50 (transfer learning)
- EfficientNet-B0 (transfer learning)
- MobileNetV2 (transfer learning)

**Training time:** ~80 mins on MacBook M1, Should look output first

**Outputs:**
- `models/classification/custom_cnn_best.h5`
- `models/classification/resnet50_best.h5`
- `models/classification/efficientnet_best.h5`
- `models/classification/mobilenet_best.h5`

### Step 3: Evaluate Models

```bash
# Open and run evaluation notebook
notebooks/04_evaluate_classification.ipynb
```

**Metrics generated:**
- Accuracy, precision, recall, F1-score
- Confusion matrices
- ROC curves
- Per-class accuracy analysis
- Inference speed benchmarks

**Outputs:**
- `results/classification/metrics/` - JSON files with metrics
- `results/classification/confusion_matrices/` - PNG visualizations
- `results/classification/model_comparison.xlsx` - Excel summary

### Step 4: Run Web Application

```bash
# Launch Streamlit app
streamlit run app/app.py
```

**Features:**
- Single object classification with model selection
- Multi-object detection (after Milestone 2)
- Performance comparison dashboard

**Access:** Open browser to `http://localhost:8501`

### Expected Results (Milestone 1)

| Model | Accuracy | Inference Time | Model Size |
|-------|----------|----------------|------------|
| MobileNetV2 | 99.36% | 5.84 ms | 25.94 MB |
| Custom CNN | 85.94% | 21.10 ms | 19.51 MB |
| EfficientNet | 6.87% | 28.67 ms | 33.96 MB |
| ResNet50 | 3.19% | 15.55 ms | 269.28 MB |

**Best Model:** MobileNetV2 (recommended for deployment)


## Project Overview

Build a discriminative deep learning system that can:
1. Identify single object images using CNN models (Milestone 1 ✓)
2. Detect and locate multiple objects using YOLOv8 (Milestone 2 - upcoming)

## Dataset Statistics

**Total:** 4,108 images across 39 object classes

| Split | Images | Per Class (mean ± std) |
|-------|--------|------------------------|
| Train | 2,871 | 73.6 ± 9.2 |
| Val | 611 | 15.7 ± 2.0 |
| Test | 626 | 16.1 ± 2.1 |

**Quality Metrics:**
- Pass rate: 97.9% (3,329/3,402 images)
- Main issue: Blur (95.9% of failures)
- Brightness: mean=123.5, std=27.3
- Sharpness: mean=1040.2, std=1152.2


## Project Structure

```
CNN_Attendance_System/
├── data/
│   ├── raw/                          # Original images by object
│   ├── processed/
│   │   └── single_objects_enhanced/  # Enhanced preprocessed images
│   │       ├── all_preprocessed/
│   │       ├── train/                # 2,871 training images
│   │       ├── val/                  # 611 validation images
│   │       └── test/                 # 626 test images
│   ├── statistics/                   # Quality reports, split info
│   └── class_mapping_enhanced.json   # Class to index mapping
│
├── models/
│   └── classification/               # Trained model weights (.h5)
│       ├── custom_cnn_best.h5
│       ├── resnet50_best.h5
│       ├── efficientnet_best.h5
│       └── mobilenet_best.h5
│
├── results/
│   └── classification/
│       ├── metrics/                  # Performance JSON files
│       ├── confusion_matrices/       # Confusion matrix visualizations
│       ├── roc_curves/               # ROC curve plots
│       └── model_comparison.xlsx     # Excel comparison table
│
├── notebooks/
│   ├── 01_data_preprocessing_enhanced.ipynb  # Enhanced preprocessing
│   ├── 03_train_classification.ipynb         # Model training
│   └── 04_evaluate_classification.ipynb      # Model evaluation
│
├── app/
│   └── app.py                        # Streamlit web application
│
├── scripts/
│   └── training/                     # Training utility scripts
│
└── requirements.txt                  # Python dependencies
```


## Detailed Usage Guide

### Data Preprocessing Options

**Basic preprocessing (original):**
```bash
notebooks/01_data_preprocessing.ipynb
```

**Enhanced preprocessing (recommended):**
```bash
notebooks/01_data_preprocessing_enhanced.ipynb
```

**Enhancements include:**
- Quality filtering (brightness, sharpness, entropy)
- CLAHE for lighting normalization
- Edge enhancement for boundary clarity
- Conservative augmentation for class balancing


### Multi-Object Detection (Milestone 2)

Coming soon - will include:

```bash
# Generate multi-object images
notebooks/02_multi_object_generation.ipynb

# Train YOLOv8 models
notebooks/05_train_yolo.ipynb

# Evaluate detection performance
notebooks/06_evaluate_yolo.ipynb
```


## Key Dependencies

```
tensorflow==2.15.0
opencv-python==4.8.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
streamlit==1.29.0
ultralytics==8.0.0  # For YOLOv8
jupyter==1.0.0
```


## Training Configuration

### Custom CNN
- Architecture: 4 conv blocks + 2 FC layers
- Parameters: ~19M
- Learning rate: 0.001 (initial)
- Batch size: 32
- Epochs: 50 (early stopping patience=12)
- Optimizer: Adam

### MobileNetV2 (Best Model)
- Base: ImageNet pretrained
- Unfrozen layers: Top 20
- Learning rate: 0.0003
- Batch size: 32
- Epochs: 50 (early stopping patience=15)
- Optimizer: Adam


## Troubleshooting

**Issue:** Models fail to load
```bash
# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# Reinstall if needed
pip install tensorflow==2.15.0
```

**Issue:** Preprocessing fails on M1 Mac
```bash
# Install ARM-compatible OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

**Issue:** Streamlit app won't start
```bash
# Check port availability
lsof -i :8501

# Use different port
streamlit run app/app.py --server.port 8502
```

**Issue:** Out of memory during training
```bash
# Reduce batch size in training notebook
# Change: BATCH_SIZE = 32
# To: BATCH_SIZE = 16
```


## Performance Optimization Tips

**For faster training:**
1. Use GPU if available (CUDA setup)
2. Reduce image size to 128x128 (trade accuracy for speed)
3. Use MobileNetV2 or Custom CNN (faster than ResNet50)

**For better accuracy:**
1. Use enhanced preprocessing pipeline
2. Increase training epochs (50 → 100)
3. Try ensemble methods (combine multiple models)
4. Experiment with different augmentation strategies


## Success Criteria

**Milestone 1 (Single-Object Classification):**
- [x] Dataset: 4,108 images, 39 classes
- [x] Preprocessing: Quality filtering + enhancement
- [x] Models: 4 architectures trained
- [x] Accuracy: >90% achieved (99.36% with MobileNetV2)
- [x] Evaluation: Complete metrics and visualizations

**Milestone 2 (Multi-Object Detection):**
- [ ] Multi-object dataset: 1,200+ images
- [ ] YOLOv8 models: Train n, s, m variants
- [ ] Detection mAP50: >0.85
- [ ] Application: Streamlit demo functional
- [ ] Report: 12-15 pages complete


## Citation

If you use this project or methodology, please cite:

```
IE 7615 Discriminative Deep Learning Project
CNN-Based Automated Class Attendance System
Team: Quoc Hung Le, Hassan, Khoa
Northeastern University, February 2026
```


## License

This project is for educational purposes as part of IE 7615 coursework.
**Course:** IE 7615 - Discriminative Deep Learning  
**Institution:** Northeastern University  
**Semester:** Spring 2026
