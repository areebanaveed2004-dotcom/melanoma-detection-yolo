# Skin Cancer Detection via Deep Learning

An AI-powered skin cancer detection system that uses deep learning object detection (YOLO) to identify and classify skin lesions from images, aimed at supporting early diagnosis.

## Overview

Skin cancer is among the most common cancers worldwide, and early detection significantly improves treatment outcomes. This project trains a YOLO-based object detection model to detect and classify 7 types of skin lesions from images.

**Lesion classes detected:**
- Melanoma
- Nevus
- BCC (Basal Cell Carcinoma)
- AK (Actinic Keratosis)
- BKL (Benign Keratosis-like Lesions)
- DF (Dermatofibroma)
- VASC (Vascular Lesions)

## Project Structure

```
.
├── main.py                  # Run inference on sample images
├── src/
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── train.py             # Model training script
│   ├── predict_pipeline.py  # Prediction pipeline
│   ├── model_arch.py        # Model architecture definition
│   └── utils.py             # Helper utilities
├── models/
│   ├── best.pt               # Best-performing model weights (used for inference)
│   └── last.pt                # Final-epoch checkpoint (includes optimizer state, for resuming training)
├── EvalMedia/                # Sample images for testing inference
├── results/                  # Training metrics and evaluation plots
├── inference video           # Sample inference demo video
└── requirements.txt
```

> **Note:** The full training dataset is not included in this repository due to file size, but both trained model weights (`best.pt` and `last.pt`) are included.

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The trained model (`models/best.pt`) is already included, so you can run inference right away on the sample images in `EvalMedia/`:

```bash
python main.py
```

This will run detection on each image and display the annotated results one by one.

## Model & Dataset

**Model files (`models/` folder):**
- **`best.pt`** — The model weights from the training epoch with the best validation performance. This is the file loaded by `main.py` for inference/deployment.
- **`last.pt`** — A full checkpoint from the final training epoch (165). Besides model weights, it also stores the optimizer state, EMA weights, and training metadata, so it's mainly useful for resuming training from where it left off, rather than for direct inference.

**Dataset:**
This project was trained on the **ISIC 2018 (International Skin Imaging Collaboration) dataset**, a widely-used public benchmark dataset for skin lesion analysis, reformatted into YOLO detection format with 7 lesion classes.

**Training config:** 165 epochs, 800x800 image size, batch size 16, YOLO architecture.

## Results

Training and evaluation metrics are available in the `results/` folder, including:
- Precision-Recall curve
- Confusion matrix (raw and normalized)
- F1-confidence curve
- Overall training results

## Tech Stack

- Python, PyTorch, Ultralytics YOLO
- OpenCV, Albumentations (data augmentation)
- NumPy, Pandas, Matplotlib, Seaborn

## Author

Areeba Naveed
