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
├── EvalMedia/                # Sample images for testing inference
├── results/                  # Training metrics and evaluation plots
├── inference video/           # Sample inference demo video
└── requirements.txt
```

> **Note:** The trained model weights (`models/best.pt`) and the full training dataset are not included in this repository due to file size. See the [Model & Dataset](#model--dataset) section below.

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

Place a trained model at `models/best.pt`, then run inference on the sample images in `EvalMedia/`:

```bash
python main.py
```

This will run detection on each image and display the annotated results one by one.

## Model & Dataset

- **Model weights:** Not included here due to size. [Add your Google Drive / Hugging Face link here if you host it elsewhere.]
- **Dataset:** Trained on a public skin lesion dataset (7 classes, YOLO detection format). [Add a link to the dataset source here if it's public.]

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
