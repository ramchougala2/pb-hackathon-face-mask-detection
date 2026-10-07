# Face Mask Detection

A real-time face mask detection system built using transfer learning (MobileNetV2) and OpenCV. The model classifies faces as **WithMask** or **WithoutMask** and runs live on a webcam feed with bounding boxes and confidence scores.

## Overview

This project was built as a college course project to demonstrate an end-to-end computer vision pipeline: data preprocessing, transfer learning, model evaluation, and real-time deployment.


## Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model training (transfer learning on MobileNetV2)
- **OpenCV** — face detection (Haar Cascade) and real-time webcam inference
- **scikit-learn** — evaluation metrics (confusion matrix, precision/recall/F1)
- **Google Colab** — training environment (free GPU)

## Dataset

[Face Mask ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) (Kaggle), pre-split into Train (10,000), Validation (800), and Test (992) images across two classes: `WithMask` and `WithoutMask`.

## Model & Approach

- Base model: **MobileNetV2** pretrained on ImageNet, with the top layers frozen
- Custom classification head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(2, softmax)`
- Data augmentation (rotation, zoom, shifts, brightness, horizontal flip) applied to the training set only
- Trained for 10 epochs using the Adam optimizer and categorical cross-entropy loss

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 99.12% |
| Test Accuracy | ~99% |
| Precision (WithMask) | 0.98 |
| Recall (WithMask) | 1.00 |
| Precision (WithoutMask) | 1.00 |
| Recall (WithoutMask) | 0.98 |

See `notebook/training_notebook.ipynb` for the full training run, confusion matrix, and classification report.

## Project Structure

```
mask_detector_project/
├── detect_mask.py              # Real-time webcam detection script
├── mask_detector_model.h5      # Trained model weights
├── requirements.txt            # Python dependencies
├── notebook/
│   └── training_notebook.ipynb # Full training & evaluation notebook
└── README.md
```

## Setup & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/a-sad25/face-mask-detection.git
   cd face-mask-detection
   ```

2. Create and activate a virtual environment (Python 3.11 recommended, since TensorFlow does not yet support the newest Python releases):
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the live detector:
   ```bash
   python detect_mask.py
   ```
   Press `q` to quit the webcam window.

## Limitations

- Uses OpenCV's Haar Cascade for face detection, which can struggle with side profiles, occlusion, or low light — a DNN-based face detector would likely improve robustness.
- Trained on a dataset with mostly front-facing, well-lit faces, so performance may drop in more challenging real-world conditions.
- Binary classification only (mask / no mask) — does not currently detect incorrectly worn masks.

## Future Improvements

- Swap Haar Cascade for a more robust DNN-based face detector (e.g. MTCNN or a pretrained SSD face detector)
- Add a third class for "mask worn incorrectly"
- Deploy as a Streamlit web app for easier sharing/demo access

## Author

[*(Asad Aman)*](https://github.com/a-sad25/)
