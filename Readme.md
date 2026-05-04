# Zind Amini Cocoa Contamination Detection

This project implements a YOLOv8 object detection model to identify cocoa plant diseases as part of the **Zindi Amini Cocoa Contamination Challenge**. The model is designed to detect and classify diseases and contamination in cocoa plants, helping farmers identify and manage crop health issues early.

## About the Project

The Zindi Amini Cocoa Contamination Challenge is focused on leveraging machine learning and computer vision to solve agricultural problems in cocoa production. Cocoa contamination and plant diseases significantly impact crop yield and quality. This project uses state-of-the-art YOLOv8 object detection technology to automatically identify affected areas in cocoa plants.

### Key Features:
- **Real-time Detection**: YOLOv8-based model for fast and accurate disease detection
- **High Performance**: Trained over 60 epochs with excellent convergence
- **Multiple Metrics**: Uses precision, recall, and mAP scores for comprehensive evaluation
- **Practical Application**: Can be deployed for field monitoring and crop management

## Model Performance

The trained model achieves strong performance metrics on the validation dataset:

| Metric | Final Value |
|--------|-------------|
| **Precision** | 75.78% |
| **Recall** | 66.25% |
| **mAP50** | 74.62% |
| **mAP50-95** | 51.67% |
| **Training Epochs** | 60 |

The model shows consistent improvement throughout training with validation metrics stabilizing after epoch 40+.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Zind-Amini-Cocoa-contamination-Detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Requirements

The following packages are required to run this project:

- **ultralytics** - YOLOv8 object detection framework

### Additional Dependencies (Python):
```
ultralytics
opencv-python
numpy
pandas
matplotlib
torch
torchvision
```

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook

Run the `amini-cocoa-contamination-challenge.ipynb` notebook to train the model, evaluate results, or perform inference:
```bash
jupyter notebook amini-cocoa-contamination-challenge.ipynb
```

The notebook includes:
- Data loading and preprocessing
- YOLOv8 model training
- Model evaluation and metrics calculation
- Inference on test images

### Python App

The `app.py` file contains the application code for deployment. You can extend it to:
- Create a web interface for predictions
- Set up API endpoints for model inference
- Build a real-time detection system

Run the application:
```bash
python app.py
```

### Model Inference

The trained model weights are stored in `best.pt`. You can use it for inference with the Ultralytics YOLO library:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Perform inference
results = model.predict(source='image.jpg', conf=0.5)

# Display results
results[0].show()
```

## Project Files

- **amini-cocoa-contamination-challenge.ipynb** - Main Jupyter notebook with complete pipeline
- **app.py** - Flask/FastAPI application for deployment (can be extended)
- **best.pt** - Trained YOLOv8 model weights
- **requirements.txt** - Python dependencies
- **results.csv** - Training metrics across 60 epochs
- **results.png** - Visualization of training results
- **confusion_matrix.png** - Confusion matrix for model evaluation
- **labels.jpg** - Dataset class labels visualization
- **Box*.png** - Performance curves (Precision, Recall, F1)

## Dataset

The project uses the **Zindi Cocoa Dataset** from the Amini Cocoa Contamination Challenge. The dataset includes:
- Training images with annotated bounding boxes
- Validation split for model evaluation
- Test set for final evaluation

The dataset structure is organized as:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   └── val/
└── Test.csv
```

## Results

### Training Results:
- **Total Training Time**: ~9,140 seconds (~2.5 hours)
- **Number of Epochs**: 60
- **Final Precision**: 75.78%
- **Final Recall**: 66.25%
- **Final mAP50**: 74.62%
- **Final mAP50-95**: 51.67%

### Performance Curves:
The project includes several visualization files showing model performance:
- `results.png` - Overall training curves
- `confusion_matrix.png` - Detailed class predictions
- `BoxP_curve.png` - Precision curve
- `BoxR_curve.png` - Recall curve
- `BoxF1_curve.png` - F1-score curve

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. You can:
- Improve model architecture
- Add data augmentation techniques
- Optimize hyperparameters
- Create better preprocessing pipelines
- Build deployment features

## License

This project is provided as part of the Zindi Amini Cocoa Contamination Challenge.

## Author

**Muhammad Qasim Shabbeer**

## Acknowledgments

- Zindi for hosting the Amini Cocoa Contamination Challenge
- Ultralytics for the YOLOv8 framework
- Contributors and participants in the challenge





