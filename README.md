# CurveShapeNet

**CurveShapeNet** is a versatile toolkit for analyzing 2D shapes, offering features like shape detection, symmetry analysis, curve completion, and shape recognition using deep learning models.

## Overview

CurveShapeNet processes and analyzes 2D shapes represented as polylines. The main functionalities include:

- **Shape Regularization:** Identifying and classifying geometric shapes such as circles, ellipses, rectangles, polygons, and stars.
- **Symmetry Detection:** Detecting reflective and rotational symmetries in shapes.
- **Curve Completion:** Completing missing parts of shapes through interpolation.
- **Shape Recognition:** Training a neural network to recognize and classify shapes from images of their polylines.

## Directory Structure

```
curve_project/
│
├── data/
│   ├── isolated.csv
│   ├── circle.csv
│   ├── ellipse.csv
│   ├── star.csv
│   ├── rectangle.csv
│   ├── polyfon.csv
│   ├── frag0.csv
│   ├── frag1.csv
│   ├── occlusion1.csv
│   ├── occlusion2.csv
│
├── model/
│   └── shape_recognition_model.pth
│
├── scripts/
│   ├── main.py
│   ├── curve_regularization.py
│   ├── symmetry_detection.py
│   ├── curve_completion.py
│   ├── train_model.py
│   └── utils.py
│
└── output/
    └── predictions/
```

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/CurveShapeNet.git
   cd CurveShapeNet
   ```

2. **Install Dependencies:**
   Install all required libraries using pip:
   ```bash
   pip install numpy matplotlib scipy torch torchvision pillow scikit-learn
   ```

3. **Mount Google Drive (Optional, for Colab users):**
   If you're using Google Colab, you can mount your Google Drive to save outputs and access datasets:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage

### 1. Shape Regularization

- **Script:** `curve_regularization.py`
- **Functionality:** Identifies and classifies geometric shapes from a set of points.
- **Run Example:** 
  - Load your dataset, call `identify_shapes`, and get a list of detected shapes.

### 2. Symmetry Detection

- **Script:** `symmetry_detection.py`
- **Functionality:** Detects reflective and rotational symmetries in shapes.
- **Run Example:** 
  - Load your dataset, call `detect_symmetry`, and get symmetry details.

### 3. Curve Completion

- **Script:** `curve_completion.py`
- **Functionality:** Completes missing parts of shapes using interpolation.
- **Run Example:** 
  - Load your dataset, call `complete_curve`, and plot or save completed shapes.

### 4. Shape Recognition Model Training

- **Script:** `train_model.py`
- **Functionality:** Trains a ResNet-18 model to recognize and classify shapes.
- **Run Example:** 
  - Point to your dataset, run the script to start training, and save the model.

### 5. Predicting Shape from New Data

- **Script:** `main.py`
- **Functionality:** Uses the trained model to predict the shape of new data.
- **Run Example:** 
  - Load new data, run the script, and save prediction results as images.

## Running the Project

1. **Prepare Your Dataset:** Ensure your dataset is in CSV format, with each row representing a point in a shape (X, Y coordinates).
2. **Execute the Scripts:** Run the scripts in the order mentioned above, starting with data processing, then training the model, and finally making predictions.

## Conclusion

**CurveShapeNet** provides a comprehensive toolset for working with 2D shapes, from identifying and regularizing shapes to detecting symmetries, completing curves, and recognizing shapes using deep learning models. It is a solid foundation for research projects, computer vision applications, or any shape analysis tasks.
