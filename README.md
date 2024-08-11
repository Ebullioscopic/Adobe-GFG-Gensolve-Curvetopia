# CURVETOPIA

**CURVETOPIA** is a versatile toolkit for analyzing 2D shapes, offering features like shape detection, symmetry analysis, curve completion, and shape recognition using deep learning models.

## Overview

CURVETOPIA processes and analyzes 2D shapes represented as polylines. The main functionalities include:

- **Shape Regularization:** Identifying and classifying geometric shapes such as circles, ellipses, rectangles, polygons, and stars.
- **Symmetry Detection:** Detecting reflective and rotational symmetries in shapes.
- **Curve Completion:** Completing missing parts of shapes through interpolation.
- **Shape Recognition:** Training a neural network to recognize and classify shapes from images of their polylines.

## Output

![image](https://github.com/user-attachments/assets/5b528879-bd9f-4c2e-a1f4-252696d9dd22)

![image](https://github.com/user-attachments/assets/71164a12-546e-4e2a-8ca1-f8c2886c0b12)

![image](https://github.com/user-attachments/assets/fd3768e5-4554-4d4e-8c5b-ae77fbd6d09b)


## Directory Structure

```
Adobe-GFG-Gensolve-Curvetopia/
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
   git clone https://github.com/Ebullioscopic/Adobe-GFG-Gensolve-Curvetopia
   cd Adobe-GFG-Gensolve-Curvetopia
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

Here's a condensed version of the code snippets with a sample execution:

---

# **Running on Google Colab: Shape Regularization and Occlusion Handling**

This notebook provides tools for processing, regularizing, and visualizing 2D shapes from CSV files.

### **Code Snippets**

```python
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def read_csv(csv_path):
    return [np.genfromtxt(csv_path, delimiter=',')[np.genfromtxt(csv_path, delimiter=',')[:, 0] == i][:, 1:] for i in np.unique(np.genfromtxt(csv_path, delimiter=',')[:, 0])]

def plot(paths_XYs, colours):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=colours[i % len(colours)], linewidth=2)
    plt.show()

def regularize_circle(points):
    center = np.mean(points, axis=0)
    radius = np.median(np.linalg.norm(points - center, axis=1))
    theta = np.linspace(0, 2 * np.pi, len(points))
    return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))

def regularize_shape(points):
    return regularize_circle(points) if identify_shape(points) == 'circle' else bezier_curve_fit(points)

def identify_shape(points):
    hull = ConvexHull(points)
    compactness = 4 * np.pi * hull.volume / (calculate_perimeter(hull) ** 2)
    return 'circle' if compactness > 0.88 else 'unknown'

def calculate_perimeter(hull):
    return sum(np.linalg.norm(hull.points[hull.vertices[i]] - hull.points[hull.vertices[(i + 1) % len(hull.vertices)]]) for i in range(len(hull.vertices)))

def bezier_curve_fit(points, num_points=100):
    tck, _ = splprep(points.T, s=0)
    x_new, y_new = splev(np.linspace(0, 1, num_points), tck, der=0)
    return np.vstack((x_new, y_new)).T
```

### **Sample Execution**

```python
# Load and process CSV data
colours = ['r', 'g', 'b', 'y']
path_XYs = read_csv('/content/isolated.csv')
plot(path_XYs, colours)

# Regularize and plot shapes
regularized_path_XYs = [regularize_shape(shape) for path in path_XYs for shape in path]
plot(regularized_path_XYs, colours)
```

This compact code reads shape data from a CSV, regularizes it into circles or Bezier curves, and then plots both the original and regularized shapes using Matplotlib.
## Running the Project

1. **Prepare Your Dataset:** Ensure your dataset is in CSV format, with each row representing a point in a shape (X, Y coordinates).
2. **Execute the Scripts:** Run the scripts in the order mentioned above, starting with data processing, then training the model, and finally making predictions.

## Conclusion

**CURVETOPIA** provides a comprehensive toolset for working with 2D shapes, from identifying and regularizing shapes to detecting symmetries, completing curves, and recognizing shapes using deep learning models. It is a solid foundation for research projects, computer vision applications, or any shape analysis tasks.

## Contributors

[<img src="https://github.com/Ebullioscopic.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="Ebullioscopic GitHub"/><br /><sub></sub>](https://github.com/Ebullioscopic/Adobe-GFG-Gensolve-Curvetopia)
[Hariharan Mudaliar](https://github.com/Ebullioscopic)

[<img src="https://github.com/A-Akhil.png" style="width: 60px; height: 60px; border-radius: 50%;" alt="A-Akhil GitHub"/><br /><sub></sub>](https://github.com/Ebullioscopic/Adobe-GFG-Gensolve-Curvetopia)
[A Akhil](https://github.com/A-Akhil)
