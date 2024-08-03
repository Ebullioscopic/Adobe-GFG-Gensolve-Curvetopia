# curve_regularization.py

# from utils import read_csv, plot
import numpy as np
from scipy.spatial import ConvexHull

def identify_shapes(path_XYs):
    regular_shapes = []
    for shape in path_XYs:
        shape = np.array(shape)  # Convert shape to NumPy array for operations
        if is_circle(shape):
            regular_shapes.append('Circle')
        elif is_ellipse(shape):
            regular_shapes.append('Ellipse')
        elif is_rectangle(shape):
            regular_shapes.append('Rectangle')
        elif is_polygon(shape):
            regular_shapes.append('Polygon')
        elif is_star(shape):
            regular_shapes.append('Star')
        else:
            regular_shapes.append('Unknown')
    return regular_shapes

def is_circle(XYs):
    center = np.mean(XYs, axis=0)
    radii = np.linalg.norm(XYs - center, axis=1)
    return np.std(radii) < 0.05 * np.mean(radii)

def is_ellipse(XYs):
    cov = np.cov(XYs.T)
    eigvals, _ = np.linalg.eigh(cov)
    return eigvals[0] / eigvals[1] > 0.5

def is_rectangle(XYs):
    hull = ConvexHull(XYs)
    return len(hull.vertices) == 4

def is_polygon(XYs):
    hull = ConvexHull(XYs)
    return len(hull.vertices) > 4

def is_star(XYs):
    return len(XYs) > 10 and np.mean(np.linalg.norm(XYs - np.mean(XYs, axis=0), axis=1)) > 0.1 * np.max(np.linalg.norm(XYs - np.mean(XYs, axis=0), axis=1))

# Sample usage (adjust the paths and function calls as needed)
# path_XYs = read_csv('/content/drive/MyDrive/curve_project/data/isolated.csv')
# shapes = identify_shapes(path_XYs)
# print(shapes)
# plot(path_XYs, title='Regularized Shapes')

