# symmetry_detection.py

import numpy as np
# from utils import read_csv, plot

def detect_symmetry(path_XYs):
    symmetries = []
    for shape in path_XYs:
        if has_reflection_symmetry(shape):
            symmetries.append('Reflection Symmetry')
        elif has_rotational_symmetry(shape):
            symmetries.append('Rotational Symmetry')
        else:
            symmetries.append('No Symmetry')
    return symmetries

def has_reflection_symmetry(XYs):
    for angle in np.linspace(0, np.pi, 180):
        rotated_XYs = rotate_points(XYs, angle)
        if np.allclose(rotated_XYs, np.flip(rotated_XYs, axis=0)):
            return True
    return False

def has_rotational_symmetry(XYs):
    center = np.mean(XYs, axis=0)
    for angle in np.linspace(0, 2 * np.pi, 360):
        rotated_XYs = rotate_points(XYs - center, angle) + center
        if np.allclose(XYs, rotated_XYs):
            return True
    return False

def rotate_points(XYs, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return XYs.dot(rotation_matrix)

# if __name__ == "__main__":
#     path_XYs = read_csv('/content/drive/MyDrive/curve_project/data/frag0.csv')
#     symmetries = detect_symmetry(path_XYs)
#     print(symmetries)
#     plot(path_XYs, title='Symmetry Detection')


