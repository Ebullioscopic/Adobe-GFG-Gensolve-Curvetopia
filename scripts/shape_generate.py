# Shape Generation

import numpy as np
import csv

def generate_polyline(shape_type, num_points=50):
    if shape_type == 'circle':
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = 10  # fixed radius
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif shape_type == 'ellipse':
        theta = np.linspace(0, 2 * np.pi, num_points)
        a, b = 10, 5  # semi-major and semi-minor axes
        x = a * np.cos(theta)
        y = b * np.sin(theta)
    elif shape_type == 'rectangle':
        x = np.array([0, 10, 10, 0, 0])
        y = np.array([0, 0, 5, 5, 0])
    elif shape_type == 'polygon':
        n_sides = 6  # hexagon
        theta = np.linspace(0, 2 * np.pi, n_sides + 1)
        x = 10 * np.cos(theta)
        y = 10 * np.sin(theta)
    elif shape_type == 'star':
        angles = np.linspace(0, 2 * np.pi, num_points)
        radii = np.where(np.arange(num_points) % 2 == 0, 10, 5)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
    else:
        raise ValueError("Unsupported shape_type. Use 'circle', 'ellipse', 'rectangle', 'polygon', or 'star'.")

    return np.stack([x, y], axis=-1)

def save_polyline_to_csv(polyline, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "X", "Y"])  # Header
        for i, (x, y) in enumerate(polyline):
            writer.writerow([1, x, y])  # ID is kept constant for simplicity

# Generate and save polylines for various shapes
shapes = ['circle', 'ellipse', 'rectangle', 'polygon', 'star']
for shape in shapes:
    polyline = generate_polyline(shape)
    save_polyline_to_csv(polyline, f'/content/drive/MyDrive/curve_project/data/{shape}.csv')
