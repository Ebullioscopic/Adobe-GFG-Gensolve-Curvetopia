# main.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from load_model import load_model
from utils import read_csv, plot

# Ensure output directories exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Predict the shape using the model
def predict_shape(model, image, device):
    # Transform the image to match the input format of the model
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Get the prediction from the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main():
    # Load the trained model
    model_path = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    model = load_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ensure output directories exist
    ensure_directory_exists('/content/drive/MyDrive/curve_project/output/predictions/')

    # Define the CSV files containing new polylines for prediction
    csv_files = [
        '/content/drive/MyDrive/curve_project/data/circle.csv',
        '/content/drive/MyDrive/curve_project/data/ellipse.csv',
        '/content/drive/MyDrive/curve_project/data/rectangle.csv',
        '/content/drive/MyDrive/curve_project/data/polygon.csv',
        '/content/drive/MyDrive/curve_project/data/star.csv'
    ]

    # Mapping of label indices to shape names
    label_mapping = {0: 'Circle', 1: 'Ellipse', 2: 'Rectangle', 3: 'Polygon', 4: 'Star'}

    for csv_file in csv_files:
        path_XYs = read_csv(csv_file)

        # Plot each shape, make predictions, and save the results
        for i, shape in enumerate(path_XYs):
            fig, ax = plt.subplots()

            # Validate that shape has at least one valid polyline with two columns (X and Y)
            valid_shape = True
            for XY in shape:
                if XY.shape[1] != 2:
                    print(f"Warning: Found XY with shape {XY.shape} in file {csv_file}. Skipping this shape.")
                    valid_shape = False
                    break

            if not valid_shape:
                continue  # Skip this shape if it's not valid

            plot([shape], title=f"Original Shape: {os.path.basename(csv_file)}")

            # Convert plotted shape to an image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image)

            # Predict the shape
            predicted_label = predict_shape(model, image, device)
            predicted_shape_name = label_mapping[predicted_label]
            plt.title(f"Predicted Shape: {predicted_shape_name}")

            # Save the plot with prediction
            output_path = f"/content/drive/MyDrive/curve_project/output/predictions/{os.path.basename(csv_file).replace('.csv', f'_prediction_{i}.png')}"
            plt.savefig(output_path)
            print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    main()
