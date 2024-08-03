# load_model.py

import torch
from torchvision import models
import torch.nn as nn

def load_model(model_path, num_classes=5):
    # Initialize the model (same architecture as used during training)
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

if __name__ == "__main__":
    model_path = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")