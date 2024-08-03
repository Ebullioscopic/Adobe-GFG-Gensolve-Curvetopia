# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
from utils import read_csv

class PolylineDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def create_data_from_polylines(csv_files, img_size=(64, 64)):
    data = []
    labels = []
    label_mapping = {'circle': 0, 'ellipse': 1, 'rectangle': 2, 'polygon': 3, 'star': 4}

    for csv_file in csv_files:
        path_XYs = read_csv(csv_file)
        for shape in path_XYs:
            fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
            ax.set_xlim(0, img_size[0])
            ax.set_ylim(0, img_size[1])

            for XY in shape:
                if XY.shape[1] < 2:
                    continue
                ax.plot(XY[:, 0], XY[:, 1], 'k', linewidth=2)

            ax.axis('off')
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_size[1], img_size[0], 3)
            data.append(Image.fromarray(image))
            shape_name = os.path.basename(csv_file).split('.')[0]
            labels.append(label_mapping[shape_name])
            plt.close(fig)

    return data, labels

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, patience=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    csv_files = [
        '/content/drive/MyDrive/curve_project/data/circle.csv',
        '/content/drive/MyDrive/curve_project/data/ellipse.csv',
        '/content/drive/MyDrive/curve_project/data/rectangle.csv',
        '/content/drive/MyDrive/curve_project/data/polygon.csv',
        '/content/drive/MyDrive/curve_project/data/star.csv'
    ]

    data, labels = create_data_from_polylines(csv_files, img_size=(64, 64))

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = PolylineDataset(train_data, train_labels, transform=transform)
    val_dataset = PolylineDataset(val_data, val_labels, transform=transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    }

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 shape classes

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, patience=3)

    model_save_path = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    main()
