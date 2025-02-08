import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Configuration
DATASET_PATH = "YOUR_WAY_TO_CAPTCHA_FOLDER"
IMG_HEIGHT = 50
IMG_WIDTH = 200
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
symbols = "abcdefghijklmnopqrstuvwxyz0123456789"
num_symbols = len(symbols)
max_captcha_length = 5

def preprocess_image(image_path):
    """Loads, normalizes, and preprocesses an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert colors if the background is dark
    if np.mean(image) < 127:
        image = cv2.bitwise_not(image)

    # Reduce noise with Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Resize and normalize the image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype(np.float32) / 255.0

    return image

class CaptchaDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = preprocess_image(path)

        if image is None:
            return None

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

        # Extract label from filename
        label = os.path.basename(path).split('.')[0]
        target = torch.zeros((max_captcha_length, num_symbols))
        for i, char in enumerate(label):
            if char in symbols:
                target[i, symbols.index(char)] = 1

        return image, target

# Model architecture
class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Output layers (one for each character in the captcha)
        self.branches = nn.ModuleList([
            nn.Linear(256, num_symbols) for _ in range(max_captcha_length)
        ])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        outputs = [branch(x) for branch in self.branches]
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, max_captcha_length, num_symbols)

# Prepare data
all_images = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)]
train_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)

train_dataset = CaptchaDataset(train_files)
test_dataset = CaptchaDataset(test_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = CaptchaModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model():
    """Training function."""
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = 0
            for i in range(max_captcha_length):
                loss += criterion(outputs[:, i, :], labels[:, i, :].argmax(dim=1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}')

def test_model():
    """Testing function."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.argmax(dim=2).cpu().numpy()

            outputs = model(images).cpu().numpy()
            predicted = outputs.argmax(axis=2)

            total += labels.shape[0]
            correct += (predicted == labels).all(axis=1).sum()

    print(f'Test Accuracy: {correct / total:.4f}')

def predict_captcha(image_path):
    """Function to predict the captcha from an image."""
    image = preprocess_image(image_path)

    if image is None:
        return "Error loading file."

    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image).cpu().numpy()

    captcha = ''.join(symbols[np.argmax(output[0, i, :])] for i in range(max_captcha_length))
    return captcha

# Train the model
train_model()

# Test the model
test_model()

# Example of using the model for prediction
captcha_prediction = predict_captcha("YOUR_WAY_TO_CAPTCHA_SAMPLE_IMAGE")
print(f"Predicted CAPTCHA: {captcha_prediction}")

