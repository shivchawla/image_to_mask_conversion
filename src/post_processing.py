import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from model import UNet, EdgeDetectionModel
from data import GridDataset 
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
in_channels = 3
out_channels = 1

# Data path
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Now you can use data_dir to access files within data directory
image_path = os.path.join(data_dir, 'images')
mask_path = os.path.join(data_dir, 'masks')
model_path = os.path.join(data_dir, 'models', 'unet_model.pth')
print(model_path)

# Load the model
model = EdgeDetectionModel(in_channels=in_channels, out_channels=out_channels).to(device)

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
test_dataset = GridDataset(image_path, mask_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Now you can use data_dir to access files within data directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'masks'))
os.makedirs(output_dir, exist_ok=True)

# Inference and saving masks
with torch.no_grad():
    for idx, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Invert the grayscale mask
        # inverted_mask = cv2.btiw(preds[idx, 0])

        # Save the mask
        fName = os.path.join(output_dir, f'output_{idx}.png')
        cv2.imwrite(fName, preds[0, 0])
