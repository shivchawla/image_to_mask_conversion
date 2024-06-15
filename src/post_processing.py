import torch
import numpy as np
import cv2
from model import UNet
from data import get_train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_train_loader(subset=True)

model = UNet(in_channels=3, out_channels=1).to(device)
model.eval()

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = preds.cpu().numpy()
        preds = (preds > 0.5).astype(np.uint8) * 255
        
        for i in range(preds.shape[0]):
            cv2.imwrite(f'output_{i}.png', cv2.bitwise_not(preds[i, 0]))
