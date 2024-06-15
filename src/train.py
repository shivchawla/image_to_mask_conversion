import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import UNet
from data import get_train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train UNet model.')
    parser.add_argument('--subset', action='store_true',
                        help='Use a subset of the dataset.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # Get train_loader using the function defined in data.py
    train_loader = get_train_loader(subset=args.subset)

    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        model.train()
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print("Training completed.")

if __name__ == "__main__":
    main()