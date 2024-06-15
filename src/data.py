from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random

class GridDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        print(f"Initializing GridDataset with image_dir: {image_dir} and mask_dir: {mask_dir}")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        print(f"Length of dataset: {len(self.images)}")
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_mask.png'))

        print(f"Loading image: {img_path}, mask: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_train_loader(subset=False):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])
    # Data path
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # Now you can use data_dir to access files within data directory
    image_path = os.path.join(data_dir, 'images')
    mask_path = os.path.join(data_dir, 'masks')

    train_dataset = GridDataset(image_path, mask_path, transform=transform)

    if subset:
        # Create a random subset of indices
        random.seed(42)  # Set seed for reproducibility
        subset_size = 100
        indices = random.sample(range(len(train_dataset)), subset_size)

        # Create a new dataset with subset of indices
        train_dataset = [train_dataset[i] for i in indices]

    # Assuming GridDataset is defined as before
    print("Creating train dataset and loader...")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print("Train loader created successfully.")

    return train_loader

