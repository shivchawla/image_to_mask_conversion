import os
import random
from PIL import Image, ImageDraw

# Function to create a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Function to generate images with random grids
def generate_grid_image(image_size, min_grid_size, max_grid_size, image_path, mask_path, start_index):
    image = Image.new('RGB', (image_size, image_size), color='white')
    mask = Image.new('L', (image_size, image_size), color=0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    y = 0
    while y < image_size:
        grid_height = random.randint(min_grid_size, max_grid_size)
        x = 0
        while x < image_size:
            grid_width = random.randint(min_grid_size, max_grid_size)
            color = random_color()
            draw_image.rectangle([x, y, x + grid_width, y + grid_height], fill=color)
            draw_mask.rectangle([x, y, min(x + grid_width, image_size-1), min(y + grid_height, image_size-1)], outline=255, fill=0)
            x += grid_width
        y += grid_height

    # Save images with unique filenames
    image_filename = f'image_{start_index}.png'
    mask_filename = f'image_{start_index}_mask.png'
    image.save(os.path.join(image_path, image_filename))
    mask.save(os.path.join(mask_path, mask_filename))

# Parameters
image_sizes = [256] #, 512, 768, 1024]  # Sizes of the images (256x256, 512x512, 768x768, 1024x1024)
num_images_per_size = 10  # Number of images to generate for each size
num_images = 2000  # Number of images to generate

# Directory to save images and masks
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Now you can use data_dir to access files within data directory
image_path = os.path.join(data_dir, 'images')
mask_path = os.path.join(data_dir, 'masks')

# Create save directory if it doesn't exist
os.makedirs(image_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

# Loop through different image sizes
for size_index, image_size in enumerate(image_sizes):
    min_grid_size = 20  # Minimum size of the grid squares
    max_grid_size = image_size // 5  # Maximum size of the grid squares relative to image size

    existing_images = [filename for filename in os.listdir(image_path) if filename.endswith('.png')]
    start_index = len(existing_images)

    # Generate and save images and masks
    for i in range(num_images):
        generate_grid_image(image_size, min_grid_size, max_grid_size, image_path, mask_path, start_index + i)

    print(f'Generated {num_images_per_size} images and masks for size {image_size}x{image_size}.')
