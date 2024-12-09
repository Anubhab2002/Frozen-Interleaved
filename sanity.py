import os
from PIL import Image, UnidentifiedImageError

# Define the directory containing the images
image_dir = '/teamspace/studios/this_studio/Frozen-Interleaved/frozen/datasets/DialogCC/images/'

# Initialize counters
total_images = 0
readable_images = 0

# Loop through all files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        total_images += 1
        image_path = os.path.join(image_dir, filename)
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verifies if the image file is readable
            readable_images += 1
        except (UnidentifiedImageError, IOError):
            # Unreadable images will be caught here
            print(f"Unreadable image: {filename}")

# Print the result
print(f"Readable images: {readable_images} out of {total_images}")
