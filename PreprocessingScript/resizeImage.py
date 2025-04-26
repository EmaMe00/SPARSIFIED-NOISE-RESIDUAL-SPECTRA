"""
Processing step: 1

This script resizes images in a folder. It should be used on the raw image dataset.
"""

import os
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image

# Set image dimensions and number of channels
image_width = 256
image_height = 256
in_channels = 3

# Paths to input and output folders
path_train = "./daResize"
path_output = "./Dresden_resized/"

# Make sure the output directory exists, otherwise create it
os.makedirs(path_output, exist_ok=True)

# Transformations to resize and normalize the data
transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
    # transforms.Grayscale(),  # Uncomment if working with in_channel=1, otherwise keep it commented for in_channel=3
    transforms.ToTensor(),
])

# Load images from the input folder
data = datasets.ImageFolder(root=path_train, transform=transform)

# Get the list of classes (subfolders) from the dataset
classes = data.classes

# Loop through all images in the dataset with a progress bar
for idx, (img_tensor, label) in enumerate(tqdm(data, desc="Saving transformed images")):
    # Convert the image tensor to a PIL object to save it
    img = transforms.ToPILImage()(img_tensor)
    
    # Get the subfolder (class) name for this image
    class_name = classes[label]
    
    # Create the path for the subfolder in the output directory
    class_output_path = os.path.join(path_output, class_name)
    os.makedirs(class_output_path, exist_ok=True)  # Create the subfolder if it doesn't exist
    
    # Build the saving path for the image
    output_path = os.path.join(class_output_path, f"image_{idx}.png")
    
    # Save the transformed image in the corresponding subfolder
    img.save(output_path)

print("Transformation and saving of images completed.")
