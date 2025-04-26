"""
Processing step: 3

This script computes the sparsified spectra (O_i in the paper). It must be used on the fft noise residuals data generate by SyntheticImagesAnalysis.
Denoiser link: https://github.com/grip-unina/SyntheticImagesAnalysis/
"""

import pandas as pd
import os
import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2

# Initial settings
size = 222  # Image size
need_resize = 1  # Set to 1 to enable resizing of the images

# Main path for the FFT images to be processed
input_folder = './TestSet_fft/'
path = os.path.join(input_folder, '**', '*.png')  # Support for all subfolders

# Find all images in the input folder (including subfolders)
files = glob(path, recursive=True)

# Destination folder for the O_i images
base_output_folder = "./Processed/"

# Create the base output folder if it does not exist
os.makedirs(base_output_folder, exist_ok=True)

# Function to resize images
def resize_image(img, size):
    img = cv2.resize(img, (size, size))
    return img

# Builds an O_i matrix from the FFT image, keeping only outlier values
def get_top_outlier(matrix):
    quartiles = np.percentile(matrix, [25, 50, 75], axis=0)
    IQR = quartiles[2] - quartiles[0]
    lower_limit = quartiles[0] - 1.5 * IQR
    upper_limit = quartiles[2] + 1.5 * IQR

    outlier_indices = np.where((matrix > upper_limit))
    outlier_matrix = np.zeros_like(matrix)
    outlier_matrix[outlier_indices] = matrix[outlier_indices]
    return outlier_matrix

# Load and process the images
img_cv2_list = []
for file_path in tqdm(files, desc="Loading images"):
    img_cv2 = cv2.imread(file_path)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_cv2_list.append((file_path, img_cv2))  # Also save the path to reconstruct the subfolder structure

# Resize the images if needed
if need_resize == 1:
    for i in tqdm(range(len(img_cv2_list)), desc="Resizing images"):
        file_path, img = img_cv2_list[i]
        img_cv2_list[i] = (file_path, resize_image(img, size))

# Save the processed images
for file_path, img in tqdm(img_cv2_list, desc="Processing images", unit="img"):
    # Compute the O_i matrix containing only the outlier values
    outlier_matrix = get_top_outlier(img)
    
    # Reconstruct the destination path, maintaining the subfolder structure
    relative_path = os.path.relpath(file_path, input_folder)
    output_path = os.path.join(base_output_folder, relative_path)
    output_dir = os.path.dirname(output_path)

    # Create the directory structure if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the O_i image
    mpimg.imsave(output_path, outlier_matrix, cmap='gray')

print("Elaboration finished!")
