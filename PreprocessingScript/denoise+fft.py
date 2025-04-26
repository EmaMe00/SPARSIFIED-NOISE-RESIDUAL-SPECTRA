"""
Processing step: 2

This script must be in the main folder of the denoiser: SyntheticImagesAnalysis 
Denoiser link: https://github.com/grip-unina/SyntheticImagesAnalysis/

This script gets the raw images dataset as input, denoise them and compute the noise residual fft.
In the end it saves all fft noise residuals in the output folder.
"""

import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar
from denoiser import get_denoiser

# Settings
Zhang = 1  # Set to 1 if using Zhang's denoiser
size = 222
need_resize = 1

# Function to compute FFT and residuals
def get_fft_and_residual(files, files_denoised, dataset_name):    
    img_cv2_list = []
    for file_path in files:
        img_cv2 = cv2.imread(file_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        if Zhang == 1:
            img_cv2 = img_cv2[17:-17, 17:-17] 
        img_cv2_list.append(img_cv2)

    img_cv2_list_denoised = []
    for file_path in files_denoised:
        img_cv2 = cv2.imread(file_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_cv2_list_denoised.append(img_cv2)  

    index = 100000
    img_cv2_list_denoised = img_cv2_list_denoised[:index]
    img_cv2_list = img_cv2_list[:index]
    print("Number of images considered: " + str(len(img_cv2_list)) + " for dataset " + dataset_name)

    if need_resize == 1:
        for i in range(len(img_cv2_list)):
            img_cv2_list[i] = resize_image(img_cv2_list[i], size)
            img_cv2_list_denoised[i] = resize_image(img_cv2_list_denoised[i], size)

    residuals_list = []
    for i in range(len(img_cv2_list)):
        residual = img_cv2_list[i] - img_cv2_list_denoised[i]
        residuals_list.append(residual)
    
    list_residual = []
    list_residual_fft = []
    for elem in residuals_list:
        result = np.mean(elem, -1)
        min_value = result.min()
        max_value = result.max()
        normalized_result = (result - min_value) / (max_value - min_value)
        list_residual.append(normalized_result)

        fft_result = np.fft.fft2(result, axes=(0, 1), norm='ortho')
        fft_shifted = np.fft.fftshift(fft_result, axes=(0, 1))
        spectrum = 20 * np.log(np.abs(fft_shifted) + 1)

        min_value = spectrum.min()
        max_value = spectrum.max()
        normalized_spectrum = (spectrum - min_value) / (max_value - min_value)
        list_residual_fft.append(normalized_spectrum)
    
    return img_cv2_list, list_residual, list_residual_fft

# Function to resize an image
def resize_image(img, size):
    img = cv2.resize(img, (size, size))
    return img

# Load an image
def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB')) / 256.0

# Paths
input_folder = './file_da_elaborare/TestSet_daelab/'

output_folder_denoised = './TestSet_denoised/'
output_folder_residual = './TestSet_residual/'
output_folder_fft = './TestSet_fft/'

# Find all images in input folder (including subfolders)
image_files = glob(os.path.join(input_folder, '**', '*.png'), recursive=True)

# Create output directories if they don't exist
os.makedirs(output_folder_denoised, exist_ok=True)
os.makedirs(output_folder_residual, exist_ok=True)
os.makedirs(output_folder_fft, exist_ok=True)

# Initialize denoiser
denoiser = get_denoiser(sigma=1, cuda=False)

# Sort files lexicographically
image_files.sort()

# Lists to store outputs
image_files_denoised = []
residual_images = []

# Process all images
for idx, image_file in enumerate(tqdm(image_files, desc="Processing")):

    # Load original image
    original_img = imread(image_file)
    if original_img is None:
        print(f"Error: could not load image {image_file}.")
        continue

    # Apply denoising
    try: 
        denoised_img = denoiser.denoise(original_img)
    except Exception as e:
        print("Failed to denoise image: " + image_file)
        continue

    # Compute residual
    try: 
        residual_img = denoiser(original_img)
        residual_images.append(residual_img)
    except Exception as e:
        print("Failed to compute residual for image: " + image_file)
        continue

    # Generate file name
    file_name = f"{idx:06}"  # Files named like '000000', '000001', etc.

    # Create output directory structure
    relative_path = os.path.relpath(image_file, input_folder)
    relative_dir = os.path.dirname(relative_path)
    output_dir_denoised = os.path.join(output_folder_denoised, relative_dir)
    output_dir_residual = os.path.join(output_folder_residual, relative_dir)
    output_dir_fft = os.path.join(output_folder_fft, relative_dir)

    os.makedirs(output_dir_denoised, exist_ok=True)
    os.makedirs(output_dir_residual, exist_ok=True)
    os.makedirs(output_dir_fft, exist_ok=True)

    # Save denoised image
    output_path_denoised = os.path.join(output_dir_denoised, f"{file_name}_denoised.png")
    denoised_img_pil = Image.fromarray((denoised_img * 256).astype(np.uint8))
    denoised_img_pil.save(output_path_denoised)
    image_files_denoised.append(output_path_denoised)

    # Save residual image
    output_path_residual = os.path.join(output_dir_residual, f"{file_name}_residual.png")
    residual_img_pil = Image.fromarray((residual_img * 1.5).astype(np.uint8))
    residual_img_pil.save(output_path_residual)

# Get FFT of residuals
_, _, list_residual_fft = get_fft_and_residual(image_files, image_files_denoised, "dataset")

# Save FFT images
for idx, fft_img in enumerate(list_residual_fft):
    file_name = f"{idx:06}_fft.png"
    
    relative_path = os.path.relpath(image_files[idx], input_folder)
    relative_dir = os.path.dirname(relative_path)
    
    output_dir_fft = os.path.join(output_folder_fft, relative_dir)
    os.makedirs(output_dir_fft, exist_ok=True)
    
    output_path_fft = os.path.join(output_dir_fft, file_name)
    fft_img_pil = Image.fromarray((fft_img * 256).astype(np.uint8))
    fft_img_pil.save(output_path_fft)
