# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:16:29 2024

@author: yisss
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# def calculate_snr(original_image, noisy_image):
#     original_image = original_image.astype(np.float32)
#     noisy_image = noisy_image.astype(np.float32)
    
#     # Calculate signal (original image mean)
#     signal_mean = np.mean(original_image)
    
#     # Calculate noise (difference between original and noisy images)
#     noise = original_image - noisy_image
#     noise_std = np.std(noise)
    
#     # Calculate SNR
#     snr = signal_mean / noise_std
    
#     return snr

# image check
image1_path = 'final_images/New/8-bit-256-x-256-Grayscale-Lena-Image.png' # original image
# image2_path = 'final_images/final_image_55dB.png'
image3_path = 'final_images/New/processed_image_35dB_2D.png'

image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

# check, if the images are loaded successfully
if image1 is None or image3 is None:
    print("Error: One or both images not found or unable to load.")
    exit()

# make sure the size of two images are same 
if image1.shape != image3.shape:
    print("Error: Images must have the same dimensions.")
    exit()

# SSIM Calculation
ssim_value, ssim_image = ssim(image1, image3, full=True)
print(f"SSIM: {ssim_value}")




# cv2.imshow("SSIM Image", (ssim_image * 255).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def calculate_snr(original_image, noisy_image):
#     original_image = original_image.astype(np.float32)
#     noisy_image = noisy_image.astype(np.float32)
    
#     # Calculate signal (original image mean)
#     signal_mean = np.mean(original_image)
    
#     # Calculate noise (difference between original and noisy images)
#     noise = original_image - noisy_image
#     noise_std = np.std(noise)
    
#     # Calculate SNR
#     snr = signal_mean / noise_std
    
#     return snr

# snr = calculate_snr(image1, image3)
# print(f"SNR:{snr}")