import numpy as np
import cv2

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# Load the image
image = cv2.imread('sample_image.JPG', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
mean = 0
std_dev = 25
noisy_image = add_gaussian_noise(image, mean, std_dev)

# Apply Gaussian smoothing filter to denoise the image
kernel_size = 5
sigma = 1
denoised_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigmaX=sigma)

# Display the original image, noisy image, and denoised image
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
