from PIL import Image
import cv2
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt


# image_path = '512_X_512.bmp'  
image_path = 'lena.bmp'  
image = Image.open(image_path)

# Convert to grayscale explicitly if it's not already
if image.mode != 'L':
    image = image.convert('L')  # Convert to grayscale (L mode is grayscale in PIL)


image_matrix = np.array(image)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):

    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # Add Salt (white pixels)
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255  # White pixels
    
    # Add Pepper (black pixels)
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0  # Black pixels
    
    return noisy_image

noisy = add_salt_and_pepper_noise(image_matrix, salt_prob=0.15, pepper_prob=0.05)

def plot(x , title , image):
    plt.subplot(2,3,x) 
    plt.title(title)
    plt.imshow(image, cmap='gray')
def psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

plot(1 , "original image" , image_matrix)
plot(2 , "noisy image" , noisy)

# Harmonic Mean Filter
def harmonic_mean_filter(image, size=5):
    def harmonic_mean(pixels):
        pixels = np.where(pixels == 0, 1, pixels)  # Avoid division by zero
        return len(pixels) / np.sum(1.0 / pixels)
    
    return scipy.ndimage.generic_filter(image.astype(np.float32), harmonic_mean, size=(size, size))

# Geometric Mean Filter
def geometric_mean_filter(image, size=5):
    def geometric_mean(pixels):
        return np.exp(np.mean(np.log(np.where(pixels > 0, pixels, 1))))  # Avoid log(0)
    
    return scipy.ndimage.generic_filter(image.astype(np.float32), geometric_mean, size=(size, size))

# Apply filters
harmonic_filtered = harmonic_mean_filter(noisy, size=5)
geometric_filtered = geometric_mean_filter(noisy, size=5)

plot(3 , "harmonic filter" , harmonic_filtered)
plot(4 , "geometric filter" , geometric_filtered)

# Compute PSNR values
psnr_harmonic = psnr(image_matrix, harmonic_filtered)
psnr_geometric = psnr(image_matrix, geometric_filtered)

# Print PSNR results
print(f"PSNR after Harmonic Mean Filtering: {psnr_harmonic:.2f} dB")
print(f"PSNR after Geometric Mean Filtering: {psnr_geometric:.2f} dB")

plt.show()