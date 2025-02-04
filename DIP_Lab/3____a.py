from PIL import Image
import cv2
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

noisy_image = add_salt_and_pepper_noise(image_matrix, salt_prob=0.05, pepper_prob=0.05)

def plot(x , title , image):
    plt.subplot(2,2,x) 
    plt.title(title)
    plt.imshow(image, cmap='gray')
def psnr(original, filtered):
    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

plot(1 , "original image" , image_matrix)
plot(2 , "noisy image" , noisy_image)

# Apply a 5x5 Average Filter
average_filtered = cv2.blur(noisy_image, (5, 5))

plot(3 , "average filter", average_filtered)
# Apply a 5x5 Median Filter
median_filtered = cv2.medianBlur(noisy_image, 5)
plot(4 , "median filter" , median_filtered)

psnr_avg = psnr(image_matrix, average_filtered)
psnr_median = psnr(image_matrix, median_filtered)

print(f"PSNR of Average Filter: {psnr_avg:.5f} dB")
print(f"PSNR of Median Filter: {psnr_median:.5f} dB")

plt.show()