from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or another interactive backend like 'Qt5Agg' or 'GTK3Agg'
import matplotlib.pyplot as plt


image_path = 'lena.bmp'  
image = Image.open(image_path)

# Convert to grayscale explicitly if it's not already
if image.mode != 'L':
    image = image.convert('L')  # Convert to grayscale (L mode is grayscale in PIL)


image_matrix = np.array(image)

plt.subplot(1,3,1) 
plt.title("original image")
plt.imshow(image_matrix, cmap='gray')


gamma = 1.8
power_law_image = np.power(image_matrix, gamma)

# Inverse Logarithmic Transform
c = 1  # Scaling factor
inverse_log_image = c * (np.exp(image_matrix*3) - 1)

# Display Results




plt.subplot(1, 3, 2)
plt.title(f"Power-Law Transform (Gamma = {gamma})")
plt.imshow(power_law_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Inverse Logarithmic Transform")
plt.imshow(inverse_log_image, cmap='gray')

plt.show()
