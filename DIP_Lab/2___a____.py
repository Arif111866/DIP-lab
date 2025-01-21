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
plt.subplot(1,2,1) 
plt.title("original image")
plt.imshow(image_matrix, cmap='gray')


for i in range(len(image_matrix)):  # Rows
    for j in range(len(image_matrix[0])):  # Columns
        if 50 <= image_matrix[i][j] <= 100:
            image_matrix[i][j] = min(image_matrix[i][j] *2 , 255)

plt.subplot(1,2,2) 
plt.title("enhanced image")
plt.imshow(image_matrix, cmap='gray')
plt.show()