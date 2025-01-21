from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



image_path = '512_X_512.bmp'  
image = Image.open(image_path)

# Convert to grayscale explicitly if it's not already
if image.mode != 'L':
    image = image.convert('L')  # Convert to grayscale (L mode is grayscale in PIL)

# Convert the image to a numpy array
image_matrix = np.array(image)

# Display the image in grayscale
plt.subplot(3,2,1)
plt.imshow(image_matrix, cmap='gray')



w = 512
h = 512
new_w = int(w/2)
new_h = h//2
image_2 = image.resize((new_w , new_h))
new_image_matrix = np.array(image_2)

plt.subplot(3,2,2)
plt.imshow(new_image_matrix , cmap='gray')


reduced_image_matrix = (image_matrix >> 1) << 1
reduced_image = Image.fromarray(reduced_image_matrix)
plt.subplot(3,2,5)
plt.imshow(reduced_image_matrix , cmap='gray')


w , h= image.size
new_w = int(w/4)  
new_h = int(h/4) 
image_2 = image.resize((new_w , new_h))
new_image_matrix = np.array(image_2)
plt.subplot(3,2,3)
plt.imshow(new_image_matrix , cmap='gray')


w , h= image.size
new_w = int(w/8) 
new_h = int(h/8)
image_2 = image.resize((new_w , new_h))
new_image_matrix = np.array(image_2)
plt.subplot(3,2,4)
plt.imshow(new_image_matrix , cmap='gray')


# Hide axis and display
# plt.axis('off')
plt.show()


