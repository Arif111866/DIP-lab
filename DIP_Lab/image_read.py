from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = '512_X_512.bmp'  
image = Image.open(image_path)

image_matrix = np.array(image)
binary_matrix = np.zeros(image_matrix.shape)
cnt = 0
cnt2 =0
for i in range(image_matrix.shape[0]): 
    for j in range(image_matrix.shape[1]):  
        if image_matrix[i, j,0] > 0:
            binary_matrix[i, j] = 1 
            cnt  = cnt +1 
        else:
            binary_matrix[i, j] = 0  
            cnt2 = cnt2 + 1


print(cnt , cnt2)
print("Image Matrix (shape):")
np.set_printoptions(threshold=np.inf)
print(binary_matrix)
print("Shape of the matrix:", image_matrix.shape)


data = [cnt , cnt2] 
t = [0 , 1]
plt.bar(t , data) 
plt.title('Histogram of black and white')

plt.show()
