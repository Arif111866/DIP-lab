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
print(image_matrix)
plt.subplot(4,3,1)
plt.title("original")
plt.imshow(image_matrix, cmap='gray')

sum_of_all = np.zeros((512, 512), dtype=int)


def ok(bitNumber , plotNumber):
    tem = np.copy(image_matrix)
    x = (1 << bitNumber) 
    print(x) ;
    for i in range(tem.shape[0]):  
        for j in range(tem.shape[0]): 
            tem[i][j] = (image_matrix[i][j] & x)
    plt.subplot(3,3,plotNumber)
    plt.title("MSB-" + str(bitNumber))
    plt.imshow(tem, cmap='gray')

bitNumber = 7
plotNumber = 2
while(plotNumber <= 9):
    ok(bitNumber , plotNumber)
    bitNumber -= 1 
    plotNumber += 1

plt.show()
