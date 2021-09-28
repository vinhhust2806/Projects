import cv2
import glob
import matplotlib.pyplot as plt
import math

image_path = glob.glob(r'boat\*.jpg')
image_path.sort()

images = []
for file_path in image_path:
    anh = cv2.imread(file_path)
    anh = cv2.cvtColor(anh,cv2.COLOR_BGR2RGB)
    images.append(anh)

num_images = len(images)

plt.figure(figsize=[30,10])
num_columns = 3
num_rows = math.ceil(num_images / num_columns)

for i in range(0,num_images):
    plt.subplot(num_rows , num_columns , i+1)
    plt.axis('off')
    plt.imshow(images[i])
    
stitcher = cv2.Stitcher_create()
status,result = stitcher.stitch(images)

if status == 0:
    plt.figure(figsize=[30,10])
    plt.imshow(result)

plt.show()
