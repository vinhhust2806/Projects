import cv2
import numpy as np
import matplotlib.pyplot as plt

anh_mc = cv2.imread('pg4.png')
anh_mc = cv2.resize(anh_mc,(964,699))

bg_1 = cv2.imread('background4.png')
bg_1 = cv2.resize(bg_1,(964,699))

mask = anh_mc - bg_1
mask = np.sum(mask,axis=2)/3
mask = mask.astype('uint8')

_, difference_binary = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)

bg_2 = cv2.imread('weather_forecast.jpg')
bg_2 = cv2.resize(bg_2,(964,699))

difference_binary = difference_binary.reshape(699,964,1)

output = np.where(difference_binary==0,bg_2,anh_mc)
cv2.imshow(output)