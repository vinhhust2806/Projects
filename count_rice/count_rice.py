import cv2 
import numpy as np

anh = cv2.imread('rice.png',0)
anh1 = anh[0:350,:]
anh1[anh1>140] = 255
anh1[anh1<=140] = 0
anh1 = cv2.medianBlur(anh1,5)
anh2 = anh[350:,:]
anh2[anh2>100] = 255
anh2[anh2<=100] = 0
anh2 = cv2.medianBlur(anh2,5)
anh = np.vstack((anh1,anh2))
contours , hierarchy = cv2.findContours(anh,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_L1)
print('so hat gao = ',len(contours))