import glob
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
a = glob.glob('fruits/*')

def similarity(data,test):
  simila = []
  data = np.array(data)
  anhtest = cv2.imread(test,0)
  anhtest = cv2.calcHist([anhtest] , [0],
                         None,[250],[0,256])
  for i in data:
    anh=cv2.imread(i,0)
    anh=cv2.resize(anh,(600,600))
    anh = cv2.calcHist([anh] , [0],
                         None,[250],[0,256])
    a = anh.flatten()
    b = anhtest.flatten()
    simila.append(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
  simila = np.argsort(simila)[::-1][0:5]
  return data[simila]
print(similarity(a,'fruits/1 (20).jpg'))