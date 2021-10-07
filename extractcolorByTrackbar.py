import cv2
import numpy as np

path = '123,jpg'
def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("Hue Min","TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max","TrackBars")
    smin = cv2.getTrackbarPos("Sat Min","TrackBars")
    smax = cv2.getTrackbarPos("Sat Max","TrackBars")
    vmin = cv2.getTrackbarPos("Val Min","TrackBars")
    vmax = cv2.getTrackbarPos("Val Max","TrackBars")
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    mask = cv2.inRange(imgHSV,lower,upper)
    cv2.imshow('img',img)
    cv2.imshow('imgHSV',imgHSV)
    cv2.imshow('mask',mask)
    cv2.waitKey(1)