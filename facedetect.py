import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_path = "Ảnh-Thẻ-Vinh.jpg"
img = cv2.imread(img_path)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face = faceCascade.detectMultiScale(imgGray,1.1,4)

for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

cv2.imshow('img',img)
cv2.waitKey(0)
