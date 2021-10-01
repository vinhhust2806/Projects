import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

anh = cv2.VideoCapture(0)
win_name = 'Camera Preview'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
while cv2.waitKey(1) != ord('q'):
    _,frame = anh.read()
    if not _:
        break
    
    frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(frameGray,1.1,4)
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
    cv2.imshow('img',frame)

anh.release()
cv2.destroyAllWindows()
