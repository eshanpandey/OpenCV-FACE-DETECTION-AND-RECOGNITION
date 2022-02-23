import cv2 as cv
import numpy as np
haar_cascade=cv.CascadeClassifier('haar_face.xml')

people=['Ben Affleck','Brad Pitt','Angelina Jolie','Jennifer Aniston','Jennifer Lawrence']


face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(r'E:\work\Python\OPENCV\VERIFY\5.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('img',img)

#DETECT FACES
faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]

    label, confidence=face_recognizer.predict(faces_roi)
    print(f'label={people[label]},with a confidence of ={confidence}')

    cv.putText(img, str(people[label]),(20,20),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('Detected Face',img)
cv.waitKey(0)