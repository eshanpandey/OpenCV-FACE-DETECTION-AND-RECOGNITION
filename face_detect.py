import imp
import cv2 as cv
import numpy as np

img=cv.imread('Photos/group.jpg')
cv.imshow('group',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray group',gray)

haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)

print (f'number of faces found = {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected faces',img)

cv.waitKey(0)