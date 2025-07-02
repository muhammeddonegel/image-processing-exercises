import cv2
import matplotlib.pyplot as plt

#import image
image = cv2.imread("yayalar.jpg", 0)
cv2.imshow("Original", image)

#detecting edges
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edge detection", edges)

#import haar cascade to face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#doing face detection and then visualisation result
face_rect = face_cascade.detectMultiScale(image)
for (x, y, w, h) in face_rect:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
cv2.imshow("Face detection", image)


# Calling human detection algoritm and set svm
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())   
 
rects, weights = hog.detectMultiScale(image, padding=(8,8), scale=1.01)
for (xA, yA, xB, yB) in face_rect:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0,0,255), 2)
cv2.imshow("Human detection", image)

