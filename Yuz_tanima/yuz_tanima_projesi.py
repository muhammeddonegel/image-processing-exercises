import cv2
import matplotlib.pyplot as plt


#ice aktar
q7 = cv2.imread("Q7.jpg", 0)
plt.figure(), plt.imshow(q7, cmap = "gray"), plt.axis("off")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(q7)

for (x, y, w, h) in face_rect:
    cv2.rectangle(q7, (x, y), (x + w, y + h), (255,255,255), 10)
plt.figure(), plt.imshow(q7, cmap = "gray"), plt.axis("off")


#besiktas
bjk = cv2.imread("ilk11.png", 0)
plt.figure(), plt.imshow(bjk, cmap = "gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(bjk, minNeighbors = 7)

for (x, y, w, h) in face_rect:
    cv2.rectangle(bjk, (x, y), (x + w, y + h), (255,255,255), 5)
plt.figure(), plt.imshow(bjk, cmap = "gray"), plt.axis("off")


#video uzerinde yuz tespiti
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
        
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255),3)
            
        cv2.imshow("face detect", frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()












































