import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create() #create face recognizer
recognizer.read('deneme/deneme.yml') #file to read specified
pathClassifier = 'haarcascade_frontalface_default.xml'

#path to used
faceClassifier = cv2.CascadeClassifier(pathClassifier)
font = cv2.FONT_HERSHEY_SIMPLEX
vid_cam = cv2.VideoCapture(0)

while True:
    
    ret, camera = vid_cam.read()
    gray =cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray, 1.2, 5)
    
    for(x,y,w,h) in faces:
        
        cv2.rectangle(camera, (x-20, y-20), (x+w+20, y+h+20), (0,255,0), 4)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        print(Id)
        
        if(Id == 1):
            Id = "Muhammed"
        
        if(Id == 2):
            Id = "Yilmaz"
            
        if(Id == 3):
            Id = "Eymen"

        if(Id == 4):
            Id = "Ardil"
            
        cv2.rectangle(camera, (x-22, y-90),(x+w+22, y-22), (0,255,0), -1)
        
        cv2.putText(camera, str(Id), (x, y-40), font, 1, (255, 255, 255), 2) # define the name variable to be written
        
    cv2.imshow('camera', camera)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
vid_cam.release()
cv2.destroyAllWindows()
