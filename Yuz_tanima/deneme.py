import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagesPaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for imagePath in imagesPaths:
        gray_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        face_np = np.array(gray_img, 'uint8')
        
        # extract id from file name
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = detector.detectMultiScale(face_np)
        
        
        for(x,y,w,h) in faces:
            faceSamples.append(face_np[y:y+h, x:x+w])
            ids.append(id)
            
    return faceSamples, ids

faces, ids = getImagesAndLabels('veri')
recognizer.train(faces, np.array(ids))

#Save model
if not os.path.exists('deneme'):
    os.makedirs('deneme')
    
recognizer.write('deneme/deneme.yml')
print("Model basariyla olusturuldu ve kaydedildi.")
