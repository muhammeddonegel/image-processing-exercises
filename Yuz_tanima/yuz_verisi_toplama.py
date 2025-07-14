import cv2 

vid_cam = cv2.VideoCapture(0) #camera describing
# import already face classifier
face_dedector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_name = 2

sayi = 1

while(True):
    
    _, img_frame = vid_cam.read() # reading camera
    gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY) # add gray toning
    faces = face_dedector.detectMultiScale(gray, 1.3, 5) #determine upper and lower  border

    for (x,y,w,h) in faces: #determine values for frama measure
        cv2. rectangle(img_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        sayi += 1
        
        cv2.imwrite("veri/User." + str(face_name) + '.' + str(sayi) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('cerceve', img_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    elif sayi > 100:
        break

vid_cam.release()
cv2.destroyAllWindows()
