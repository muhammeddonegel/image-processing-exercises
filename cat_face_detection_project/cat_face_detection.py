import cv2
import matplotlib.pyplot as plt
import os

files = os.listdir() #reading image from directory
print(files)

img_path_list = [] #empty list for images

for f in files: #add images to empty list
    if f.endswith(".jpg"): #sadece jpg formatindakiler
        img_path_list.append(f) #
        
print(img_path_list)


for j in img_path_list:
    print(j)
    image = cv2.imread(j)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.02, minNeighbors=4)
    #scaleFactor = zoom multiple/coefficient
    
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), 2)
        cv2.putText(image, "kedi {}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
    
 
    cv2.imshow(j, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): 
        cv2.destroyAllWindows()
        continue









