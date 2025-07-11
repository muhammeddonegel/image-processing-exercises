import cv2
import os

files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg") or f.endswith(".jpeg"):
        img_path_list.append(f)

print(img_path_list)

# hog tanimlayicisi
hog = cv2.HOGDescriptor()

#tanimlayiciya SVM ekleme
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, weight) = hog.detectMultiScale(image, padding = (8, 8), scale = 1.05)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 3)
        

    cv2.imshow("Yaya: ", image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): 
        cv2.destroyAllWindows()
        continue
