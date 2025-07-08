import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

image = cv2.imread("dogAndCat.jpg")

result = model(image)

annotated = result[0].plot()

# Görüntüyü göster
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)

while True:
    
    success,frame = cap.read()
    
    if not success:
        break
    
    results = model(frame)
    annotated = results[0].plot()
    
    cv2.imshow("video", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

        