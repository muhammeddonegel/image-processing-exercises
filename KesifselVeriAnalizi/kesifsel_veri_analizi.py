"""
- veri setini
-veri setini indir
-resim2video
eda->
"""
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

pathIn = r"img1"
pathOut = "deneme2.mp4"

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# img = cv2.imread(pathIn + "//" + files[44])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)

fps = 30
size = (1920, 1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)
for i in files:
    print(i)
    
    filename = pathIn + "//" + i
    
    img = cv2.imread(filename)
    
    out.write(img)
    
out.release()
