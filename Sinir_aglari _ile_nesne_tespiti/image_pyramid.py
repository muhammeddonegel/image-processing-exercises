import cv2
import matplotlib.pyplot as plt

def image_pyramid(image, scale = 1.5, minsSize =(224,244)):
    
    yield image
    
    while True:
        w = int(image.shape[1]/scale)
        image = cv2.resize(image, dsize = (w,w))
        
        if image.shape[0] < minsSize[1] or image.shape[1] < minsSize[0]:
            break
        
        yield image

# img = cv2.imread("kedi.jpg")
# im =image_pyramid(img, 1.5, (10,10))

# for i, image in enumerate(im):
#     print(i)
    
#     if i == 10:
#         plt.imshow(image)

























