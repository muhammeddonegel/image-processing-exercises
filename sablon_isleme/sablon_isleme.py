import cv2
import matplotlib.pyplot as plt

#template matching: sablon esleme

img = cv2.imread("kedi.jpg", 0)
print(img.shape)

template = cv2.imread("kedi_kafa.jpg", 0)
print(template.shape)
h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    
    method = eval(meth) # cv2.TM_CCOEFF --> 
    
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in[cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    img_copy = img.copy()
    
    cv2.rectangle(img_copy, top_left, bottom_right, 255,2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eslesen Sonuc"), plt.axis("off")
    plt.subplot(122), plt.imshow(img_copy, cmap = "gray")
    plt.title("Tespit edilen Sonuc"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()






















