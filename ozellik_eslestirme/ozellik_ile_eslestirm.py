import cv2
import matplotlib.pyplot as plt

cikolatalar = cv2.imread("cikolatalar.jpg", 0)
plt.figure(), plt.imshow(cikolatalar, cmap = "gray"), plt.axis("off")

#aranacak olan goruntu
damak = cv2.imread("damak.jpg", 0)
plt.figure(), plt.imshow(damak, cmap = "gray"), plt.axis("off")

# Eğer damak büyükse, cikolatalara göre yeniden boyutlandır
height, width = cikolatalar.shape
damak_resized = cv2.resize(damak, (width, height))  # şimdi boyutlar eşit

#orb tanimlaici
#kose-kenar gibi nesneye ait ozellikler
orb = cv2.ORB_create()

#anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(damak_resized, None)
kp2, des2 = orb.detectAndCompute(cikolatalar, None)

#bf matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

#noktalari eslestir
matches = bf.match(des1, des2)

#mesafaye gore sirala
matches = sorted(matches, key = lambda x: x.distance)

#eslesen resimleri gorsellestirelim
plt.figure()
img_match = cv2.drawMatches(damak_resized, kp1, cikolatalar, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"), plt.title("orb")


#sift
sift = cv2.xfeatures2d.SIFT_create()

#bf
bf = cv2.BFMatcher()

#anahtar nokta tespiti sift ile
kp1, des1 = sift.detectAndCompute(damak_resized, None)
kp2, des2 = sift.detectAndCompute(cikolatalar, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel = []

for match1, match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        guzel.append([match1])
        
plt.figure()
sift_matches = cv2.drawMatchesKnn(damak_resized, kp1, cikolatalar, kp2, guzel, None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("Sift")
plt.show()











































