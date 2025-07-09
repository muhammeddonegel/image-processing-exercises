import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# from PIL import Image

# to make rgba 4
# def make_white_transparent(input_path, output_path):
#     img = Image.open(input_path).convert("RGBA")
#     new_data = []

#     for pixel in img.getdata():
#         # RGB'si beyaz olanları saydamlaştır
#         if pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200:
#             new_data.append((255, 255, 255, 0))
#         else:
#             new_data.append(pixel)

#     img.putdata(new_data)
#     img.save(output_path)
#     print(f"✔ Şeffaf hale getirildi: {output_path}")

# # Kullanım:
# make_white_transparent("Resources/Ball1.png", "Resources/Ball_clean.png")
# make_white_transparent("Resources/bat1.png", "Resources/bat_clean.png")


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


# import all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/GameOver.png")
imgBall = cv2.imread("Resources/Ball_clean.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat_clean.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat_clean1.png", cv2.IMREAD_UNCHANGED)

# img = Image.open("Resources/bat_clean.png")
# img.show()

# print("Bat1:", imgBat1.shape)
# print("Bat2:", imgBat2.shape)


# print("Ball shape:", imgBall.shape)  # (yükseklik, genişlik, kanal sayısı)
# print("Top boyutu:", imgBall.shape)



imgBackground = cv2.resize(imgBackground, (1280, 720))
imgGameOver = cv2.resize(imgGameOver, (1280, 720))


# Hel detector
detector = HandDetector(detectionCon = 0.8, maxHands = 2)


# Variables
ballPos = [100,100]
speedX = 15
speedY = 15
gameOver = False
score = [0,0]

while True:
    
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()
    
    # find hand and bookmarks
    hands, img = detector.findHands(img, flipType = False)

    #
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)
    
    # cheack the hands
    if hands and not gameOver:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            
            if hand['type'] == "Left":
                
                img = cvzone.overlayPNG(img, imgBat1, (25, y1))
                
                if 25 < ballPos[0] < 25 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1
                    
            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1220, y1))
                    
                if 1220 - w1 < ballPos[0] <1220 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30   
                    score[1] += 1
                        

    # Game Over
    if not gameOver and (ballPos[0] < 10 or ballPos[0] > 1265):
        gameOver = True
        speedX = 0
        speedY = 0
        
    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
    
    # move ball, if game not over yet
    else:
        
        # move ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY
            
        ballPos[0] += speedX
        ballPos[1] += speedY
            
        # Draw ball
        if 0 <= ballPos[0] <= 1280 and 0 <= ballPos[1] <= 720:
            img = cvzone.overlayPNG(img, imgBall, ballPos)
            
        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
            
        img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("r"):
        ballPos = [100,100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0,0]
        imgGameOver = cv2.imread("Resources/gameOver.png")
            
    elif key == ord("q"):break
        
cap.release()
cv2.destroyAllWindows()


            
    
    
    
            
            
    





























