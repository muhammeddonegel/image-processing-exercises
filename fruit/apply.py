import cv2
import pickle
import numpy as np

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img
    
cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(3,480)

with open("model_trained3.p", "rb") as f:

    model = pickle.load(f)

while True:
    
    success, frame = cap.read()
    
    if not success:
        print("Kameradan görüntü alınamadı.")
        break
    
    img = cv2.resize(frame, (32,32))
    img = preProcess(img)
    img = img.reshape(1,32,32,1)
    
    
    #predict
    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probVal = np.max(predictions)
    
    class_names = ['Apple', 'Apricot', 'Avocado', 'Banana', 'Beans', 'Beetroot',
                   'Blackberrie', 'Blueberry', 'Cabbage', 'Cactus', 'Caju', 'Cantaloupe',
                   'Carambula', 'Carrot', 'Cauliflower', 'Cherimoya', 'Cherry', 'Chestnut',
                   'Clementine', 'Cocos', 'Corn', 'Cucumber', 'Dates', 'Eggplant', 'Fig',
                   'Ginger', 'Gooseberry', 'Granadilla', 'Grape', 'Grapefruit', 'Guava',
                   'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats',
                   'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mangostan', 'Maracuja',
                   'Melon', 'Mulberry', 'Nectarine', 'Nut', 'Onion', 'Orange', 'Papaya',
                   'Passion', 'Peach', 'Pear', 'Pepino', 'Pepper', 'Physalis', 'Pineapple',
                   'Pistachio', 'Pitahaya', 'Plum', 'Pomegranate', 'Pomelo', 'Potato',
                   'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry',
                   'Tamarillo', 'Tangelo', 'Tomato', 'Walnut', 'Watermelon', 'Zucchini']
    
    if probVal > 0.7:
        label = class_names[classIndex]          # İsim olarak al
        cv2.putText(frame, f"{label} ({probVal:.2f})", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
    cv2.imshow("Meyve ismi: ", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()










