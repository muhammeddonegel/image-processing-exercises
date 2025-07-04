import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization  
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pickle

path = "myData"

images = []
classNo = []
class_name = []

folder_list = os.listdir(path)

#benzersiz meyve isimlerini topla
label_dict = {}
label_index = 0

for folder in folder_list:
    folder_path = os.path.join(path, folder)
    #print(folder)
  
    label_name = folder.split()[0]
    #print(label_name)
    
    if label_name not in label_dict:
        label_dict[label_name] = label_index
        class_name.append(label_name)
        label_index += 1

        
    label = label_dict[label_name]
    
    image_list = os.listdir(folder_path)
    
    for img_file in image_list[::3]:
        
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        
        if img is not None:
            images.append(img)
            classNo.append(label)
            

#print("Toplam sınıf sayısı:", len(label_dict))
#print("Sınıf etiketleri:", label_dict)

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.25, random_state= 42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.25)

x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))


x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)
x_validation = x_validation.reshape(-1, 32, 32, 1)

dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,)

dataGen.fit(x_train)

y_train = to_categorical(y_train, len(class_name))
y_test = to_categorical(y_test, len(class_name))
y_validation = to_categorical(y_validation, len(class_name))

model = Sequential()

model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (32,32,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))


model.add(Dropout(0.21))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.33))
model.add(Dense(len(class_name), activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer = Adam(), metrics=["accuracy"])

images = np.array(list(map(preProcess, images)))
images =images.reshape(-1, 32, 32, 1)

classNo = to_categorical(classNo, len(class_name))

batch_size = 64

hist = model.fit(dataGen.flow(x_train, y_train, batch_size = batch_size),
                 validation_data = (x_validation, y_validation),
                 epochs = 40, steps_per_epoch = x_train.shape[0] // batch_size, shuffle = 1)

model.save("model_trained.keras")

pickle_out = open("model_trained2.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

















