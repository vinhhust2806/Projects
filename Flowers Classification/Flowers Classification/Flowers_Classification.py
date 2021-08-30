import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers
import cv2
import pathlib
import glob

path0=glob.glob('datasets/*/daisy/*.jpg')
path1=glob.glob('datasets/*/dandelion/*.jpg')
path2=glob.glob('datasets/*/roses/*.jpg')
path3=glob.glob('datasets/*/sunflowers/*.jpg')
path4=glob.glob('datasets/*/tulips/*.jpg')
path=[path0,path1,path2,path3,path4]
x=[]
y=[]

for i in range(5):
    for j in path[i]:
        image=cv2.imread(j)
        image=cv2.resize(image,(180,180))
        x.append(image)
        y.append(i)

x=np.array(x)
y=np.array(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0)

xtrainscale=xtrain/255
xtestscale=xtest/255

data_augumentation=models.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal',input_shape=(180,180,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
])

model=models.Sequential([
    data_augumentation,
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(5,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

h=model.fit(xtrainscale,ytrain,epochs=30)

plt.figure()
plt.plot(h.history['loss'],label='loss')
plt.plot(h.history['accuracy'],label='accuracy')
plt.xlabel('epochs')
plt.ylabel('loss||accuracy')
plt.title('loss and accuracy')
plt.legend()
plt.show()

ypredict=[np.argmax(i) for i in model.predict(xtestscale)]
tf.math.confusion_matrix(ytest,ypredict)
model.evaluate(xtestscale,ytest)
