import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers , datasets , models

# set up data
(xtrain,ytrain),(xtest,ytest)=datasets.cifar10.load_data()
#xval=xtrain[40000:50000,:]
#yval=ytrain[40000:50000]
#xtrain=xtrain[0:50000,:]
#ytrain=ytrain[0:50000]

# reshape data
#xval=xval.reshape(xval.shape[0],32,32,3)
xtrain=xtrain.reshape(xtrain.shape[0],32,32,3).astype('float32')
xtest=xtest.reshape(xtest.shape[0],32,32,3).astype('float32')
xtrain=xtrain/255.0
xtest=xtest/255.0
#ytrain=np_utils.to_categorical(ytrain,10)
#yval=np_utils.to_categorical(yval,10)
#ytest=np_utils.to_categorical(ytest,10)

# model
model=models.Sequential([# cnn
                        layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
                        layers.MaxPooling2D((2,2)),
                        layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                        layers.MaxPooling2D((2,2)),
                        # Dense
                        layers.Flatten(),
                        layers.Dense(128,activation='relu'),
                        layers.Dense(10,activation='softmax')

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
h=model.fit(xtrain,ytrain,epochs=10)

# evalute model
model.evaluate(xtest,ytest)

# simulate
plt.figure()
plt.plot(np.arange(0,10),h.history['loss'],label='loss')
plt.plot(np.arange(0,10),h.history['accuracy'],label='accuracy')
plt.xlabel('epoch')
plt.ylabel('loss||accuracy')
plt.title('accuracy and loss')
plt.legend()
