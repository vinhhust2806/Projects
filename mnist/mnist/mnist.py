import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers,datasets
from keras.utils import np_utils

# set up data
(xtrain,ytrain),(xtest,ytest)=datasets.mnist.load_data()
xval=xtrain[50000:60000,:]
yval=ytrain[50000:60000]
xtrain=xtrain[0:50000,:]
ytrain=ytrain[0:50000]

# reshape data
xtrain=xtrain.reshape(xtrain.shape[0],28,28,1)
xval=xval.reshape(xval.shape[0],28,28,1)
xtest=xtest.reshape(xtest.shape[0],28,28,1)

ytrain=np_utils.to_categorical(ytrain,10)
yval=np_utils.to_categorical(yval,10)
ytest=np_utils.to_categorical(ytest,10)

# model
model=models.Sequential([
                         layers.Conv2D(32,(3,3),activation='sigmoid',input_shape=(28,28,1)),
                         layers.Conv2D(32,(3,3),activation='sigmoid'),
                         layers.MaxPool2D((2,2)),
                         layers.Flatten(),
                         layers.Dense(128,activation='sigmoid'),
                         layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

H=model.fit(xtrain,ytrain,validation_data=(xval,yval),batch_size=32,epochs=10,verbose=1)

#figure
epochs = 10
plt.plot(np.arange(0,epochs),H.history['loss'],label='loss')
plt.plot(np.arange(0,epochs),H.history['accuracy'],label='accuracy')
plt.plot(np.arange(0,epochs),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,epochs),H.history['val_accuracy'],label='val_accuracy')

plt.title('accuracy and loss')
plt.xlabel('epochs')
plt.ylabel('loss||accuracy')
plt.legend()

# evaluate model
score = model.evaluate(xtest,ytest)
print(score)

# predict
i=np.random.randint(0,10000)
plt.imshow(xtest[i].reshape(28,28),cmap='gray')
ypredict=model.predict(xtest[i].reshape(1,28,28,1))
print(np.argmax(ypredict))