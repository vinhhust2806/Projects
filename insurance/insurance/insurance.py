import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('insurance_data.csv')

xtrain,ytrain,xtest,ytest=train_test_split(data[['age','affordibility']],data['bought_insurance'],test_size=0.2,random_state=25)
xtrainscale=xtrain.copy()
xtrainscale['age']=xtrainscale['age']/100
xtestscale=xtest.copy()
xtestscale['age']=xtestscale['age']/100

model=models.Sequential([
                         layers.Dense(1,activation='sigmoid',input_shape=(2,),kernel_initializer='ones',bias_initializer='zeros')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model.fit(xtrainscale,ytrain,epochs=5000)

model.evaluate(xtestscale,ytest)
print(model.predict(xtestscale))
weights,bias=model.get_weights()