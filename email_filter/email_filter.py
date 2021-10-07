import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tensorflow as tf

def read_data(train_file,test_file):
  with open(test_file,'r') as f:
      data = f.readlines()
  test = []
  for i in data:
     i = i.replace('\n','')
     i = int(i)
     test.append(i)
  test = np.array(test)
  train = np.zeros((len(test),2500))
  with open(train_file,'r') as f1:
    data1 = f1.readlines()
    for j in data1:
      j = j.replace('\n','')
      j = j.split(' ')
      train[int(j[0])-1,int(j[1])-1] = int(j[2])
  return (train,test)

train_data,train_label= read_data('train-features-50.txt','train-labels-50.txt')

test_data , test_label = read_data('test-features.txt','test-labels.txt')


model = MultinomialNB()
model.fit(train_data,train_label)
y_pred = model.predict(test_data)
print(accuracy_score(y_pred,test_label))
print(tf.math.confusion_matrix(y_pred,test_label))