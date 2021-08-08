import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dulieu=pd.read_csv('chovay.csv').values

x1=dulieu[:,0].reshape(-1,1)
x2=dulieu[:,1].reshape(-1,1)
y=dulieu[:,2].reshape(-1,1)
plt.scatter(x1,x2)
plt.xlabel('salary')
plt.ylabel('time')

w=np.array([0,0.1,0.1]).reshape(-1,1)
N=dulieu.shape[0]
a=np.ones((N,1)).reshape(-1,1)
x=np.hstack((a,x1))
x=np.hstack((x,x2))
solan=1000
heso=0.01
cost=np.zeros((1000,1))
for i in range(solan):
    dudoan=1./(1+np.exp(-x.dot(w)))
    r=dudoan-y
    w[0]=w[0]-heso*np.sum(r)
    w[1]=w[1]-heso*np.sum(x[:,1].reshape(-1,1)*r)
    w[2]=w[2]-heso*np.sum(x[:,2].reshape(-1,1)*r)
    cost[i] = -np.sum(np.multiply(y, np.log(dudoan)) + np.multiply(1-y, np.log(1-dudoan)))
    print(cost[i])
t = 0.5
plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/t-1))/w[2]), 'g')
plt.show()

