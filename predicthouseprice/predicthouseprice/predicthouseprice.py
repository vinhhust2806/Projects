import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('gianha.csv').values
N=data.shape[0]
x=data[:,0].reshape(-1,1)
y=data[:,1].reshape(-1,1)
plt.scatter(x,y)
plt.xlabel('dien tich')
plt.ylabel('gia nha')
x=np.hstack((np.ones((N,1)),x))
print(x)
w=np.array([0.,1.]).reshape(-1,1)
cost=np.zeros((100,1))
solan=100
heso=0.000001
for i in range(solan):
    r=x.dot(w)-y
    cost[i]=0.5*np.sum(r*r)
    w[0]-=heso*np.sum(r)
    w[1]-=heso*np.sum(x[:,1].reshape(-1,1)*r)
    print(cost[i])
predict=x.dot(w)
plt.plot((x[0][1],x[N-1][1]),(predict[0],predict[N-1]),'r')
plt.show()
x1=50
y1=w[0]+x1*w[1]
print('the price of house with 50m2 :',y1)

