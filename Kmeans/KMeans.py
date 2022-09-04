import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random


means = [[2,2],[8,3],[3,6]]
cov = [[1,0],[0,1]]
N = 500
x0 = np.random.multivariate_normal(means[0],cov,N)
x1 = np.random.multivariate_normal(means[1],cov,N)
x2 = np.random.multivariate_normal(means[2],cov,N)

X = np.vstack((x0,x1,x2))
K = 3 
original_label = np.array([0]*N+[1]*N+[2]*N).T

def centroid_init(X,K):
  return X[np.random.choice(X.shape[0],K,replace=False)]

def label_assign(X,centroids):
  D = cdist(X,centroids)
  return np.argmin(D,axis=1)

def converge(centroids,new_centroids):
  return set([tuple(a) for a in centroids]) == set([tuple(b) for b in new_centroids])

def new_centroids(X,centroids,label,k):
  new_centroids = np.zeros((k,2))
  for i in range(k):
    new_centroids[i,:] = np.mean(X[label==i,:],axis=0)
  return new_centroids

def kmean(X,K):
  centroids = centroid_init(X,K)
  i=0
  while True:
    i = i+1
    label = label_assign(X,centroids)
    newcentroids = new_centroids(X,centroids,label,K)
    if converge(centroids,newcentroids):
      break
    centroids = newcentroids.copy()
  return (centroids,label,i)

p = kmean(X,K)
print(p)
for i in range(len(X)):
  if p[1][i] == 0:
    plt.scatter(X[i,0],X[i,1],color = 'r',marker = '*')
  elif p[1][i] == 1:
    plt.scatter(X[i,0],X[i,1],color = 'b',marker = '+')
  else:
    plt.scatter(X[i,0],X[i,1],color = 'g',marker = '*')
plt.show()
    