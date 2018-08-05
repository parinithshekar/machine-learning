import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat

data = loadmat("ex6data1.mat")
X = data['X']
y = data['y']

theta = np.ones([X.shape[1]+1, 1])
iters = 1000
alpha = 0.0003
C = 1
sigmaSq = 1
L = X

# Obtain positive and negative examples
X0 = np.zeros([1, X.shape[1]])
X1 = np.zeros([1, X.shape[1]])
for i in range(len(y)):
	if y[i]==1:
		X1 = np.concatenate((X1, X[i].reshape(1,2)), axis=0)
	else:
		X0 = np.concatenate((X0, X[i].reshape(1,2)), axis=0)
X1 = X1[1:, :]
X0 = X0[1:, :]

# Generate multiclass y
multi_y = []
for i in y:
	if i==1:
		multi_y.append([1, 0])
	else:
		multi_y.append([0, 1])
multi_y = np.array(multi_y)


def visualizeData(X1, X0):
	plt.scatter(X1[:, 0:1], X1[:, 1:], color='g', marker='o')
	plt.scatter(X0[:, 0:1], X0[:, 1:], color='r', marker='x')
	plt.xlabel("x1")
	plt.ylabel("x2")
	greenPatch = mpatches.Patch(color='g', label='Positive')
	redPatch = mpatches.Patch(color='r', label='Negative')
	plt.legend(handles=[greenPatch, redPatch])
	plt.show()

def hypothesis(theta, X):
	X = np.insert(X, 0, 1, axis=1)
	return X.dot(theta)

def cost10(x):
	if x >= 1:
		cost1 = 0
	else:
		cost1 = 1-x
	if x <= -1:
		cost0 = 0
	else:
		cost0 = x+1
	return cost1, cost0

def getCost(theta, X, y):
	global C, multi_y
	vec_cost = np.vectorize(cost10)
	cost = np.array(vec_cost(hypothesis(theta, X)))
	cost = np.concatenate((cost[0].reshape(len(cost[0]), 1), cost[1].reshape(len(cost[1]), 1)), axis=1)
	return (C*(cost*multi_y).sum() + (theta**2).sum()/2)

def generateGaussianFeature(X):
	# Gaussian Kernel
	global sigmaSq
	global L
	f = np.exp(-np.sum((L-X)**2, axis=1)/2*sigmaSq)
	return f.reshape(1,-1)

def getFeatureMatrix(X):
	f = np.zeros([1, X.shape[0]])
	for i in range(X.shape[0]):
		k = generateGaussianFeature(X[i])
		f = np.vstack((f, k))
	f = f[1:, :]
	return f


if __name__ == '__main__':
	print("Pliss go sksvm.py xD")


