import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import math
import random
import png

data = loadmat("bird_small.mat")
data = data['A']
X = data.reshape(-3,3)

'''
p = X.reshape(128, 128, 3)
scipy.misc.toimage(p, cmin=0.0, cmax=...).save('test.jpg')
'''

K = 8 # Program only works for K=3 because of visualizeClustered()
iters = 50

def GenerateRandomK(K, X):
	jmax = X.shape[0]
	Mu = np.zeros([1, X.shape[1]])
	#print(Mu.shape, X[0].reshape(1,-1).shape)
	for i in range(K):
		j = random.randint(0,jmax)
		Mu = np.vstack((Mu, X[j].reshape(1,-1)))
	return Mu[1:,:]

def KMeansCost(C, Mu, X):
	# Incomplete function
	MuPerX = [ Mu[i] for i in C ]
	MuPerX = np.array(MuPerX)
	print(MuPerX)

def assignCluster(X):
	global Mu
	dist = np.sum((Mu-X)**2, axis=1)
	return list(dist).index(dist.min())

def getC(X):
	C = []
	for i in range(X.shape[0]):
		C.append(assignCluster(X[i].reshape(1,-1)))
	return np.array(C).reshape(-1,1)

def moveCentroid(C, X):
	global Mu
	jmax = X.shape[0]
	delMu = np.zeros([K, X.shape[1]])
	for i in range(jmax):
		delMu[C[i]] = delMu[C[i]] + X[i]
	centreCount = np.bincount(np.squeeze(np.asarray(C))) #mbst
	'''
	print(mbst)
	mbind = np.nonzero(mbst)[0]
	print(list(zip(mbind, mbst[mbind])))
	'''
	delMu = delMu/centreCount.reshape(-1,1)
	return delMu

def visualizeClustered(C, Mu, X):
	imax = X.shape[0]
	X0 = np.zeros([1,X.shape[1]])
	X1 = np.zeros([1,X.shape[1]])
	X2 = np.zeros([1,X.shape[1]])
	for i in range(imax):
		if(C[i]==0):
			X0 = np.vstack((X0, X[i].reshape(1,-1)))
		if(C[i]==1):
			X1 = np.vstack((X1, X[i].reshape(1,-1)))
		if(C[i]==2):
			X2 = np.vstack((X2, X[i].reshape(1,-1)))
	X0 = X0[1:,:]
	X1 = X1[1:,:]
	X2 = X2[1:,:]
	plt.scatter(X0[:,0:1], X0[:,1:], color='r', marker='x')
	plt.scatter(Mu[0][0], Mu[0][1], color='black', marker='o')

	plt.scatter(X1[:,0:1], X1[:,1:], color='b', marker='x')
	plt.scatter(Mu[1][0], Mu[1][1], color='black', marker='o')

	plt.scatter(X2[:,0:1], X2[:,1:], color='g', marker='x')
	plt.scatter(Mu[2][0], Mu[2][1], color='black', marker='o')

	plt.show()
		

if __name__ == '__main__':
	Mu = GenerateRandomK(K, X)
	for i in range(iters):
		C = getC(X)
		Mu = moveCentroid(C, X)
	C = getC(X)
	for i in range(C.shape[0]):
		X[i] = Mu[C[i]]
	p = X.reshape(128, 128, 3)
	scipy.misc.toimage(p, cmin=0.0, cmax=...).save('8colors.jpg')
