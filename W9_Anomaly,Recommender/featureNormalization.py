import numpy as np
import math

def normalize(X):
	mus = []
	stds = []
	normX = np.empty([X.shape[0], 1])
	for i in range(X.shape[1]):
		mu, std, Xi = normalizeSingle(X[:, i:i+1])
		mus.append(mu)
		stds.append(std)
		normX = np.concatenate((normX, Xi), axis=1)
	return np.array([mus]).reshape(-1, 1), np.array([stds]).reshape(-1, 1), normX[:,1:]

def normalizeSingle(X):
	mu = X.mean()
	std = X.std()
	return mu, std, (X-mu)/std

def getGaussian(X):
	mus = []
	stds = []
	normX = np.empty([X.shape[0], 1])
	for i in range(X.shape[1]):
		mu, std, Xi = getGaussianSingle(X[:, i:i+1])
		mus.append(mu)
		stds.append(std)
		normX = np.concatenate((normX, Xi), axis=1)
	return np.array([mus]).reshape(-1, 1), np.array([stds]).reshape(-1, 1), normX[:,1:]

def getGaussianSingle(X):
	mu = X.mean()
	std = X.std()
	return mu, std, ( np.exp( -((X-mu)**2) / (2*(std**2)) ) / math.sqrt(2*math.pi*std**2))
