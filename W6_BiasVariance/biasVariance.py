import pandas as pd
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.markers
import matplotlib.patches as mpatches
import featureNormalization as fn

polyPower = 8

iters = 3000
lamb = 1
alpha = 0.001
theta = np.ones([polyPower+1, 1])

# Read required data
data = loadmat("ex5data1.mat")
X = data['X']
y = data['y']
mTrain = len(y)

Xval = data['Xval']
yval = data['yval']
mVal = len(yval)

Xtest = data['Xtest']
ytest = data['ytest']
mTest = len(ytest)

def visualizeData(X, y):
	plt.scatter(X, y, color='r', marker='x')
	plt.xlabel('Change in water level (X)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.show()

def polyX(X, p):
	for i in range(2, p+1):
		Xi = X[:,0:1]**i
		X = np.concatenate((X, Xi), axis=1)
	return X

def hypothesis(theta, X):
	X1 = np.insert(X, 0, 1, axis=1)
	return X1.dot(theta)

def getCost(theta, X, y):
	global lamb
	m = len(y)
	hypo = hypothesis(theta, X)
	hmy = hypo - y
	hmy = hmy**2
	return ( np.sum(hmy) + lamb*np.sum(theta[1:]**2))/(2*m)

def getError(theta, X, y):
	hypo = hypothesis(theta, X)
	hmy = hypo - y
	hmy = hmy**2
	return np.sum(hmy)/(2*len(y))

def getGradient(theta, X, y):
	global lamb
	X1 = np.ones([len(y), 1])
	hmy = hypothesis(theta, X) - y
	delTheta = []
	delT0 = np.sum(hmy.T.dot(X1))/len(y)
	delTheta.append(delT0)
	for i in range(1, len(theta)):
		delTi = ( np.sum(hmy.T.dot(X)) + lamb*theta[i] )/len(y)
		delTheta.append(delTi)
	return np.array(delTheta).reshape(len(theta), 1)

def learn(theta, X, y):
	global iters, alpha
	for i in range(iters):
		grad = getGradient(theta, X, y)
		theta = theta - alpha*grad
	return theta

def plotLearnedLine(theta, X, y):
	plt.scatter(X, y, color='r', marker='x')
	plt.plot([i for i in range(-50, 40)], [(theta[0]+theta[1]*i) for i in range(-50, 40)])
	plt.xlabel('Change in water level (X)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.show()

def plotLearnedCurve(theta, X, y):
	global polyPower
	xPlot = np.arange(X[:, 0:1].min(), X[:,0:1].max(), 0.1).reshape(-1, 1)
	yPlot = hypothesis(theta, polyX(xPlot, polyPower))
	plt.scatter(X[:,0:1], y, color='r', marker='x')
	plt.plot(xPlot, yPlot)
	plt.xlabel('Change in water level (X)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.show()
	

def plotLearningCurve(theta, X, v, Xval, yval):
	cvError = []
	trainError = []
	for i in range(len(y)):
		learnedTheta = learn(theta, X[:i,:], y[:i])
		cvError.append(getError(learnedTheta, Xval, yval))
		trainError.append(getError(learnedTheta, X[:i,:], y[:i,:]))

	blue_patch = mpatches.Patch(color='blue', label='Train Error')
	green_patch = mpatches.Patch(color='green', label='Cross Validation Error')
	plt.plot([i+1 for i in range(len(y))], cvError, color='g')
	plt.plot([i+1 for i in range(len(y))], trainError, color='b')
	plt.xlabel('Size of the training set (m)')
	plt.ylabel('Error in prediction')
	plt.legend(handles=[blue_patch, green_patch])
	plt.show()


if __name__ == '__main__':

	# Use theta for p=1
	theta1 = np.ones([2, 1])
	visualizeData(X, y)
	# Learn theta
	learnedTheta = learn(theta1, X, y)
	plotLearnedLine(learnedTheta, X, y)
	# CV vs Train Error based on size of training set
	plotLearningCurve(learnedTheta, X, y, Xval, yval)

	# Obtain Normalized features
	pX = polyX(X, polyPower)
	mus, stds, normX = fn.normalize(pX)
	ymu, ystd, normy = fn.normalizeSingle(y)

	pXval = polyX(Xval, polyPower)
	musval, stdsval, normXval = fn.normalize(pXval)
	ymuval, ystdval, normyval = fn.normalizeSingle(yval)
	
	# Plot learning curve for polynomial linear regression
	plotLearningCurve(theta, normX, normy, normXval, normyval)

	# Learn theta for polynomial LR
	learnedTheta = learn(theta, normX, normy)
	# Visualize learned curve for polynomial LR
	plotLearnedCurve(learnedTheta, normX, normy)
	








