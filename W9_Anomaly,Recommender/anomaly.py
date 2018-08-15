import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import featureNormalization as fn
import obtainThreshold as ot

data = loadmat("ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']

def initVisualizaiton(X):
	plt.scatter(X[:, 0:1], X[:, 1:], color='b', marker='x')
	#plt.scatter(center[0], center[1], color='black', marker='o')
	plt.show()

def finalVisualization(normal, anomalies, center):
	plt.scatter(normal[:,0:1], normal[:,1:], color='g', marker='x')
	plt.scatter(anomalies[:,0:1], anomalies[:,1:], color='r', marker='x')
	plt.scatter(center[0], center[1], color='black', marker='o')
	plt.show()

def getSeparation(X, PX, thres):
	rows = X.shape[0]
	normal = np.zeros([1,X.shape[1]])
	anomalies = np.zeros([1,X.shape[1]])
	#print(normal.shape, X[0].shape)
	for i in range(rows):
		if(PX[i][0] >= thres):
			normal = np.concatenate((normal, X[i].reshape(1,-1)), axis=0)
		else:
			anomalies = np.concatenate((anomalies, X[i].reshape(1,-1)), axis=0)
	return normal[1:, :], anomalies[1:, :]


if __name__ == '__main__':
	initVisualizaiton(X)

	mus, stds, XGauss = fn.getGaussian(X)
	vars = stds**2
	#print(XGauss)

	PX = np.prod(XGauss, axis=1).reshape(-1,1)

	center = np.where(np.in1d(PX, PX.max()))[0]
	center = X[center][0]
	
	thres, Fscore = ot.getThreshold(Xval, yval, PX.mean())
	normal, anomalies = getSeparation(X, PX, thres)
	print('thres:', thres, 'Fscore:', Fscore)

	finalVisualization(normal, anomalies, center)






