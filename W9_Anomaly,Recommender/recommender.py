import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat("ex8_movies.mat")
Y = data['Y']
R = data['R']
nu = Y.shape[1]
nm = Y.shape[0]
nf = 100

# Unused
'''
paramData = loadmat("ex8_movieParams.mat")
X = paramData['X']
Theta = paramData['Theta']
num_users = paramData['num_users'][0][0]
num_movies = paramData['num_movies'][0][0]
num_features = paramData['num_features'][0][0]
'''

Theta1 = np.random.rand(nu, nf) 
X1 = np.random.rand(nm, nf)
lamb = 10
alpha = 0.0003
iters = 200

def getXGrad(Theta, X, Y, R):
	global lamb
	Ypred = (X.dot(Theta.T) - Y)*R
	grad = Ypred.dot(Theta) + lamb*X
	return grad

def getThetaGrad(Theta, X, Y, R):
	global lamb
	Ypred = (X.dot(Theta.T) - Y)*R
	grad = Ypred.T.dot(X) + lamb*Theta
	return grad


def getCost(Theta, X, Y, R):
	global lamb
	Ypred = X.dot(Theta.T)
	Ypred = Ypred * R
	CostMatrix = ((Ypred - Y)**2)/2
	return CostMatrix.sum() + (lamb/2)*((X**2).sum() + (Theta**2).sum())


if __name__ == '__main__':
	
	xaxis = [i for i in range(iters)]
	J = []
	for i in range(iters):
		X1 = X1 - alpha*(getXGrad(Theta1, X1, Y, R))
		Theta1 = Theta1 - alpha*(getThetaGrad(Theta1, X1, Y, R))
		J.append(getCost(Theta1, X1, Y, R))
	Ypred = X1.dot(Theta1.T)
	Error = np.sum((Y - Ypred*R)**2)
	
	#print(R)
	Recommend = np.abs(R-1)/255

	print('Top Movie recoomendation(index) for {} users'.format(nu))
	print(np.argmax(Ypred * Recommend, axis=0))
	print('LOL Sorry :P')
	#print(Recommend)
	#print(Error)
	#Yperff = X.dot(Theta.T)
	#print(Y[0][0], Ypred[0][0], Yperff[0][0])

	plt.plot(xaxis, J)
	plt.title('Cost over iterations')
	plt.xlabel('Iteration number')
	plt.ylabel('Cost')
	plt.show()

