import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import featureNormalization as fn

'''
data = loadmat("ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']
'''

def getThreshold(Xval, yval, seed, step):
	thres = 0
	Fscore = 0
	for i in np.arange(0, seed, step):
		Ypred = generateY(Xval, i)
		Fi = getFscore(yval, Ypred)
		if(Fi>Fscore):
			thres=i
			Fscore = Fi
	return thres, Fscore 

def generateY(Xval, seed):
	rows = Xval.shape[0]
	mus, stds, XGaussval = fn.getGaussian(Xval)
	PXval = np.prod(XGaussval, axis=1).reshape(-1,1)
	Ypred = []
	for i in range(rows):
		if(PXval[i]>=seed):
			Ypred.append(0)
		else:
			Ypred.append(1)
	return np.array(Ypred).reshape(-1,1)

def getFscore(yval, Ypred):
	truep, falsep, truen, falsen = 0, 0, 0, 0
	for i in zip(yval, Ypred):
		if(i==(0,0)):
			truen += 1
		if(i==(0,1)):
			falsep += 1
		if(i==(1,0)):
			falsen += 1
		if(i==(1,1)):
			truep += 1
	precision = truep / (truep + falsep + 1)
	recall = truep / (truep + falsen + 1)
	Fscore = (2 * precision * recall) / (precision + recall + 1)
	return Fscore