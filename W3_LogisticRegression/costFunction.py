import numpy as np
import math
import pandas as pd
import sigmoid as sg
'''
theta = np.array([0.1, 0.2, 0.15])
x = np.array([[8, 7], [9, 4]])
y = np.array([1, 0])
'''
def TX(theta, X):
	X = np.insert(X, 0, 1, axis=1) # Add x0
	return(X.dot(theta.T))

def cost(theta, X, Y):
	# Obtain x.dot(theta.T)
	tx = TX(theta, X)
	# Obtain hypothesis for all values of tx
	h = sg.sigmoid_vector(tx)
	om = np.vectorize(one_minus)
	omh = om(h)
	# Obtain log of h and i-h
	logh = np.log(h)
	logomh = np.log(omh)
	# calculate cost function
	co = np.vectorize(cost_one)
	res = co(logh, logomh, Y)
	res_sum = np.sum(res)
	return res_sum/len(Y)

def cost_one(h, omh, y):
	return (-y*h - (1-y)*omh)

def one_minus(k):
	return 1-k

def gradient(theta, X, Y):
	global hmy_sum
	tx = TX(theta, X)
	h = sg.sigmoid_vector(tx)
	X = np.insert(X, 0, 1, axis=1) # Add x0
	gradient_vector = X.T.dot(h-Y)
	return gradient_vector
