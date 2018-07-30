import numpy as np
from scipy.stats import logistic
import math

def sigmoid(val):
	#print(val)
	#return 1/(1+np.exp(-double(val)))
	return logistic.cdf(val)

def sigmoid_vector(vec):
	sig_v = np.vectorize(sigmoid)
	return sig_v(vec)

def sigmoid_matrix(matrix):
	sm = np.vectorize(sigmoid)
	return sm(matrix)