import numpy as np
import math

def sigmoid(val):
	return 1/(1+math.exp(-val))

def sigmoid_vector(vec):
	sig_v = np.vectorize(sigmoid)
	return sig_v(vec)

def sigmoid_matrix(matrix):
	sm = np.vectorize(sigmoid)
	return sm(matrix)
