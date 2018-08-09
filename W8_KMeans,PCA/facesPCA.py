import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import math
import random
import featureNormalization as fn

K = 100

data = loadmat("ex7faces.mat")
X = data['X']

def visualize(im_ar):
	# Matplotlib be best
	w_h = int(math.sqrt(im_ar.shape[1]))
	for i in range(25):
		plt.subplot(5, 5, i+1)
		plt.imshow(im_ar[i].reshape(w_h, w_h).T, cmap="gray")
		plt.axis('off')
	plt.show()

if __name__ == '__main__':

	# Mean normalize feature-scale and all features
	mus, stds, normX = fn.normalize(X)

	# Calculate CoVariance Matrix
	CoV = normX.T.dot(normX)/normX.shape[0]

	# Singular Value Decomposition
	U, S, V = np.linalg.svd(CoV)
	#print(U.shape, S.shape, V.shape)

	# Select K number of vectors
	Ured = U[:,:K]

	# Project from N-dimensional onto the K-dimensional space
	Z = Ured.T.dot(normX.T)

	# Convert back to N dimensions
	Xapprox = Z.T.dot(Ured.T)
	'''
	w_h = int(math.sqrt(X.shape[1]))
	scipy.misc.toimage(X[0].reshape(w_h, w_h).T).save('original.jpg')
	scipy.misc.toimage(Xapprox[0].reshape(w_h, w_h).T).save('postPCA.jpg')
	'''
	visualize(X)
	visualize(Xapprox)