import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy as sp
import matplotlib.pyplot as plt
import math
import random
import featureNormalization as fn

K = 1

data = loadmat("ex7data1.mat")
X = data['X']
X1 = X[:, 0:1]
X2 = X[:, 1:]

mus, stds, normX = fn.normalize(X)



CoV = normX.T.dot(normX)/normX.shape[0]

U, S, V = np.linalg.svd(CoV)
#print(U.shape, S.shape, V.shape)

Ured = U[:,:K]

Z = Ured.T.dot(normX.T)

Xapprox = Z.T.dot(Ured.T)

plt.scatter(normX[:,0:1], normX[:,1:], color='b')
plt.scatter(Xapprox, Xapprox, color='r', marker='x')
plt.show()
