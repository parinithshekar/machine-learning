from sklearn import svm
import numpy as np
import pandas as pd
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mlxtend.plotting import plot_decision_regions

data = loadmat("ex6data1.mat")
X = data['X']
y = data['y']

# Obtain positive and negative examples
X0 = np.zeros([1, X.shape[1]])
X1 = np.zeros([1, X.shape[1]])
for i in range(len(y)):
	if y[i]==1:
		X1 = np.concatenate((X1, X[i].reshape(1,2)), axis=0)
	else:
		X0 = np.concatenate((X0, X[i].reshape(1,2)), axis=0)
X1 = X1[1:, :]
X0 = X0[1:, :]

def visualizeData(X1, X0):
	plt.scatter(X1[:, 0:1], X1[:, 1:], color='g', marker='o')
	plt.scatter(X0[:, 0:1], X0[:, 1:], color='r', marker='x')
	plt.xlabel("x1")
	plt.ylabel("x2")
	greenPatch = mpatches.Patch(color='g', label='Positive')
	redPatch = mpatches.Patch(color='r', label='Negative')
	plt.legend(handles=[greenPatch, redPatch])
	plt.show()

if __name__ == '__main__':
	clf = svm.SVC()
	clf.fit(X, y.reshape(len(y),))
	plot_decision_regions(X=X, 
	                      y=y.reshape(len(y),),
	                      clf=clf, 
	                      legend=1)
	plt.show()
	print(clf.predict(np.array([4, 4]).reshape(1,-1)))