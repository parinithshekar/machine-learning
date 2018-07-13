import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Reading data from file
data = pd.read_table("ex1data2.txt", sep=",", header=None)
data.columns = ["size", "bedrooms", "price"]

# Mean and Standard deviation for every feature set
x1_mean = data["size"].mean()
x1_std = data["size"].std()

x2_mean = data["bedrooms"].mean()
x2_std = data["bedrooms"].std()

y_mean = data["price"].mean()
y_std = data["price"].std()

# Scaled and normalized features
X1 = np.array([ (x1-x1_mean)/x1_std for x1 in data["size"] ]).reshape(-1,1)
X2 = np.array([ (x2-x2_mean)/x2_std for x2 in data["bedrooms"] ]).reshape(-1,1)
Y = np.array([ (y-y_mean)/y_std for y in data["price"] ]).reshape(-1,1)

# No.of iterations and Learning rate
iters = 400
alpha = 0.03

# Parameters to tune
t0 = 0
t1 = 0
t2 = 0

def hypo(x1, x2):
	return float(t0 + t1*x1 + t2*x2)

def cost():
	res = 0
	for x1,x2,y in zip(X1,X2,Y):
		res = res + (hypo(x1, x2)-y)**2
	return res/(2*len(Y))

def _t0():
	res = 0
	for x1,x2,y in zip(X1, X2, Y):
		res = res + hypo(x1, x2)-y
	return res/len(Y)

def _t1():
	res = 0
	for x1,x2,y in zip(X1, X2, Y):
		res = res + (hypo(x1, x2)-y)*x1
	return res/len(Y)

def _t2():
	res = 0
	for x1,x2,y in zip(X1, X2, Y):
		res = res + (hypo(x1, x2)-y)*x2
	return res/len(Y)

def iter():
	global t0
	global t1
	global t2
	t0_err = _t0()
	t1_err = _t1()
	t2_err = _t2()
	t0 = t0 - alpha*t0_err
	t1 = t1 - alpha*t1_err
	t2 = t2 - alpha*t2_err

if __name__ == '__main__':
	_cost = []
	for i in range(iters):
		_cost.append(cost())
		iter()

	# Input must be transformed to the SCALED_NORMALIZED values and the prediciton must be inverted(backtransformed).
	# This little hack reduces no.of iterations and increases learning rate
	print( (hypo((1650-x1_mean)/x1_std, (3-x2_mean)/x2_std)*y_std)+y_mean )

	x_cost = np.linspace(0, iters, iters).reshape(-1, 1)
	y_cost = np.array(_cost).reshape(-1,1)
	plt.plot(x_cost, y_cost)
	plt.title("Cost function minimization")
	plt.xlabel("Iteration No")
	plt.ylabel("Cost function value")
	plt.show()
