import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_table("ex1data1.txt", sep=",", header=None)
data.columns = ["population", "profit"]
X = data["population"]
Y = data["profit"]


iters = 1500
alpha = 0.01

t0 = 0
t1 = 0

def hypo(x):
	return float(t0 + t1*x)

def cost():
	res=0
	m = len(X)
	for x,y in zip(X,Y):
		res = res + (hypo(x)-y)**2
	return res/(2*m)

def _t0():
	res = 0
	m = len(X)
	for x,y in zip(X,Y):
		res = res + (hypo(x)-y)
	return res/m

def _t1():
	res = 0
	m = len(X)
	for x,y in zip(X,Y):
		res = res + (hypo(x)-y)*x
	return res/m

def iter():
	global t0
	global t1
	t0_err = _t0()
	t1_err = _t1()
	t0 = t0 - alpha*t0_err
	t1 = t1 - alpha*t1_err

if __name__ == '__main__':
	_cost = []
	for i in range(iters):
		_cost.append(cost())
		iter()

	x = np.linspace(0, 25, 1000)
	x_cost = np.linspace(0, iters, iters)

	obj = linear_model.LinearRegression()
	obj.fit(X.values.reshape(-1, 1), Y)
	sk_x = np.linspace(0, 25, 1000).reshape(-1, 1)

	plt.subplot(1, 2, 1)
	plt.scatter(x=data["population"], y=data["profit"], c="red")
	plt.plot(x, (t0+t1*x))
	plt.plot(sk_x, obj.predict(sk_x), c="black")
	plt.title("Population VS Profit")
	plt.xlabel("Population")
	plt.ylabel("Profit")

	plt.subplot(1, 2, 2)
	plt.plot(x_cost, _cost)

	plt.show()
