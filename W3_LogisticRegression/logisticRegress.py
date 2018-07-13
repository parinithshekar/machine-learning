import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import costFunction as cf
import sigmoid as sg

import sklearn as sk
from sklearn import metrics, datasets
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('ex2data1.txt', header=None)
data.columns = ['Test1', 'Test2', 'Result']

X = data.drop('Result', axis=1)
Y = data['Result']

#Ysk = Y.replace(0, 2)

Xnp = np.array(X)
Ynp = np.array(Y)

iters = 50000 # Doesnt saturate after 200,000 itarations also
alpha = 0.00001 # 0.00001 upper limit
theta = np.zeros(X.shape[1]+1) # Initialize with zeroes
cost_change = []
print("Initial theta:", theta)

skmod = LogisticRegression().fit(X, Y)

# Learning
for i in range(iters):
	gradient_vector = cf.gradient(theta, Xnp, Ynp)
	theta = theta - alpha*gradient_vector
	cost_change.append(cf.cost(theta, Xnp, Ynp))
print("Cost after training:", cf.cost(theta, Xnp, Ynp))
print("Thetas after training:", theta)
x_test = np.array([1, 45, 85])
print("Final prediction:", sg.sigmoid(x_test.T.dot(theta)))

input = np.array([[45, 85]])
input.reshape(1,-1)
print("Final Prediciton SK:", skmod.predict(input))

# Assigning colors to students based on acceptance status
colors = []
for i in range(len(Y)):
	if(Y[i] == 1):
		colors.append('g')
	else:
		colors.append('r')

# Accepted VS Rejected
plt.subplot(1, 2, 1)
plt.scatter(X['Test1'], X['Test2'], c=colors)
plt.xlabel('Test 1 score')
plt.ylabel('Test 2 score')
green_patch = mpatches.Patch(color='green', label='Accepted')
red_patch = mpatches.Patch(color='red', label='Rejected')

plt.plot([i for i in range(30,100)], [(0.5 - theta[0] - theta[1] * x)/theta[2] for x in range(30,100)])
plt.legend(handles=[green_patch, red_patch])

# Cost function
plt.subplot(1, 2, 2)
plt.plot([i for i in range(iters)], cost_change)

plt.show()
