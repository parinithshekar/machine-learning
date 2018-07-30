import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from mnist import MNIST
import random
from PIL import Image
import cv2
import sigmoid as sg

mndata = MNIST('mnist_data')
images, labels = mndata.load_training()
print("MNIST Dataset loaded")

batchSize = 100
numBatches = len(images)/batchSize

lamb = 1000 # Regularization term
alpha = 0.001 # Learning rate

# OpenCV too hellbent on resolution
'''
im = np.array(images[0]).reshape(28, 28)
im = im/255
frame_display = cv2.resize(im, (300, 300))

cv2.imshow("image", frame_display)
cv2.waitKey()
'''


im_ar = np.array(images) # A matrix containing image data of 60k images(60k x 784px)

'''
# Matplotlib be best
for i in range(25):
	plt.subplot(5, 5, i+1)
	plt.imshow(im_ar[i].reshape(28, 28), cmap="gray")
	plt.axis('off')
plt.show()
'''
#im_ar = np.insert(im_ar, 0, 1, axis=1)

# Generate Y
l_ar = []
for i in labels:
	label = np.zeros(10)
	label[i] = 1
	l_ar.append(label)
l_ar = np.array(l_ar)
print("Numpy Arrays generated")

# Forward prop begins
IP_SIZE = 784 # Size of every image, 28x28px
L2_SIZE = 15 # No of nodes in 2nd Layer
L3_SIZE = 15 # No of nodes in 3rd Layer
OP_SIZE = 10 # No of nodes in output layer, no of classes.

X = np.zeros(IP_SIZE)
Y = np.zeros(OP_SIZE)
a2 = np.zeros(L2_SIZE)
a3 = np.zeros(L3_SIZE)

T1 = np.random.rand(L2_SIZE, IP_SIZE)# + 1)
T2 = np.random.rand(L3_SIZE, L2_SIZE)# + 1)
T3 = np.random.rand(OP_SIZE, L3_SIZE)# + 1)

tDel1 = np.zeros([L2_SIZE, IP_SIZE])
tDel2 = np.zeros([L3_SIZE, L2_SIZE])
tDel3 = np.zeros([OP_SIZE, L3_SIZE])

D1 = np.zeros([L2_SIZE, IP_SIZE])
D2 = np.zeros([L3_SIZE, L2_SIZE])
D3 = np.zeros([OP_SIZE, L3_SIZE])

def forward_pass(x):
	global T1, T2, T3
	#x = np.insert(x, 0, 1, axis=1)
	A2 = sg.sigmoid_matrix(x.dot(T1.T))
	#print(A2)
	#A2o = np.insert(A2, 0, 1, axis=1) # Add a2(0)
	A3 = sg.sigmoid_matrix(A2.dot(T2.T))
	#print(A3.shape)
	#A3o = np.insert(A3, 0, 1, axis=1) # Add a3(0)
	Y = sg.sigmoid_matrix(A3.dot(T3.T))
	return (A2, A3, Y)

def backpropogate():
	global D1, D2, D3
	global tDel1, tDel2, tDel3
	global im_ar, l_ar, batchSize, numBatches
	global T1, T2, T3

	print("Training Begins...")
	presInd = 0
	for z in range(59999):
		#for x in range(int(batchSize)):


		inp = im_ar[presInd:(presInd+1)]
		outp = l_ar[presInd:(presInd+1)]
		A2, A3, Y = forward_pass(inp)
		#print(A3.shape)
		Del4 = Y - outp
		Del3 = np.multiply( Del4.dot(T3), np.multiply(A3, np.ones(A3.shape)-A3))
		Del2 = np.multiply( Del3.dot(T2), np.multiply(A2, np.ones(A2.shape)-A2))
		'''
		mDel4 = np.mean(Del4, axis=0)
		mDel3 = np.mean(Del3, axis=0)
		mDel2 = np.mean(Del2, axis=0)
		'''
		tDel3 = tDel3 + Del4.T.dot(A3)
		tDel2 = tDel2 + Del3.T.dot(A2)
		tDel1 = tDel1 + Del2.T.dot(inp)
		presInd += 1



		D1 = (tDel1/batchSize) + (lamb*T1)
		D2 = (tDel2/batchSize) + (lamb*T2)
		D3 = (tDel3/batchSize) + (lamb*T3)
		#print(D1.shape)
		T1 = T1 - alpha*D1
		T2 = T2 - alpha*D2
		T3 = T3 - alpha*D3


	print("Training Done.")


if __name__ == '__main__':
	#global im_ar
	print("5 Before training:\n",forward_pass(im_ar[0:1]))
	backpropogate()
	print("5 After training:\n",forward_pass(im_ar[0:1]))
