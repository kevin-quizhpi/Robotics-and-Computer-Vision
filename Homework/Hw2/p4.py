#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt




# Problem 4


# Creating postion vectors for house vertices
p1 = np.array([0,0,0,1])[np.newaxis]
p2 = np.array([10,0,0,1])[np.newaxis]
p3 = np.array([10,10,0,1])[np.newaxis]
p4 = np.array([0,10,0,1])[np.newaxis]
p5 = np.array([0,0,10,1])[np.newaxis]
p6 = np.array([10,0,10,1])[np.newaxis]
p7 = np.array([10,10,10,1])[np.newaxis]
p8 = np.array([0,10, 10,1])[np.newaxis]
p9 = np.array([5, 10, 15,1])[np.newaxis]
p10 = np.array([5, 0, 15,1])[np.newaxis]


# Rotation & translation matrix
RT = np.array([
	[-0.707, -0.707, 0,   3],
	[ 0.707, -0.707, 0, 0.5],
	[ 	  0,	  0, 1,   3]])


# Perpective Project + Scale & Shift
K = np.array([
	[100,   0, 200],
	[ -0, 100, 200],
	[  0,   0,   1]])

# Order to plot the house
p = [p1,p2,p3,p4,p1,p5,p6,p2,p6,p7,p3,p7,p8,p4,p8,p5,p10,p9,p7,p9,p8]

cor = np.array([[0, 0]])
for x in range(0,len(p)):

	# Matrix multiplication of Camera Matrix by point
	pIM = np.matmul(RT,p[x].T)
	pIM = np.matmul(K,pIM)

	#Division by z component
	z = pIM[2]
	# We have image coordinates
	pIM = pIM/z
	pIM = np.delete(pIM, 2)
	cor = np.concatenate((cor,[pIM]))

cor = np.delete(cor,0,0)	# Removing first zero points

# Plotting the house!

	
fig =plt.figure()
plt.plot(cor[:,0],cor[:,1])
plt.show()


