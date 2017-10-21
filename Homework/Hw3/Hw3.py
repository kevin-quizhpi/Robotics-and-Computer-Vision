from __future__ import print_function
import sys
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
import math


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

# Order to plot the house
p = [p1,p2,p3,p4,p1,p5,p6,p2,p6,p7,
	p3,p7,p8,p4,p8,p5,p10,p6,p10,p9,p7,p9,p8]

# Intial Tranformation matrix
# Rotation & translation matrix
T0 = np.array([
	[1, 0, 0,   6],
	[0, 0, 1,   6],
	[0, 1, 0,   6]])

#Translation vector
t = np.array([6,6,6])[np.newaxis].T

# Perpective Project + Scale & Shift
K = np.array([
	[10,   0, 0],
	[ -0, 10, 0],
	[  0,  0, 1]])



def pressC(eventC):
    if eventC.key=="c":
        plt.close(eventC.canvas.figure)

def pressN(eventN):
	if eventN.key == "n":
		plt.close(eventN.canvas.figure)

def drawMyObject(cor):

	# Plotting the house!
	fig =plt.figure()
	plt.plot(cor[:,0],cor[:,1])
	fig = plt.gcf()
	cid = fig.canvas.mpl_connect('key_press_event', pressC)
	plt.show()

def getCoordinates(RT, K):
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

	return np.delete(cor,0,0)	# Removing first zero points

# Angles are accepted in Degrees
def getT(angZ, angY, angX):

	angZ = math.radians(angZ);
	angY = math.radians(angY);
	angX = math.radians(angX);

	Z = np.array([
		[math.cos(angZ),   -math.sin(angZ), 0],
		[math.sin(angZ),	math.cos(angZ), 0],
		[			  0,				 0, 1]
		]);

	Y = np.array([
		[ math.cos(angY), 0, 	math.sin(angY)],
		[			   0, 1,				 0],
		[-math.sin(angY), 0,	math.cos(angY)]
		]);

	X = np.array([
		[1,				 0, 			  0],
		[0,	math.cos(angX), -math.sin(angX)],
		[0,	math.sin(angX),  math.cos(angX)]
		]);

	R = np.matmul(np.matmul(Z,Y), X)
	Rt = np.concatenate((R,t),axis = 1)
	return Rt


def main():
	print('start')

	# Part one: Wire frame of house
	cor = getCoordinates(T0,K);
	drawMyObject(cor);

	# Part two: will do 6 rotations of the house
	# about the z axis only
	rotations = 8
	for x in range(1, rotations+1):
		aZ = 360/rotations * x;
		T = getT(aZ,0,0);
		cor = getCoordinates(T,K);
		fig =plt.figure()
		plt.plot(cor[:,0],cor[:,1])
		fig = plt.gcf()
		cid = fig.canvas.mpl_connect('key_press_event', pressN)
		plt.show()

if __name__ == '__main__':
    main()











