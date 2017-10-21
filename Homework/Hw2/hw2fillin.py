#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# import modules used here -- sys is a very standard one
import sys

refPt = []
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1
src = np.array([[0, 0]])


def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image,lx,ly, src

 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates 
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		print  (x,y)
		lx = x
		ly = y
		# Concatenate captured points to src matrix
		src = np.concatenate((src,refPt))

# Function to calculate homography matrix between two sets of points.
def getHomography(src, dst):

	A = np.zeros((2,9),dtype = int)

# Estimate the homography 

	for i in range(0,4):
		
	# Accessing array elements from src & des point matrices
		x = src.item((i,0))
		y = src.item((i,1))
		xP = des.item((i,0))
		yP = des.item((i,1))

	# Creating temp 2x9 matrix to concatenate to A
		B = np.zeros((2,9),dtype=int)

	# Populating 2x9 A matrix for each point
		B[0,0] = -x
		B[0,1] = -y
		B[0,2] = -1
		B[0,6] = xP*x
		B[0,7] = xP*y
		B[0,8] = xP
		B[1,3] = -x
		B[1,4] = -y
		B[1,5] = -1
		B[1,6] = yP*x
		B[1,7] = yP*y
		B[1,8] = yP

	# Concatenating our 2x9 to make up the A matrix
		A = np.concatenate((A,B))
		i += 1		# Increment pointer

	# Remove first two empty rows
	A = np.delete(A, np.s_[:2],0)

	# Compute SVD
	U, s, V = np.linalg.svd(A)

	# Solution to Aq = 0 is the last column of V in vector form
	# It is extracted and reshaped to make up the homography matrix
	H = np.reshape(V[8],(3,3))
	return H



# Gather our code in a main() function
def main():
	# Read Image
	image = cv2.imread('ts.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)

	global src


 
# keep looping until the 'q' key is pressed

	while True:
	# display the image and wait for a keypress
		print image
		image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
		print image
		cv2.imshow(windowName, image)
		key = cv2.waitKey(1) & 0xFF
	

 
	# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break
 

	# Close the window will exit the program
	cv2.destroyAllWindows()
	src = np.delete(src,0,0)	# Removing first zero points



# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()


# Prolbem 2: 

des = np.array([
	[00, 00],	
	[1499,00],
	[1499, 999],
	[00, 999]
	
	])

# Reloading source image
srcIm = cv2.imread('ts.jpg',1)
# Getting Homography matrix
H = getHomography(src,des)
# Extracting billboard
dst = cv2.warpPerspective(srcIm,H,(1500,1000))
# Saving billboard to file dst

while True:
	
	cv2.imshow('dst',dst)
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
cv2.destroyAllWindows();


# Problem 3

# Load test image
test = cv2.imread('testimg.jpg',1)
# Going to crop to same size as tranformed image
testCrop = test[200:1200,0:1500]

# Finding homography between destination and source in opposite
# relation
H2 = getHomography( src,des)
# Using inverse of Homography matrix to map source to destination
H2 = np.linalg.inv(H2)
# Image with perspective changed



bill = cv2.warpPerspective(testCrop,H2,(srcIm.shape[1],srcIm.shape[0]),)

# Remove area where we want to replace
srcIm =  cv2.fillPoly(srcIm,[src],(0,0,0))
# Add test image to ts.jpg
bill +=srcIm



while True:
	
	cv2.imshow('bill',bill)
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
cv2.destroyAllWindows();


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



















