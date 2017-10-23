#!/usr/bin/env python
import cv2
import numpy as np

windowName = "Bean";
h = np.array([1, 4, 6, 4, 1],float);
sum = np.sum(h)
h = h/sum

def display(window, im):
	while True:
	
		cv2.imshow(window, im)
		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("n"):
			break
	cv2.destroyAllWindows()

def getGaussian(im, fil):
	height, width = im.shape[:2]
	temp = np.zeros((height, width,3), np.uint8)
	cv2.filter2D(im, -1, fil, temp)
	temp =cv2.resize(temp,(height/2,width/2))

	return temp

def getLaplacian(im, fil):
	height, width = im.shape[:2]
	temp = np.zeros((height/2,width/2,3), np.uint8)
	temp = getGaussian(im, fil)
	im = cv2.resize(im,(height/2,width/2))
	Lap = im - temp

	return Lap, temp



def main():
	image = cv2.imread('tstimg.jpg',1);
	G = cv2.resize(image,(500,500))
	image = cv2.resize(image,(500,500))
	display(windowName, image)
	tmp = np.zeros((500, 500, 3), np.uint8)

	# Problem 1: 4 levels of a guassian pyramid
	# Press N to cycle through images
	print('Press N to advance through images')
	for x in range(1,5):
		G = getGaussian(G,h)
		display('Guassian', G)

		
	# Problem 2: 4 levels of a laplacian pyramid
	# Press N to cycle through images
	tmp = np.zeros((500/2, 500/2, 3), np.uint8)
	G = cv2.resize(image,(500,500))
	for x in range(1,5):
		L, G = getLaplacian(G,h)
		display('laplacian', L)




# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()