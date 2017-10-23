#!/usr/bin/env python
import cv2
import numpy as np

windowName = "Bean";
h = np.array([1, 4, 6, 4, 1],float);
sum = np.sum(h)
h = h/sum
print h

def display(window, im):
	while True:
	
		cv2.imshow(window, im)
		key = cv2.waitKey(1) & 0xFF
		# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break
	cv2.destroyAllWindows()


def main():
	image = cv2.imread('Bean.jpg',1);
	image = cv2.resize(image,(500,500))
	display(windowName, image)


	height, width = image.shape[:2]
	G1 = np.zeros((height,width,3), np.uint8);
	#cv2.sepFilter2D(image,-1,h,h,G1)

	cv2.filter2D(image,-1,h, G1)
	
	# #cv2.pyrDown(G1,G1,(height/2,width/2))
	display('Gaussian 1',G1)
	L1 =  image - G1
	display('Laplacian1', L1)
	G1 = cv2.resize(G1,(height/2,width/2))
	image = cv2.resize(image,(height/2,width/2))
	width, height = G1.shape[:2]
	G2 = np.zeros((height,width,3), np.uint8);
	cv2.filter2D(G1,-1,h, G2)
	
	#cv2.pyrDown(G2,G2,(height/2,width/2))
	display('Gaus2sian ',G2)
	L2 = G1 - G2
	display('Laplacian', L2)
	G2 = cv2.resize(G2,(height/2,width/2))
	G1 = cv2.resize(G1,(height/2,width/2))
	width, height = G2.shape[:2]
	G3 = np.zeros((height,width,3), np.uint8);
	cv2.filter2D(G2,-1,h, G3)
	
	#cv2.pyrDown(G3,G3,(height/2,width/2))
	display('Gaussian 2',G3)
	L3 = G2 - G3
	display('Laplacian2', L3)
	G3 = cv2.resize(G3,(height/2,width/2))
	G2 = cv2.resize(G2,(height/2,width/2))
	width, height = G3.shape[:2]
	#G3 = cv2.resize(G3,(500,500))
	#L3 = cv2.resize(L3,(500,500))

	G4 = np.zeros((height,width,3), np.uint8);
	cv2.filter2D(G3, -1,h,G4)
	L4 = G3 - G4
	display('G4',G4)
	display('L4',L4)


	# G5 = np.zeros((height,width,3), np.uint8);
	# cv2.filter2D(G4, -1,h,G5)
	# L5 = G5 - G4
	# display('G5',G5)
	# display('L5',L5)

	# G6 = np.zeros((height,width,3), np.uint8);
	# cv2.filter2D(G5, -1,h,G6)
	# L6 = G6 - G5
	# display('G6',G6)
	# display('L6',L6)



# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()