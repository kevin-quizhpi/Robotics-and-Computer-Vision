import cv2
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D as axes3d 

# Load images in
imLeft = cv2.imread('left.jpg',1)
imRight = cv2.imread('right.jpg', 1)

def close(event):
	if event.key == 'c':
		plt.close(event.canvas.figure)

def treePlot(pts):
	fig = plt.figure()
	ax = plt.axes(projections = '3d')
	fig.canvas.mpl_connect('key_press_event', close)
	ax.scatter(pts[0], pts[1], pts[2])
	plt.show()



def main():
    
	#finding point correspondences
	sift = cv2.xfeatures2d.SIFT_create()
	keyLeft , desLeft = sift.detectAndCompute(imLeft, None)
	keyRight , desRight = sift.detectAndCompute(imRight, None)
	
	bf = cv2.BFMatcher()

	matches = bf.match(desLeft,desRight, k= 2)


	# Ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.95*n.distance:
			good.append([m])


	# Draw first 10 matches.
	img3 = cv2.drawMatches(imLeft, keyLeft, imRight, keyRight,matches[:10], flags=2)

	plt.imshow(img3),plt.show()


if __name__ == "__main__":
    main()