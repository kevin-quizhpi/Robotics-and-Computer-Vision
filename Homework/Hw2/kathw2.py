#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt




# import modules used here -- sys is a very standard one
import sys
#-----------------------------------------------------------------------------
#Problem 2

refPt = []
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1

#created a new array
arr = np.array([[0, 0]])

def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image,lx,ly, arr
    
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates 
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		arr = np.concatenate((arr,refPt))
		print  (x,y)
		lx = x
		ly = y
				   
# Gather our code in a main() function
def main():
	# Read Image
	image = cv2.imread('ts.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
	global arr
    
# keep looping until the 'q' key is pressed
	while True:
	# display the image and wait for a keypress
            image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
            cv2.imshow(windowName, image)            
            key=cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                break
 

	# Close the window will exit the program
	cv2.destroyAllWindows()
	arr = np.delete(arr,0,0)	
	print(arr)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

# Select End Points of foreshortened window or billboard
x1p=00
x2p=1499
x3p=1499
x4p=00
y1p=00
y2p=00
y3p=999
y4p=999

#destination array
#my destination array is the points that are clicked on
dest=arr

x1=dest[0,0]
x2=dest[1,0]
x3=dest[2,0]
x4=dest[3,0]
y1=dest[0,1]
y2=dest[1,1]
y3=dest[2,1]
y4=dest[3,1]


#here is my homograpy 8x9 matrix
hmg=np.array([
      [ -x1,  -y1, -1, 0, 0, 0, x1p*x1, x1p*y1, x1p],
      [ 0,  0, 0, -x1, -y1, -1, y1p*x1, y1p*y1, y1p],
      [ -x2,  -y2, -1, 0, 0, 0, x2p*x2, x2p*y2, x2p],
      [ 0,  0, 0, -x2, -y1, -1, y2p*x2, y2p*y2, y2p],
      [ -x3,  -y3, -1, 0, 0, 0, x3p*x3, x3p*y3, x3p],
      [ 0,  0, 0, -x3, -y1, -1, y3p*x3, y3p*y3, y3p],
      [ -x4,  -y4, -1, 0, 0, 0, x4p*x4, x4p*y4, x4p],
      [ 0,  0, 0, -x4, -y1, -1, y4p*x4, y4p*y4, y4p]  
      ])
print('\n\nhmg: ')
print(hmg)

#function created to find SVD and then to find the nullspace of the matrix
U,s,V = np.linalg.svd(hmg)

print(V)
# Estimate the homography 
print("Estimated homography matrix: ")
#print(nullspace(hmg))
H=np.reshape(V[8,:],(3,3))
print(H)

#multiply the 9x1 matrix
print("Estimated homography matrix: ")
#end=(np.dot(hmg, nullspace(hmg)))
#newrow=np.array([1]);
#end=np.append(end,newrow)
#print(np.dot(hmg, nullspace(hmg)))
# Set the corresponding point in the frontal view as 
#final=nullspace(hmg)

print("Homography Matrix : ")
#print(final)


# Warp source image to destination based on homography
image = cv2.imread('ts.jpg',1);
#reshapre the matrix into a 3x3
#full=final.reshape(3,3)
print("Reshaped Homography Matrix 3x3: ")
#print(full)

dst = cv2.warpPerspective(image,H,(1500,1000))

# refPt = []
# image3 = np.zeros((512, 512, 3), np.uint8)

#output the final image
cv2.imshow("HW Window:", dst)
cv2.waitKey(3000)