import numpy as np 

def main():
	a = np.array([4.29,2.2,5.51,5.2,10.1, -8.24,1.33,4.8,-6.62])
	a = a.reshape(3,3)
	b = np.array([0,0,0])

	det = np.linalg.det(a)		# Determinent is nonzero
	print(det)	
	#If vectors are independent solution to ax = b should be 0
	x = np.linalg.solve(a,b)
	#They are independent so it prints[0,0,0]
	print(x)




if __name__ == '__main__':
	main()