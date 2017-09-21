#Kevin Quizhpi
#332:472
#Hw 1 problem 3
import numpy as np

def main():
	A = np.array([0,1,1,1,1.9,1,3,1,3.9,1,5,1])
	A = A.reshape(6,2)
	b = np.array([1,3.2,5,7.2,9.3,11.1])
	b = b.reshape(6,1)
	At = A.transpose()
	# Here we begin least squares estimation: inv(At*A)*At*b)
	q = At.dot(A)
	q = np.linalg.inv(q)
	q = q.dot(At)
	q = q.dot(b)
	print('q is: ')	
	print(q)
	print('Line estimated to: y =', q[0,0],"x + ",q[1,0])



if __name__ == '__main__':
	main()