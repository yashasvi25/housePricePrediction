import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def hypothesis(X , theta , b , batchsize):
	l = (b * batchsize) + 1
	r = (b + 1) * batchsize
	return np.dot(X[l : r + 1 , :] , theta)

def update_gradients(theta , grad , X , Y , b , batchsize):
	l = (b * batchsize) + 1
	r = (b + 1) * batchsize
	hyp = hypothesis(X , theta , b , batchsize)
	hyp -= Y[l : r + 1 , : ]

	for i in range(grad.shape[0]):
		grad[i] = np.sum(hyp * X[l : r + 1 , i].reshape(-1 , 1))

def error_func(theta , X_ , Y):
	return np.sum(((np.dot(X_ , theta)) - Y) ** 2)

def gradient_descent(X_ , Y , alpha = 0.001 , batchsize = 1):
	m = X_.shape[0]
	n = X_.shape[1]

	theta = np.zeros((n , 1))#values of parameters 
	grad = np.zeros((n , 1)) #values of gradients wrt to thetaj(0 <= j < n)

	e = []
	for b in range(m // batchsize):
		update_gradients(theta , grad , X_ , Y , b , batchsize)
		for i in range(n):
			theta[i][0] -= alpha * grad[i][0]

		e.append(error_func(theta , X_ , Y))

	return theta,e
		
if __name__ == '__main__':
	#inputting data
	df = pd.read_csv('Datasets/Train.csv')
	temp = df.values

	X = temp[: , :-1]
	Y = temp[: , -1:]
	
	#normalization
	means = np.mean(X , axis = 0)
	stds = np.std(X , axis = 0)
	X_ = (X - means) / stds ##if a particular feature has equal values fro all examples 
	                        ##then ZeroDivisionError

	#adding a column of ones for x0
	X_ = np.hstack((np.ones((X_.shape[0] , 1)) , X_))
	
	#calculation of parameters
	batchsize = 16
	theta,e = gradient_descent(X_ , Y , 0.005 , int(batchsize))

	fig = plt.figure()

	fig.add_subplot(2 , 2 , 1)
	plt.title("Actual output and predicted output")
	plt.plot(Y , 'ob' , label = 'Actual output')
	plt.plot(np.dot(X_ , theta) , 'or' , label = 'Predicted output')
	plt.legend()

	fig.add_subplot(2 , 2 , 4)
	plt.title("Error vs iterations (Total batchsize : 1600)")
	plt.plot(e , '-b' , label = "Batchsize : %d"%(batchsize))
	plt.legend()
	plt.tight_layout()
	plt.show()

print(error_func(theta , X_ , Y))