import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def hypothesis(X , theta):
	return np.dot(X , theta)

def update_gradients(theta , grad , X , Y):
	hyp = hypothesis(X , theta)
	hyp -= Y

	for i in range(grad.shape[0]):
		grad[i] = np.sum(hyp * X[: , i].reshape(-1 , 1))

def error_func(theta , X_ , Y):
	return np.sum(((np.dot(X_ , theta)) - Y) ** 2)

def gradient_descent_mini_batch(X_ , Y , maxitr = 100 , alpha = 0.001):
	m = X_.shape[0]
	n = X_.shape[1]

	theta = np.zeros((n , 1))#values of parameters 
	grad = np.zeros((n , 1)) #values of gradients wrt to thetaj(0 <= j < n)

	for k in range(maxitr):
		update_gradients(theta , grad , X_ , Y)
		for i in range(n):
			theta[i][0] -= alpha * grad[i][0]

	return theta
		
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
	theta = gradient_descent_mini_batch(X_ , Y , 100 , 0.001)

	plt.plot(Y , 'ob' , label = 'Actual output')
	plt.plot(np.dot(X_ , theta) , 'or' , label = 'Predicted output')
	plt.legend()
	plt.show()

	print(error_func(theta , X_ , Y))