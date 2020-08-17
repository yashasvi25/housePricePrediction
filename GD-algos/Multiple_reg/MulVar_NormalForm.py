import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression

def hypothesis(X , theta):
	return np.dot(X , theta)

def getTheta(X , Y):
	firstPart = np.linalg.pinv(np.dot(X.T , X))
	seconPart = np.dot(X.T , Y)

	theta = np.dot(firstPart , seconPart)
	return theta 

if __name__ == '__main__':
	#generating data
	X , Y = make_regression(n_samples = 400 , n_features = 1 , n_informative = 1 , noise = 6.8 , random_state = 11)
	Y = Y.reshape(-1 , 1)

	#normalize and adding a column of 1
	X_ = (X - X.mean()) / X.std()
	X_ = np.hstack((np.ones((X_.shape[0] , 1)) , X_))

	#calculating
	theta = getTheta(X_ , Y)

	#plotting data points
	plt.figure()
	plt.scatter(X , Y , c = 'blue' , s = 7.6 , marker = 'o' , label = 'Actual values')
	plt.plot(X , hypothesis(X_ , theta) , color = 'red' , label = 'Predicted values')
	plt.title('Normal form of regression')
	plt.legend()
	plt.show()
