#Linear Regression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#hypothesis
def hypothesis(theta , X):
	return theta[0] + theta[1] * X

#error
def error_func(theta , X , Y):
	return np.sum((hypothesis(theta , X) - Y) ** 2)

#updating gradient
def update_gradient(theta , grad , X , Y):
	temp = hypothesis(theta , X) - Y
	grad[0] = np.sum(temp)
	grad[1] = np.sum(temp * X)

#gradient descent algo to calculate 
#optimum theta0 and theta 1
def gradient_descent(X , Y , alpha = 0.001):
	theta = np.array([0.0 , 0.0])
	grad = np.array([0.0 , 0.0])

	ini_error = error_func(theta , X , Y)
	e = 1
	error_list = []
	while e >= 0.0000000001:
		update_gradient(theta , grad , X , Y)
		theta[0] -= alpha * grad[0]
		theta[1] -= alpha * grad[1]
		curr_error = error_func(theta , X , Y)
		e = ini_error - curr_error
		ini_error = curr_error
		error_list.append([curr_error , theta[0] , theta[1]])
	elist = np.array(error_list)
	return theta , elist

#main 
if __name__ == '__main__' :
	#inputting data
	dfx = pd.read_csv('Datasets/hsize.csv')
	dfy = pd.read_csv('Datasets/hprice.csv')
	X_ = dfx.values.reshape(-1)
	Y = dfy.values.reshape(-1)

	#normalization
	mean , std = X_.mean() , X_.std()
	X = (X_ - mean) / std
	

	fig = plt.figure(figsize = (10 , 10))

	#plotting input
	fig.add_subplot(3 , 2 , 1)
	plt.title("REGRESSION for Data")
	plt.scatter(X_ , Y , color = 'orange' , label = 'Actual data' , s = 3)

	#calculating and plotting the line
	theta , error_list = gradient_descent(X , Y , 0.00001)
	Y_dash = hypothesis(theta , X)
	plt.plot(X_ , Y_dash , color = 'blue' , label = 'Regression line')

	print("Theta values for hypothesis : " , theta)

	##PLOTTING OTHER THINGS
	error_change = error_list[: , 0 : 1]
	theta0_change = error_list[: , 1 : 2]
	theta1_change = error_list[: , 2 : 3]


	#outputting the plot
	plt.legend()
	plt.xlabel('House Size (Square foot)')
	plt.ylabel('House Price($)')

	fig.add_subplot(3 , 2 , 2)
	plt.plot(error_change , color = 'red' , label = 'Error')
	plt.legend()
	plt.title("Error change vs iterations")
	plt.xlabel('No. of iterations')
	plt.ylabel('Error')


	#TEST DATA
	fig.add_subplot(3 , 2 , 3)
	Xtest = np.random.randint(0 , 14000 , 20)
	Xtest_temp = (Xtest - mean) / std
	Ytest = hypothesis(theta , Xtest_temp)
	plt.scatter(Xtest , Ytest , color = 'magenta' , marker = 'o' , label = 'Predicted value')
	plt.legend()
	plt.title("Predicted values for test data")
	plt.xlabel('Size_test(Square foot)')
	plt.ylabel('Price_test($)')

	#plotting 3d to visualize gradient descent
	##generating a theta0 and theta1 matrix
	axes = fig.add_subplot(3 , 2 , 4 , projection = '3d')

	theta0 = np.arange(200000 , 900000 , 5000)	
	theta1 = np.arange(50000 , 500000 , 5000)
	theta0 , theta1 = np.meshgrid(theta0 , theta1)
	J = np.zeros(theta0.shape)

	m = theta0.shape[0]
	n = theta1.shape[1]

	for i in range(m):
		for j in range(n):
			J[i][j] = error_func([theta0[i][j] , theta1[i][j]] , X , Y)

	axes.plot_surface(theta0 , theta1 , J , cmap = "rainbow" , alpha = 0.68)
	#alpha is for transparency
	axes.scatter(theta0_change , theta1_change , error_change , color = 'black' , label = 'Gradient \nDescent \nTrajectory')
	
	plt.legend()
	plt.title("Error function for different \ntheta0 and theta1")
	plt.xlabel("theta0")
	plt.ylabel("theta1")

	#3d contour plot
	axes = fig.add_subplot(3 , 2 , 5 , projection = '3d')
	axes.contour(theta0 , theta1 , J , cmap = "rainbow")
	#alpha is for transparency
	axes.scatter(theta0_change , theta1_change , error_change , color = 'black' , label = 'Gradient \nDescent \nTrajectory')
	plt.legend()
	plt.title("Error function for different \ntheta0 and theta1 (Contour plot)")
	plt.xlabel("theta0")
	plt.ylabel("theta1")

	#2d contour plot
	fig.add_subplot(3 , 2 , 6)
	plt.contour(theta0 , theta1 , J)
	#alpha is for transparency
	plt.scatter(theta0_change , theta1_change , color = 'black' , label = 'Gradient \nDescent \nTrajectory')
	plt.legend()
	plt.title("Error function for different \ntheta0 and theta1 (Contour plot)")
	plt.xlabel("theta0")
	plt.ylabel("theta1")

	plt.tight_layout()
	plt.show()
