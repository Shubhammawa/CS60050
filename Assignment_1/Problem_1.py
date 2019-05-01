import numpy as np
from matplotlib import pyplot as plt
import math


# Part a
# Generation of data

def data_generation(num_instances=10):
    ''' Generates x and y vectors with dimension equal to number 
        of instances'''
    num_instances = num_instances    # Defined here to easily change

    x = np.random.uniform(size = num_instances)

    y = np.sin(2*np.pi*x) + np.random.normal(loc=0.0, scale=0.3, size = num_instances)
    
    return [x,y]


def Phi_matrix(x,degree):
	''' Creates Phi matrix with dimension (n+1,m) where n is degree
		of polynomial and m is no of instances '''
	num_instances = x.shape[0]
	Phi = np.ones(shape=num_instances)
	for i in range(1,degree+1):
	    Phi = np.append(arr = Phi, values = np.power(x,i),axis=0)
	Phi = np.reshape(Phi,[degree+1, num_instances])

	return Phi


# Part b
# Splitting dataset into train and test sets (80:20)
def train_test_split(x, y, train_size_ratio = 0.8):
    num_instances = x.shape[0]
    train_size_ratio = train_size_ratio
    indices = np.random.permutation(num_instances)
    train_idx = indices[:int(train_size_ratio*num_instances)]
    test_idx = indices[int(train_size_ratio*num_instances):]

    x_train = x[train_idx]
    x_test = x[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return [x_train, y_train, x_test, y_test]


# Part c
# Fitting regression model
def initialize_weights(degree):
	''' Initializes weights using uniform distribution '''
	W = np.random.uniform(size=degree+1)

	return W


def hypothesis(Phi, W):
	''' Computes the hypothesis function for input Phi and weight 
		vector W, dimension - (1,m) ; m = no of instances'''
	h = np.matmul(W,Phi)

	return h
    

def compute_loss(h, y):
	''' Computes the loss summed over all training examples '''
	num_instances = h.shape[0]
	loss = np.sum(np.power((h - y),2))/(2*num_instances)

	return loss
    

def gradient_descent(h,y,Phi,W, num_iterations=500,alpha=0.01):
	''' Uses gradients of the loss function to minimize the loss '''
	losses = []
	num_instances = h.shape[0]
	#losses.append(compute_loss(h,y,num_instances))
	i = 0
	while(i<= num_iterations):
		loss = compute_loss(h, y)
		losses.append(loss)
		#print("Loss after iteration {} = {}".format(i,loss))
        # dW is a vector containing derivatives of loss function
        # w.r.t all weights, dimension - same as that of weight vecctor
		dW = (1/num_instances)*np.matmul(Phi,(h-y).T)

        # Update weights using dW
		W = W - alpha*dW

        # Recompute the hypothesis function with updated weights
		h = hypothesis(Phi, W)
        
        # Thresholding condition
		if i>0:
			if (np.abs(losses[i-1]-losses[i])<0.0005):
				break
		i = i + 1

	return [W,losses]

def linear_reg(Phi, y, degree, num_iterations=10):
	''' Calls all defined functions to fit a linear regression
		model to input data '''
	num_instances = Phi.shape[1]
    
	W = initialize_weights(degree)
    
	h = hypothesis(Phi, W)
    
	W,losses = gradient_descent(h,y,Phi,W,num_iterations)
    
	return [W,losses,h]

data = data_generation(10)
x = data[0]
y = data[1]
print("x: ",x)
print("\ny: ",y)


x_train, y_train, x_test, y_test = train_test_split(x,y)

print("\nx_train: ",x_train)
print("\ny_train: ",y_train)
print("\nx_test: ",x_test)
print("\ny_test: ",y_test)
print("\n")

Weights = []
Losses = []
Test_errors = []
Y_estimates = []
Train_errors = []
for i in range(1,10):
	print("\nModel fitting for degree {}\n".format(i))
	degree = i
	Phi_train = Phi_matrix(x_train,degree)
	Phi_test = Phi_matrix(x_test,degree)

	# print(Phi_train.shape)
	# print(y_train.shape)
	# print(Phi_test.shape)
	# print(y_test.shape)

	W,losses,h = linear_reg(Phi_train,y_train,degree,500)
	Weights.append(W)
	Losses.append(losses)
	Y_estimates.append(h)
	#print(W)

	# Measuring squared error on test set
	h_test = hypothesis(Phi_test,W)
	test_error = compute_loss(h_test,y_test)
	Test_errors.append(test_error)

	h_train = hypothesis(Phi_train,W)
	train_error = compute_loss(h_train,y_train)
	Train_errors.append(train_error)
	#print(test_error)
Weights = np.array(Weights)
Losses = np.array(Losses)
Test_errors = np.array(Test_errors)
#with open('Results.txt','w') as file:
print('\nY_estimates: \n',Y_estimates)
print('\nWeights: \n',Weights)
#print('\nLosses: \n',Losses)
print('\nTest_errors: \n',Test_errors)
print('\nTrain_errors: \n',Train_errors)
#np.savetxt('Losses.csv',Losses,delimiter=",")
#np.savetxt('Test_errors.csv',Test_errors,delimiter=",")
np.save('x',x)
np.save('y',y)
np.save('Weights',Weights)
np.save('Test_errors',Test_errors)
np.save('Train_errors',Test_errors)
