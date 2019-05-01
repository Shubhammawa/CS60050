import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

x = np.load('../Part_1/x.npy')
y = np.load('../Part_1/y.npy')
Weights = np.load('../Part_1/Weights.npy')
print(Weights.shape)
plt.scatter(x,y)
plt.savefig("Data")
plt.close()
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('All_fits.pdf')

def Phi_matrix(x,degree):
	''' Creates Phi matrix with dimension (n+1,m) where n is degree
		of polynomial and m is no of instances '''
	num_instances = x.shape[0]
	Phi = np.ones(shape=num_instances)
	for i in range(1,degree+1):
	    Phi = np.append(arr = Phi, values = np.power(x,i),axis=0)
	Phi = np.reshape(Phi,[degree+1, num_instances])

	return Phi

def hypothesis(Phi, W):
	''' Computes the hypothesis function for input Phi and weight 
		vector W, dimension - (1,m) ; m = no of instances'''
	h = np.matmul(W,Phi)

	return h

for i in range(1,10):
	X = np.linspace(0,1,num=50)
	PHI = Phi_matrix(X,degree=i)
	H = hypothesis(PHI,Weights[i-1])
	#x = np.random.uniform(size=10)
	#y = np.sin(2*np.pi*x)
	plt.figure(i)
# x = [1,2,3]
# y = [1,4,9]
	plt.scatter(x,y)
	plt.plot(X,H)
#plt.savefig('Fit',format='pdf')
	#plt.plot(x,y)
	pp.savefig()
pp.close()

Train_errors = np.load('../Part_1/Train_errors.npy')
Test_errors = np.load('../Part_1/Test_errors.npy')

error = PdfPages('Errors_vs_degree.pdf')
n = np.arange(1,10)
plt.scatter(n,Train_errors)
plt.scatter(n,Test_errors)
error.savefig()
error.close()
