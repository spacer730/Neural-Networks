import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

def xor_net(x1,x2,weights):
	y=np.dot(np.array([weights[0],weights[1]]),np.array([1,x1,x2]))
	z=np.dot(np.array(weights[2]),np.array([1,y[0],y[1]]))
	return z

def mse(weights):
	mse00=(0-xor_net(0,0,weights))**2
	mse01=(1-xor_net(0,1,weights))**2
	mse10=(1-xor_net(1,0,weights))**2
	mse11=(0-xor_net(1,1,weights))**2
	mse=mse00+mse01+mse10+mse11
	return mse

def missclassified(weights):
	missclassified=0
	if xor_net(0,0,weights)>0.5:
		missclassified+=1
	if xor_net(0,1,weights)<=0.5:
		misclassified+=1
	if xor_net(1,0,weights)<=0.5:
		misclassified+=1
	if xor_net(1,1,weights)>0.5:
		misclassified+=1
	return missclassified
	

def grdmse(weights):
	eps=0.001
	numrows=weights.shape[0]
	numcols=weights.shape[1]
	grmdse=np.zeros((3,3))
	for i in range(numrows):
		for j in range(numcols):
			a=np.zeros((3,3))
			a[i][j]=eps
			grdmse[i][j]=(mse(weights+a)-mse(weights))/eps

def trainnetwork(learningrate):
	np.random.seed(0)
	weights=np.random.randn(3,3)
	counter=0
	mse=[]
	difmse=[]
	for i in range(2000): #difmse>0:
		#a=mse(weights)
		weights=weights-learningrate*grdmse(weights)
		#b=mse(weights)
		#difmse.append(b-a)
		#mse.append(b)
		counter+=1

weights=np.random.randn(3,3)
a=np.zeros((3,3))
a[0][0]=0.001
print(mse(weights+a))
trainnetwork(0.2)

plt.figure()
plt.plot(range(len(difmse)),difmse)
plt.show()
