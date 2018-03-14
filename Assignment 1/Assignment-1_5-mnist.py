import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return max(0,x)

def xor_net(x,weights):
	y=np.dot(np.array([weights[0],weights[1]]),np.array([1,x1,x2]))
	#y=list(map(lambda y: sigmoid(y), y))
	#y=list(map(lambda y: relu(y), y))
	y=list(map(lambda y: np.tanh(y), y))
	z=np.tanh(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))
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
	grdmse=np.zeros((3,3))
	for i in range(numrows):
		for j in range(numcols):
			a=np.zeros((3,3))
			a[i][j]=eps
			grdmse[i][j]=(mse(weights+a)-mse(weights))/eps
	return grdmse

def trainnetwork(learningrate):
	#weights=np.random.randn(3,3)
	weights=np.random.rand(3,3)
	counter=0
	mselist=[]
	difmse=[]
	updatedmse=mse(weights)
	while counter < 2000:
		initmse=mse(weights)
		weights=weights-learningrate*grdmse(weights)
		updatedmse=mse(weights)
		difmse.append(updatedmse-initmse)
		mselist.append(updatedmse)
		counter+=1
	return difmse, mselist, weights, counter

difmse, mselist, weights, counter = trainnetwork(0.1)

print(xor_net(0,1,weights))
print(xor_net(1,0,weights))
print(xor_net(0,0,weights))
print(xor_net(1,1,weights))

plt.figure()
plt.plot(range(counter),mselist)
plt.show()
