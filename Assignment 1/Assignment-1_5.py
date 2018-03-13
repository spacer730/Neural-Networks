import numpy as np
import numpy.ma as ma
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
	

def grdmse(weights):
	numrows=weights.shape[0]
	numcols=weights.shape[1]
	for i in range(numrows):
		for j in range(numcols):
			(mse(weights+eps*)-mse(weights))/eps

def trainnetwork(learningrate):
	seed()
	weights=np.random.rand(2,3)
	counter=0
	for i in range(2000): #difmse>0:
		a=mse(weights)
		weights=weights-learningrate*grdmse(weights)
		b=mse(weights)
		difmse=b-a
		counter+=1
		


