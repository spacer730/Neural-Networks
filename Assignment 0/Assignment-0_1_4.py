import numpy as np
import matplotlib.pyplot as plt

x_1=np.array([0,0,1,1])
x_2=np.array([0,1,0,1])

def sigmoid(x):
	return 1/(1+np.exp(-x))

def cutoff(x):
	for i in np.arange(len(x)):
		if x[i]>=0.5:
			x[i]=1
		else:
			x[i]=0
	return x

def model(x_1,x_2):
	w=np.random.randn(6,1)
	t_1=sigmoid(w[0,0]*x_1+w[1,0]*x_2)
	t_2=sigmoid(w[2,0]*x_1+w[3,0]*x_2)
	y=sigmoid(w[4,0]*t_1+w[5,0]*t_2)
	y_hat=cutoff(y)
	return y_hat

AND_counter=0
XOR_counter=0

for i in np.arange(1000000):
	y=model(x_1,x_2)
	if (y==np.array([0.,1.,1.,0.])).all():
		XOR_counter+=1

	if (y==np.array([0.,0.,0.,1.])).all():
		AND_counter+=1

print('Generated XOR function '+str(XOR_counter)+' times out of one million')
print('Generated AND function '+str(AND_counter)+' times out of one million')
