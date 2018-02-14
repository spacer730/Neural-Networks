import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

xaxis=np.linspace(0,1,100)
x_train=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
x_test=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
y_train=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
y_test=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
n=[9.,15.,100.]
poly=[] #Here we will store the coefficients of the polynomials

for i in np.arange(3):
	x_train[i]=np.random.uniform(0,1,n[i])
	y_train[i]=np.random.normal(0,0.05,n[i])+0.5+0.4*np.sin(2*np.pi*x_train[i])
	x_test[i]=np.random.uniform(0,1,n[i])
	y_test[i]=np.random.normal(0,0.05,n[i])+0.5+0.4*np.sin(2*np.pi*x_test[i])
	for d in np.arange(10):
		poly.append(np.zeros((3,d+1)))
		poly[d][i]=np.polyfit(x_train[i],y_train[i],d)
		p=np.poly1d(poly[d][i])
		plt.figure(10*i+d+1)
		plt.plot(x_train[i],y_train[i],'ro')
		plt.plot(x_test[i],y_test[i],'bo')
		plt.legend(handles=[mpatches.Patch(color='red', label='Training set'),mpatches.Patch(color='blue', label='Test set')])
		plt.plot(xaxis,p(xaxis))
		plt.ylim(0,1.2)
		plt.title('Polynomial fit of degree '+str(d))
		plt.savefig('Pol. fit degree '+str(d)+' for '+str(n[i])+' datapoints.png')

error_train=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
error_test=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
reg_error=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
MSE_train=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
MSE_test=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])

for i in np.arange(3):
	for d in np.arange(10):
		for k in np.arange(n[i]):
			error_train[i][d]=error_train[i][d]+(np.polyval(poly[d][i],x_train[i])[k]-y_train[i][k])**2
			error_test[i][d]=error_test[i][d]+(np.polyval(poly[d][i],x_test[i])[k]-y_test[i][k])**2
		MSE_train[i][d]=(1/n[i])*error_train[i][d]
		MSE_test[i][d]=(1/n[i])*error_test[i][d]
	plt.figure(31+i)
	plt.plot(np.arange(10),MSE_train[i],'ro')
	plt.plot(np.arange(10),MSE_test[i],'bo')
	plt.legend(handles=[mpatches.Patch(color='red', label='Training set'),mpatches.Patch(color='blue', label='Test set')])
	plt.ylabel('MSE')
	plt.xlabel('Polynomial degree d')
	plt.title('Dataset with '+str(n[i])+' datapoints')
	plt.savefig('MSE for training and test sets with '+str(n[i])+' points against polynomial degree.png')
	
#plt.show()
