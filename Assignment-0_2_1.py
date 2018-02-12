import numpy as np
import matplotlib.pyplot as plt

x=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
y=np.array([np.zeros(9),np.zeros(15),np.zeros(100)])
n=[9,15,100]
poly=[]

for i in np.arange(3):
	x[i]=np.random.uniform(0,1,n[i])
	y[i]=np.random.normal(0,0.05,n[i])+0.5+0.4*np.sin(2*np.pi*x[i])
	for d in np.arange(10):
		poly.append(np.zeros((3,d+1)))
		poly[d][i]=np.polyfit(x[i],y[i],d)
		plt.figure(10*i+d+1)
		plt.plot(x[i],y[i],'ro')
		plt.plot(x[i],np.polyval(poly[d][i],x[i]),'bo')
		plt.title('Polynomial fit of degree'+str(d))
		plt.savefig('Pol. fit degree '+str(d)+' for '+str(n[i])+' datapoints.png')

error=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
MSE=np.array([np.zeros(10),np.zeros(10),np.zeros(10)])
for i in np.arange(3):
	for d in np.arange(10):
		for k in np.arange(n[i]):
			error[i][d]=error[i][d]+(np.polyval(poly[d][i],x[i])[k]-y[i][k])**2
		MSE[i][d]=(1/n[i])*error[i][d]
	plt.figure(31+i)
	plt.plot(np.arange(10),MSE[i],'ro')
	plt.ylabel('MSE')
	plt.xlabel('Polynomial degree d')
	plt.title('Dataset with '+str(n[i])+' datapoints')
	plt.savefig('MSE for dataset with '+str(n[i])+' points against polynomial degree.png')
	
#plt.show()
