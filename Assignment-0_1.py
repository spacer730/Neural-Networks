import numpy as np
import matplotlib.pyplot as plt

x_1=[0,0,1,1]
x_2=[0,1,0,1]
X=np.matrix([x_1,x_2])
AND=[0,0,0,1]
XOR=[0,1,1,0]

plt.figure(1)
plt.plot(X[0],X[1], 'ro')
plt.title('XOR function')
for i in np.arange(np.shape(X)[1]):
	plt.annotate(str(XOR[i]),xy=(x_1[i],x_2[i]))

plt.figure(2)
plt.plot(X[0],X[1], 'ro')
plt.title('AND function')
for i in np.arange(np.shape(X)[1]):
	plt.annotate(str(AND[i]), xy=(x_1[i],x_2[i]))

plt.show()
