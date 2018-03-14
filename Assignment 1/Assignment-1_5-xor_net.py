import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return np.maximum(0,x)

def leakyrelu(x):
	return np.maximum(0.1*x,x)

def xor_net(x1,x2,weights):
	y=np.dot(np.array([weights[0],weights[1]]),np.array([1,x1,x2]))

	y=list(map(lambda y: sigmoid(y), y))
	z=sigmoid(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))

	#y=list(map(lambda y: np.tanh(y), y))
	#z=np.tanh(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))

	#y=list(map(lambda y: relu(y), y))
	#z=relu(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))

	#y=list(map(lambda y: leakyrelu(y), y))
	#z=leakyrelu(np.dot(np.array(weights[2]),np.array([1,y[0],y[1]])))
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
		missclassified+=1
	if xor_net(1,0,weights)<=0.5:
		missclassified+=1
	if xor_net(1,1,weights)>0.5:
		missclassified+=1
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

def trainnetwork(learningrate,IN):
	np.random.seed(42)
	if IN == 'normal':
		weights=np.random.randn(3,3)
	if IN == 'uniform':
		weights=np.random.uniform(-1,1,9).reshape(3,3)
	counter=0
	mselist=[]
	missclassifiedlist=[]
	while counter < 4000:
		weights=weights-learningrate*grdmse(weights)
		mselist.append(mse(weights))
		missclassifiedlist.append(missclassified(weights))
		counter+=1
	return weights, mselist, missclassifiedlist

weights, mselist, missclassifiedlist = np.full((2,3,3,3), 0),np.zeros((2,3,4000)),np.zeros((2,3,4000))
learningrate = [0.1,0.25,0.5]
IN = ['normal','uniform']
for i in range (2):
	for j in range(3):
		weights[i][j], mselist[i][j], missclassifiedlist[i][j] = trainnetwork(learningrate[j],IN[i])

fig=plt.figure()
for i in range(2):
	for j in range(3):
			ax=fig.add_subplot(2,3,3*i+j+1, label="1")
			ax2=fig.add_subplot(2,3,3*i+j+1, label="2", frame_on=False)

			ax.plot(range(len(mselist[i][j])), mselist[i][j], color="C0")
			ax.set_xticks([0,2000,4000])
			ax.set_xlabel("Iterations", color="k")
			ax.set_ylabel("MSE", color="C0")
			ax.set_ylim([0,2])
			ax.tick_params(axis='y', colors="C0")

			ax2.plot(range(len(missclassifiedlist[i][j])), missclassifiedlist[i][j], color="C1")
			plt.text(1500, 2.8, r'$\eta=$'+str(learningrate[j]))
			ax2.set_xticks([0,2000,4000])
			ax2.yaxis.tick_right()
			ax2.set_ylabel('# missclassified units', color="C1")
			ax2.set_ylim([0,4])
			ax2.yaxis.set_label_position('right')
			ax2.tick_params(axis='y', colors="C1")

plt.subplots_adjust(hspace=0.3, wspace=1)
plt.savefig('Activationfunction sigmoid')
