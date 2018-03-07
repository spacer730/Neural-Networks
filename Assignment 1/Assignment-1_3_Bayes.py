import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

traindigitsoflines=[[] for i in range(10)] #Store which lines belong to which digits
ntrain=np.zeros(10) #Store number of training points for each digit

with open('train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		for i in range(10):
			if int(row[0])==i:
				traindigitsoflines[i].append(readtrain_out.line_num) #Read out which lines belong to which digits

ntrain=[len(traindigitsoflines[i]) for i in range(10)] #Compute number of training points for each digit

#We want to distinguish between 1 and 8. The X we will compute is the number of activated pixels. We will first compute the average activation number for 1 and 8 and then determine a good boundary between them. If the number X for a certain row is smaller than this boundary we will predict it is a 1 and if it is larger than the boundary we will predict it as an 8.

activation1=[]
activation8=[]

#Check for each row in train_in whether it is a 1 or an 8 and store the sum of the pixel values.
with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(ntrain[1]):
			if readtrain_in.line_num==traindigitsoflines[1][i]:
				computeactivation1=0
				for j in range(256):
					computeactivation1+=float(row[j])
				activation1.append(computeactivation1)

		for i in range(ntrain[8]):
			if readtrain_in.line_num==traindigitsoflines[8][i]:
				computeactivation8=0
				for j in range(256):
					computeactivation8+=float(row[j])
				activation8.append(computeactivation8)

P_C1=ntrain[1]/(ntrain[1]+ntrain[8])
P_C2=ntrain[8]/(ntrain[1]+ntrain[8])

print(P_C1)
print(P_C2)
print(activation1)

plt.hist(activation1, normed=True, bins=30)
plt.xlabel('feature')
plt.ylabel('Probability for digit 1')
plt.show()
