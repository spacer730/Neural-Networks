import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KernelDensity

# List for the ten digits, each containing list of the 256 values of each image
traindigitsoflines=[[] for i in range(10)]
# Number of training points for each digit
ntrain=np.zeros(10)

# Read in data from training set
with open('data/train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		for i in range(10):
			if int(row[0])==i:
				traindigitsoflines[i].append(readtrain_out.line_num)

# Compute number of training points for each digit
ntrain=[len(traindigitsoflines[i]) for i in range(10)] 

#We want to distinguish between 1 and 8. The X we will compute is the number of activated pixels.
# We will first compute P(C1|X) and P(C8|X) to determine a good boundary between them.
# If the number X for a certain row is smaller than this boundary, we will predict it is a 1
# and if it is larger than the boundary we will predict it is an 8.
activation1=[]
activation8=[]

# Check for each row in train_in whether it is a 1 or an 8 and store the sum of the pixel values
with open('data/train_in.csv') as train_in:
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
	activation1=np.array(activation1)
	activation8=np.array(activation8)

#Use Bayes theorem: P(C1|X)=P(X|C1)*P(C1)
P_C1=ntrain[1]/(ntrain[1]+ntrain[8])
P_C8=ntrain[8]/(ntrain[1]+ntrain[8])

P_X_C1, bins1 = np.histogram(activation1, range=[-250,0], density=True, bins=50)
widths1 = np.diff(bins1)
P_C1_X = P_X_C1 * P_C1

P_X_C8, bins8 = np.histogram(activation8, range=[-250,0], density=True, bins=50)
widths8 = np.diff(bins8)
P_C8_X = P_X_C8 * P_C8

p1=plt.bar(bins1[:-1], P_C1_X, widths1, alpha=0.5, color="b", label="P(C1|X)")
p8=plt.bar(bins8[:-1], P_C8_X, widths8, alpha=0.5, color="r", label="P(C8|X)")
plt.legend()
plt.xlabel('Activation sum X')
plt.ylabel('P(C|X)')
plt.axvline(x=-141.7, c="g", lw=2)
plt.savefig('Histograms')

# When P_C8_X=P_C1_X we have the boundary, but since we determined the probabilities from a histogram they are not continuous.
# So the criterium we use is that they lie very close to each other.
for i in range(len(P_C1_X)):
	if abs(P_C8_X[i]-P_C1_X[i])<0.0008 and P_C1_X[i]!=0 and P_C8_X[i]!=0:
		boundary=bins1[i]

# Compute activation for the row if it is digit 1 or 8 and then if it is larger than boundary it is 8 and smaller is 1
trainpredictionactivation=[]
with open('data/train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		activation=0
		for i in range(ntrain[1]):
			if readtrain_in.line_num==traindigitsoflines[1][i]:
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					trainpredictionactivation.append([readtrain_in.line_num,1])
				else:
					trainpredictionactivation.append([readtrain_in.line_num,8])

		for i in range(ntrain[8]):
			if readtrain_in.line_num==traindigitsoflines[8][i]:
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					trainpredictionactivation.append([readtrain_in.line_num,1])
				else:
					trainpredictionactivation.append([readtrain_in.line_num,8])

traintrue=[]
traincounteractivation1=0
traincounteractivation8=0

# Compare predicted digits with real digits and determine accuracy
for i in range(len(trainpredictionactivation)):
	if trainpredictionactivation[i][1]==1:
		for j in range(len(traindigitsoflines[1])):
			if trainpredictionactivation[i][0]==traindigitsoflines[1][j]:
				traincounteractivation1+=1
	elif trainpredictionactivation[i][1]==8:
		for j in range(len(traindigitsoflines[8])):
			if trainpredictionactivation[i][0]==traindigitsoflines[8][j]:
				traincounteractivation8+=1

trainaccuracy1=traincounteractivation1/ntrain[1]
trainaccuracy8=traincounteractivation8/ntrain[8]
trainaccuracy=(traincounteractivation1+traincounteractivation8)/(ntrain[1]+ntrain[8])

print("The train accuracy for correctly classifying the digit 1 is "+str(trainaccuracy1))
print("The train accuracy for correctly classifying the digit 8 is "+str(trainaccuracy8))
print("The train accuracy for correctly classifying the digits 1 and 8 is "+str(trainaccuracy))

testdigitsoflines=[[] for i in range(10)] #Store which lines belong to which digits
ntest=np.zeros(10) #Store number of training points for each digit

with open('data/test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		for i in range(10):
			if int(row[0])==i:
				testdigitsoflines[i].append(readtest_out.line_num) # Read out which lines belong to which digits

ntest=[len(testdigitsoflines[i]) for i in range(10)] # Compute number of training points for each digit

# Compute activation for the row if it is digit 1 or 8 and then if it is larger than boundary it is 8 and smaller is 1
testpredictionactivation=[]
with open('data/test_in.csv') as test_in:
	readtest_in = csv.reader(test_in, delimiter=',')
	for row in readtest_in:
		activation=0
		for i in range(ntest[1]):
			if readtest_in.line_num==testdigitsoflines[1][i]:
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					testpredictionactivation.append([readtest_in.line_num,1])
				else:
					testpredictionactivation.append([readtest_in.line_num,8])
		
		for i in range(ntest[8]):
			if readtest_in.line_num==testdigitsoflines[8][i]:
				for j in  range(256):
					activation+=float(row[j])
				if activation<boundary:
					testpredictionactivation.append([readtest_in.line_num,1])
				else:
					testpredictionactivation.append([readtest_in.line_num,8])

testtrue=[]
testcounteractivation1=0
testcounteractivation8=0

# Compare predicted digits with real digits and determine accuracy
for i in range(len(testpredictionactivation)):
	if testpredictionactivation[i][1]==1:
		for j in range(len(testdigitsoflines[1])):
			if testpredictionactivation[i][0]==testdigitsoflines[1][j]:
				testcounteractivation1+=1
	elif testpredictionactivation[i][1]==8:
		for j in range(len(testdigitsoflines[8])):
			if testpredictionactivation[i][0]==testdigitsoflines[8][j]:
				testcounteractivation8+=1

testaccuracy1 = testcounteractivation1/ntest[1]
testaccuracy8 = testcounteractivation8/ntest[8]
testaccuracy = (testcounteractivation1+testcounteractivation8)/(ntest[1]+ntest[8])

print("The test accuracy for correctly classifying the digit 1 is "+str(testaccuracy1))
print("The test accuracy for correctly classifying the digit 8 is "+str(testaccuracy8))
print("The test accuracy for correctly classifying the digits 1 and 8 is "+str(testaccuracy))
