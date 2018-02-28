import numpy as np
import numpy.ma as ma
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

#convert to numpy arrays so we can easily calculate mean and standard deviation
activation1=np.array(activation1)
activation8=np.array(activation8)

#Compute the average activation of the digits
average1=np.mean(activation1)
average8=np.mean(activation8)

std1=np.std(activation1)
std8=np.std(activation8)

labda=std1/(std1+std8)
boundary=average1+std1+(abs(average1-average8)-std1-std8)*labda

trainpredictionactivation=[]
#Compute activation for the row if it is digit 1 or 8 and then if it is larger than boundary it is 8 and smaller is 1
with open('train_in.csv') as train_in:
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

#Compare predicted digits with real digits and determine accuracy
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

with open('test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		for i in range(10):
			if int(row[0])==i:
				testdigitsoflines[i].append(readtest_out.line_num) #Read out which lines belong to which digits

ntest=[len(testdigitsoflines[i]) for i in range(10)] #Compute number of training points for each digit

testpredictionactivation=[]
#Compute activation for the row if it is digit 1 or 8 and then if it is larger than boundary it is 8 and smaller is 1
with open('test_in.csv') as test_in:
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
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					testpredictionactivation.append([readtest_in.line_num,1])
				else:
					testpredictionactivation.append([readtest_in.line_num,8])

testtrue=[]
testcounteractivation1=0
testcounteractivation8=0

#Compare predicted digits with real digits and determine accuracy
for i in range(len(testpredictionactivation)):
	if testpredictionactivation[i][1]==1:
		for j in range(len(testdigitsoflines[1])):
			if testpredictionactivation[i][0]==testdigitsoflines[1][j]:
				testcounteractivation1+=1
	elif testpredictionactivation[i][1]==8:
		for j in range(len(testdigitsoflines[8])):
			if testpredictionactivation[i][0]==testdigitsoflines[8][j]:
				testcounteractivation8+=1

testaccuracy1=testcounteractivation1/ntest[1]
testaccuracy8=testcounteractivation8/ntest[8]
testaccuracy=(testcounteractivation1+testcounteractivation8)/(ntest[1]+ntest[8])

print("The test accuracy for correctly classifying the digit 1 is "+str(testaccuracy1))
print("The test accuracy for correctly classifying the digit 8 is "+str(testaccuracy8))
print("The test accuracy for correctly classifying the digits 1 and 8 is "+str(testaccuracy))
