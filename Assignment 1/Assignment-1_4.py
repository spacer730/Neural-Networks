import numpy as np
import numpy.ma as ma
import csv

traindigitsoflines=[] #Store which lines belong to which digits

with open('train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		traindigitsoflines.append([readtrain_out.line_num,int(row[0])]) #Read out which lines belong to which digits

weights=np.random.randn(10,257)

x_train=[[] for i in range(len(traindigitsoflines))]
y_train=[[] for i in range(len(traindigitsoflines))]

#We read in the training data and store it in x_train as row vectors with length 257
with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		x_train[readtrain_in.line_num-1].extend([1.]) #First term is for the bias
		x_train[readtrain_in.line_num-1].extend(list(map(float,row)))
		#x_train[readtrain_in.line_num-1]=np.array(x_train[readtrain_in.line_num-1])
		y_train[readtrain_in.line_num-1].extend(np.dot(weights,x_train[readtrain_in.line_num-1]))

#Store the index of the misclassified data for the training set
misclassified=[]
for i in range(len(traindigitsoflines)):
	if np.argmax(y_train[i])!=traindigitsoflines[i][1]:
		misclassified.append(i)

iterationcounter=0

#Keep updating weights untill there are no more misclassified examples
while len(misclassified) != 0:
	iterationcounter+=1
	#pick random sample out of misclassified set
	sample=np.random.choice(misclassified)
	#train the weights for the sample untill the max of the output vector is equal to the digit
	while np.argmax(y_train[sample])!=traindigitsoflines[sample][1]:
		for j in [k for k in range(10) if k!=traindigitsoflines[sample][1]]:
			if y_train[sample][j]>=y_train[sample][traindigitsoflines[sample][1]]:
				weights[j]-=x_train[sample]
		weights[traindigitsoflines[sample][1]]+=x_train[sample]
		y_train[sample]=np.dot(weights,x_train[sample])
	
	#Update predictions with new weights
	for i in range(len(traindigitsoflines)):
		y_train[i]=np.dot(weights,x_train[i])

	#Store the index of the misclassified data for the training set with the update weights
	misclassified=[]
	for i in range(len(traindigitsoflines)):
		if np.argmax(y_train[i])!=traindigitsoflines[i][1]:
			misclassified.append(i)

print(iterationcounter)

trainprediction=[]
#Store the predictions from the training set with the final weights
for i in range(len(traindigitsoflines)):
	trainprediction.append(np.argmax(y_train[i]))

traincounter=0
for i in range(len(traindigitsoflines)):
	if trainprediction[i]==traindigitsoflines[i][1]:
		traincounter+=1

trainaccuracy=traincounter/len(traindigitsoflines)

print(trainaccuracy)
print(traincounter)

testdigitsoflines=[] #Store which lines belong to which digits

with open('test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		testdigitsoflines.append([readtest_out.line_num,int(row[0])]) #Read out which lines belong to which digits

x_test=[[] for i in range(len(testdigitsoflines))]
y_test=[[] for i in range(len(testdigitsoflines))]

#We read in the test data and store it in x_test as row vectors with length 257
with open('test_in.csv') as test_in:
	readtest_in = csv.reader(test_in, delimiter=',')
	for row in readtest_in:
		x_test[readtest_in.line_num-1].extend([1.]) #First term is for the bias
		x_test[readtest_in.line_num-1].extend(list(map(float,row)))
		#x_train[readtrain_in.line_num-1]=np.array(x_train[readtrain_in.line_num-1])
		y_test[readtest_in.line_num-1].extend(np.dot(weights,x_test[readtest_in.line_num-1]))

testprediction=[np.argmax(y_test[i]) for i in range(len(y_test))]

testcounter=0
for i in range(len(testdigitsoflines)):
	if testprediction[i]==testdigitsoflines[i][1]:
		testcounter+=1

testaccuracy=testcounter/len(testdigitsoflines)

print(testaccuracy)
print(testcounter)
