import numpy as np
import numpy.ma as ma
from sklearn.metrics import confusion_matrix
import sklearn
import matplotlib.pyplot as plt
import csv

digitsoflines=[[] for i in range(10)] #Store which lines belong to which digits
Sum=[np.zeros(256) for i in range(10)] #Store sum of all training sets for each digit
c=[np.zeros(256) for i in range(10)] #Store the centers of each digit
n=np.zeros(10) #Store number of training points for each digit

with open('train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		for i in range(10):
			if int(row[0])==i:
				digitsoflines[i].append(readtrain_out.line_num) #Read out which lines belong to which digits

n=[len(digitsoflines[i]) for i in range(10)] #Compute number of training points for each digit

with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(10): #Find out which digit the row is
			for j in range(n[i]):
				if readtrain_in.line_num==digitsoflines[i][j]:
					Sum[i]=Sum[i]+np.array(list(map(float, row))) #Converts the string list row in to int list and adds to sum

for i in range(10):
	c[i]=Sum[i]/n[i] #Compute the center of each digit by dividing the sum of all the training points by the number of training points for each digit

trainpredictioneuclid=[]
trainpredictionmanhattan=[]
trainpredictioncosine=[]
trainpredictioncorrelation=[]
		
with open('train_in.csv') as train_in: #Compute distance between centers and data point and prediction is closest center
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		disteuclid=[]
		distmanhattan=[]
		distcosine=[]
		distcorrelation=[]
		for i in range(10):
			disteuclid.append(np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row)))))
			distmanhattan.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='manhattan'))
			distcosine.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='cosine'))
			distcorrelation.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='correlation'))
			
		trainpredictioneuclid.append(np.argmin(disteuclid))
		trainpredictionmanhattan.append(np.argmin(distmanhattan))
		trainpredictioncosine.append(np.argmin(distcosine))
		trainpredictioncorrelation.append(np.argmin(distcorrelation))

traintrue=[]
traincountereuclid=0
traincountermanhattan=0
traincountercosine=0
traincountercorrelation=0

with open('train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		traintrue.append(int(row[0]))
		if int(row[0])==trainpredictioneuclid[readtrain_out.line_num-1]:
			traincountereuclid+=1
		if int(row[0])==trainpredictionmanhattan[readtrain_out.line_num-1]:
			traincountermanhattan+=1
		if int(row[0])==trainpredictioncosine[readtrain_out.line_num-1]:
			traincountercosine+=1
		if int(row[0])==trainpredictioncorrelation[readtrain_out.line_num-1]:
			traincountercorrelation+=1

print("Confusion matrix for training set using euclidean distance:")
print(confusion_matrix(trainpredictioneuclid,traintrue))
print("Confusion matrix for training set using manhattan distance:")
print(confusion_matrix(trainpredictionmanhattan,traintrue))
print("Confusion matrix for training set using cosine distance:")
print(confusion_matrix(trainpredictioncosine,traintrue))
print("Confusion matrix for training set using correlation distance:")
print(confusion_matrix(trainpredictioncorrelation,traintrue))

print("The accuracy of the euclidean distance algorithm on the training set is:")
print(100*traincountereuclid/len(trainpredictioneuclid))
print("The accuracy of the manhattan distance algorithm on the training set is:")
print(100*traincountermanhattan/len(trainpredictionmanhattan))
print("The accuracy of the cosine distance algorithm on the training set is:")
print(100*traincountercosine/len(trainpredictioncosine))
print("The accuracy of the correlation distance algorithm on the training set is:")
print(100*traincountercorrelation/len(trainpredictioncorrelation))

testpredictioneuclid=[]
testpredictionmanhattan=[]
testpredictioncosine=[]
testpredictioncorrelation=[]

with open('test_in.csv') as test_in: #Compute distance between centers and data point and prediction is closest center
	readtest_in = csv.reader(test_in, delimiter=',')
	for row in readtest_in:
		disteuclid=[]
		distmanhattan=[]
		distcosine=[]
		distcorrelation=[]
		for i in range(10):
			disteuclid.append(np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row)))))
			distmanhattan.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='manhattan'))
			distcosine.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='cosine'))
			distcorrelation.append(sklearn.metrics.pairwise.pairwise_distances([c[i]],[np.array(list(map(float,row)))],metric='correlation'))
			
		testpredictioneuclid.append(np.argmin(disteuclid))
		testpredictionmanhattan.append(np.argmin(distmanhattan))
		testpredictioncosine.append(np.argmin(distcosine))
		testpredictioncorrelation.append(np.argmin(distcorrelation))

testtrue=[]
testcountereuclid=0
testcountermanhattan=0
testcountercosine=0
testcountercorrelation=0

with open('test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		testtrue.append(int(row[0]))
		if int(row[0])==testpredictioneuclid[readtest_out.line_num-1]:
			testcountereuclid+=1
		if int(row[0])==testpredictionmanhattan[readtest_out.line_num-1]:
			testcountermanhattan+=1
		if int(row[0])==testpredictioncosine[readtest_out.line_num-1]:
			testcountercosine+=1
		if int(row[0])==testpredictioncorrelation[readtest_out.line_num-1]:
			testcountercorrelation+=1

print("Confusion matrix for test set using euclidean distance:")
print(confusion_matrix(testpredictioneuclid,testtrue))
print("Confusion matrix for test set using manhattan distance:")
print(confusion_matrix(testpredictionmanhattan,testtrue))
print("Confusion matrix for test set using cosine distance:")
print(confusion_matrix(testpredictioncosine,testtrue))
print("Confusion matrix for test set using correlation distance:")
print(confusion_matrix(testpredictioncorrelation,testtrue))

print("The accuracy of the euclidean distance algorithm on the test set is:")
print(100*testcountereuclid/len(testpredictioneuclid))
print("The accuracy of the manhattan distance algorithm on the test set is:")
print(100*testcountermanhattan/len(testpredictionmanhattan))
print("The accuracy of the cosine distance algorithm on the test set is:")
print(100*testcountercosine/len(testpredictioncosine))
print("The accuracy of the correlation distance algorithm on the test set is:")
print(100*testcountercorrelation/len(testpredictioncorrelation))
