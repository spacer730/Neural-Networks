import numpy as np
import numpy.ma as ma
from sklearn.metrics import confusion_matrix
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

trainprediction=[]
		
with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		dist=[]
		for i in range(10): #Find out which digit the row is
			dist.append(np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row)))))
		trainprediction.append(np.argmin(dist))

print(len(trainprediction))

traintrue=[]
counter=0

with open('train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		traintrue.append(int(row[0]))
		if int(row[0])==trainprediction[readtrain_out.line_num-1]:
			counter+=1

print(confusion_matrix(trainprediction,traintrue))

print(counter)
print(100*counter/len(trainprediction))

testprediction=[]
		
with open('test_in.csv') as test_in:
	readtest_in = csv.reader(test_in, delimiter=',')
	for row in readtest_in:
		dist=[]
		for i in range(10): #Find out which digit the row is
			dist.append(np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row)))))
		testprediction.append(np.argmin(dist))

print(len(testprediction))

testtrue=[]
counter=0

counter2=0
with open('test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		testtrue.append(int(row[0]))
		if int(row[0])==testprediction[readtest_out.line_num-1]:
			counter2+=1

print(confusion_matrix(testprediction,testtrue))

print(counter2)
print(100*counter2/len(testprediction))
