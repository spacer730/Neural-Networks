import numpy as np
import numpy.ma as ma
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

#We want to distinguish between 1 and 8. The X we will compute is the number of activated pixels. We will first compute the average activation number for 1 and 8 and then determine a good boundary between them. If the number X for a certain row is smaller than this boundary we will predict it is a 1 and if it is larger than the boundary we will predict it as an 8.

activation1=[]
activation8=[]

#Check for each row in train_in whether it is a 1 or an 8 and store the sum of the pixel values.
with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(n[1]):
			if readtrain_in.line_num==digitsoflines[1][i]:
				computeactivation1=0
				for j in range(256):
					computeactivation1+=float(row[j])
				activation1.append(computeactivation1)

		for i in range(n[8]):
			if readtrain_in.line_num==digitsoflines[8][i]:
				computeactivation8=0
				for j in range(256):
					computeactivation8+=float(row[j])
				activation8.append(computeactivation8)

print(n[1])
print(len(activation1))
print(n[8])
print(len(activation8))
	
#convert to numpy arrays so we can easily calculate mean and standard deviation
activation1=np.array(activation1)
activation8=np.array(activation8)

#Compute the average activation of the digits
average1=np.mean(activation1)
average8=np.mean(activation8)

std1=np.std(activation1)
std8=np.std(activation8)

boundary=average1+std1+(abs(average1-average8)-std1-std8)/2

trainpredictionactivation=[]
with open('train_in.csv') as train_in: #Compute activation for the row if it is digit 1 or 8 and then if it is larger than boundary it is 8 and smaller is 1
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		activation=0
		for i in range(n[1]):
			if readtrain_in.line_num==digitsoflines[1][i]:
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					trainpredictionactivation.append([readtrain_in.line_num,1])
				else:
					trainpredictionactivation.append([readtrain_in.line_num,8])
		
		for i in range(n[8]):
			if readtrain_in.line_num==digitsoflines[8][i]:
				for j in range(256):
					activation+=float(row[j])
				if activation<boundary:
					trainpredictionactivation.append([readtrain_in.line_num,1])
				else:
					trainpredictionactivation.append([readtrain_in.line_num,8])


traintrue=[]
traincounteractivation=0

print(trainpredictionactivation[0][0])
print(digitsoflines)

for i in range(len(trainpredictionactivation)):
	if trainpredictionactivation[i][1]==digitsoflines[trainpredictionactivation[i][0]-1]:
		traincounteractivation+=1

accuracy=traincounteractivation/len(trainpredictionactivation)

print(accuracy)

print(average1)
print(average8)

print(std1)
print(std8)

print(boundary)


#print(activation1)
#print(activation8)
