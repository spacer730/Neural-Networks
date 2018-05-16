import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

digitsoflines=[[] for i in range(10)] #Store which lines belong to which digits
Sum=[np.zeros(256) for i in range(10)] #Store sum of all training sets for each digit
c=[np.zeros(256) for i in range(10)] #Store the centers of each digit
r=np.zeros(10) #Store radius of each digit
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
		
with open('train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(10): #Find out which digit the row is
			for j in range(n[i]):
				if readtrain_in.line_num==digitsoflines[i][j]:
					if np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row))))>r[i]: #computer distance between center and datapoint
						r[i]=np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row)))) #Update if distance bigger than previous

distancematrix=np.zeros([10,10])

for i in range(10):
	for j in range(10):
		distancematrix[i][j]=np.dot(c[i]-c[j],c[i]-c[j]) #Computer distances between the centers
	print("The closest digit center between digit center "+str(i)+" is "+str(np.argmin(ma.array(distancematrix[i],mask=np.identity(10)[i])))) #Use masked array to ignore 0 selfdistance
	print("with distance: "+str(np.amin(ma.array(distancematrix[i],mask=np.identity(10)[i])))) #Minimum distance between a digit center and other digit centers excluding itself
	print("The radius of digit "+str(i)+" is "+str(r[i]))
	print("")

print("Thus digit 7 and 9 seem to be the hardest to differentiate from one another")
