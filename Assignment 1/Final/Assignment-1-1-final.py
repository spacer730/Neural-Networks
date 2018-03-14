import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv

# List for the ten digits, each containing list of the 256 values of each image
digitsoflines=[[] for i in range(10)]
# Array for calculating the centers of all 10 digits
Sum=[np.zeros(256) for i in range(10)] #Store sum of all training sets for each digit
# Array for coordinates of the center of each digit
c=[np.zeros(256) for i in range(10)]
# Radii of all digits in phase space
r=np.zeros(10)
# Number of images depicting digits in training data
n=np.zeros(10) 

# Compute number of training points for each digit 
with open('data/train_out.csv') as train_out:
	readtrain_out = csv.reader(train_out, delimiter=',')
	for row in readtrain_out:
		for i in range(10):
			if int(row[0])==i:			
				digitsoflines[i].append(readtrain_out.line_num)
n=[len(digitsoflines[i]) for i in range(10)] 

# Calculate array "Sum" to get coordinates of center for each digit
with open('data/train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(10): #Find out which digit the row is
			for j in range(n[i]):
				if readtrain_in.line_num==digitsoflines[i][j]:
					#Converts the string list row in to int list and adds to sum
					Sum[i]=Sum[i]+np.array(list(map(float, row))) 

# Compute the center of each digit
for i in range(10):
	c[i]=Sum[i]/n[i]

# Compute distance of image to its center in "r" and update value if the new value is bigger
# Thus we end up with the highest radius in the end for all ten digits
with open('data/train_in.csv') as train_in:
	readtrain_in = csv.reader(train_in, delimiter=',')
	for row in readtrain_in:
		for i in range(10): #Find out which digit the row is
			for j in range(n[i]):
				if readtrain_in.line_num==digitsoflines[i][j]:
					if np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row))))>r[i]:
						r[i]=np.dot(c[i]-np.array(list(map(float,row))),c[i]-np.array(list(map(float,row))))

# Array for distances between the 10 centers
distancematrix=np.zeros([10,10])

# Calculate distances between the 10 centers and print out the distance to the nearest center and its resp. digit
for i in range(10):
	for j in range(10):
		distancematrix[i][j]=np.dot(c[i]-c[j],c[i]-c[j])
	print("The closest digit center between digit center "+str(i)+" is "+str(np.argmin(ma.array(distancematrix[i],mask=np.identity(10)[i])))) #Use masked array to ignore 0 selfdistance
	print("with distance: "+str(np.amin(ma.array(distancematrix[i],mask=np.identity(10)[i])))) #Minimum distance between a digit center and other digit centers excluding itself
	print("The radius of digit", str(i), "is", str(r[i]), "\n")

print("Thus digit 7 and 9 seem to be the hardest to differentiate from one another")
