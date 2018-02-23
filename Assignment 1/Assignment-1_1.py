import numpy as np
import matplotlib.pyplot as plt
import csv

C=[[] for i in range(10)] #Here we store which lines belong to which digits

with open('test_out.csv') as test_out:
	readtest_out = csv.reader(test_out, delimiter=',')
	for row in readtest_out:
		for i in range(10):
			if int(row[0])==i:			
				C[i].append(readtest_out.line_num)

Sum=[[] for i in range(10)] #Here we calculate the sum of all training sets for each digit

with open('test_in.csv') as test_in:
	readtest_in = csv.reader(test_in, delimiter=',')
	for row in readtest_in:
		for i in range(10):
			for j in range(len(C[i])):
				if readtest_in.line_num==C[i][j]: #Checks if the linenum has output i
					Sum[i]+=list(map(float, row)) #Converts the row string in to int string and adds to sum

print(len(Sum[0]))

#center=[Sum[i]/(len(C[i]) for i in range(10)] #Here we compute the center of each digit from the training sets
