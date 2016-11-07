
from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
import csv

def allEqual(list):
    return ( len(set(list)) == 1)
def checkFeatures(list):
    cont =0
    equalColumns = []
    for i in range(len(list[0])):
        col = list[:,i]
        if (allEqual(col)):
           print("all different {} ".format(i))
           equalColumns.append(i)
           cont +=1
    print cont
    return equalColumns

def normalizeMatrixByColumns(matrix):
    newMatrix = []
    rowVector = []
    for i in range(len(matrix[0])):
        col = matrix[:,i]
        maxValue = np.amax(col)
        if (maxValue != 0):
            for j in range(len(col)):
                rowVector.append(col[j]/maxValue)
        newMatrix.append(rowVector)
        rowVector = []
    return newMatrix
def normalizeDatasetByColumns(dataset):
	return normalize(dataset, axis = 0)
def getDatasetFromCSV(fileName):
	raw_data =  open(fileName)
	header = raw_data.readline()  
	print header
	# load the CSV file as a numpy matrix
	return np.loadtxt(raw_data, delimiter=",")
def writeFileFromList(fileName,mylist):
	myfile = open(fileName, 'wb')
	wr = csv.writer(myfile)
	wr.writerow(mylist)
dataset = getDatasetFromCSV("train_file.csv")
print(dataset.shape)
print(dataset)
columns_To_Remove = checkFeatures(dataset)
writeFileFromList("columnsToRemove.csv", columns_To_Remove);
columns_normed = normalizeDatasetByColumns(dataset)
#np.savetxt('normalizedTrainDataset.csv', columns_normed, delimiter=',')
print "normalized"
print (columns_normed)

