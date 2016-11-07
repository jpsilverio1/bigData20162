# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
def allDifferent(list):
    return ( len(set(list)) == 1)
def checkFeatures(list):
    cont =0
    for i in range(len(list[0])):
        col = list[:,i]
        print col
        if (allDifferent(col)):
           print("all different {} ".format(i))
           cont +=1
    print cont
raw_data =  open("train_file.csv")
raw_data.readline()  # skip the header
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
print(dataset)
checkFeatures(dataset)
print len(dataset[0])
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]


