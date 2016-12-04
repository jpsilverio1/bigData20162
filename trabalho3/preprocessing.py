import numpy as np
import pandas as pd
import csv
def getColumnsToRemoveWithZeroStd(dataframe,featureColumns):
    columnsToRemove = []
    for column in featureColumns:
        if np.std(dataframe[column]) == 0:
            columnsToRemove.append(column)
    return columnsToRemove
def getFeatureColumns(dataframe):
    columns = dataframe.columns.tolist()
    return columns[:-1]
def calculateCorrelationBetweenColumnsInDataset(dataframe, featureColumns, epsilon):
    columnsWithHighCorrelation = []
    myfile = open("correlations.txt",'wb')
    for i in range(len(featureColumns)):
            col = dataframe[featureColumns[i]]
            for j in range(i+1, len(featureColumns)):
                pearsonCorrelation = np.corrcoef(col, dataframe[featureColumns[j]])[0][1]
                myfile.write(featureColumns[i] + " " + featureColumns[j] + " " + str(pearsonCorrelation) + " \n")
                if ((1 - abs(pearsonCorrelation) <= epsilon)):
                    columnsWithHighCorrelation.append(featureColumns[i])
    myfile.close()
    return columnsWithHighCorrelation
def writeFileFromList(fileName,mylist):
    myfile = open(fileName, 'wb')
    wr = csv.writer(myfile)
    wr.writerow(mylist)
def removeColumnsAndGenerateNewdatasets(trainDatasetFilePath, testDatasetFilePath, newTrainDatasetFilePath, newTestDatasetFilePath, columnsToRemove):
    trainDataframe = pd.read_csv(trainDatasetFilePath)
    testDataframe = pd.read_csv(testDatasetFilePath)
    trainDataframe.drop(columnsToRemove, inplace = True, axis=1)
    testDataframe.drop(columnsToRemove, inplace = True, axis=1)
    trainDataframe.to_csv(newTrainDatasetFilePath,index=False)
    testDataframe.to_csv(newTestDatasetFilePath,index=False)
originalTrainDataset = 'datasets/train_file.csv'
originalTestDataset = 'datasets/test_file.csv'
normalizedTrainDataset = 'datasets/normalized_train_file.csv'
normalizedTrainDataframe = pd.read_csv(normalizedTrainDataset)
featureColumns = getFeatureColumns(normalizedTrainDataframe)
columnsToRemoveFromNormalizedDataset = getColumnsToRemoveWithZeroStd(normalizedTrainDataframe, featureColumns)
featureColumnsWithoutColumnsToRemove = [x for x in featureColumns if x not in columnsToRemoveFromNormalizedDataset]
print len(featureColumns)
print len(featureColumnsWithoutColumnsToRemove)
print columnsToRemoveFromNormalizedDataset
columnsWithHighCorrelation =  calculateCorrelationBetweenColumnsInDataset(normalizedTrainDataframe, featureColumnsWithoutColumnsToRemove,  0.01)
print
print columnsWithHighCorrelation
print len(columnsToRemoveFromNormalizedDataset)
print len(columnsWithHighCorrelation)
#this list can have repeated elements
allColumnsToRemove = columnsToRemoveFromNormalizedDataset + columnsWithHighCorrelation
#writeFileFromList('allColumnsToRemove.txt', allColumnsToRemove)
print len(set(allColumnsToRemove))
print len(featureColumns)
#removeColumnsAndGenerateNewdatasets(originalTrainDataset, originalTestDataset, 'preprocessedData/preprocessed_train_file.csv', 'preprocessedData/preprocessed_test_file.csv', allColumnsToRemove)

