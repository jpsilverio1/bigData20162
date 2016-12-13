import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans
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
def removeColumnsApplyKmeansGenerateNewdatasets(trainDatasetFilePath, testDatasetFilePath, newTrainDatasetFilePath, newTestDatasetFilePath, columnsToRemove):
    trainDataframe = pd.read_csv(trainDatasetFilePath)
    testDataframe = pd.read_csv(testDatasetFilePath)
    trainDataframe.drop(columnsToRemove, inplace = True, axis=1)
    testDataframe.drop(columnsToRemove, inplace = True, axis=1)

    featureColumns = getFeatureColumns(trainDataframe)
    trainFeaturesDataframe = trainDataframe[featureColumns]
    X = trainFeaturesDataframe
    y = trainDataframe['TARGET']

    print "rodando kmeans"
    kmeans = KMeans(n_clusters=3, random_state=0)
    distancesToCentroidMatrixTrainData = kmeans.fit_transform(X,y)
    
    count = 0
    for centroidVector in kmeans.cluster_centers_:
        distancesToCentroidTestData = []
        distancesToCentroidTrainData = []
        for row in X.as_matrix():
            rowVector = row[:len(featureColumns)]
            dist = np.linalg.norm(rowVector-centroidVector)
            distancesToCentroidTrainData.append(dist)
        for row2 in testDataframe.as_matrix(): 
            rowVector2 = row[:len(featureColumns)]
            dist2 = np.linalg.norm(rowVector2-centroidVector)
            distancesToCentroidTestData.append(dist2)
        testDataframe['centroid' + str(count)] = distancesToCentroidTestData
        trainDataframe['centroid' + str(count)] = distancesToCentroidTrainData
        count += 1
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
columnsWithHighCorrelation =  calculateCorrelationBetweenColumnsInDataset(normalizedTrainDataframe, featureColumnsWithoutColumnsToRemove,  0.05)
print
print columnsWithHighCorrelation
print len(columnsToRemoveFromNormalizedDataset)
print len(columnsWithHighCorrelation)
#this list can have repeated elements
allColumnsToRemove = columnsToRemoveFromNormalizedDataset + columnsWithHighCorrelation
#writeFileFromList('allColumnsToRemove.txt', allColumnsToRemove)
print len(set(allColumnsToRemove))
print len(featureColumns)
removeColumnsApplyKmeansGenerateNewdatasets(originalTrainDataset, originalTestDataset, 'preprocessedData/preprocessed_train_file2.csv', 'preprocessedData/preprocessed_test_file2.csv', allColumnsToRemove)
