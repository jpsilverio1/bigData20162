import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
def normalizeDatasetByColumns(dataset):
    return normalize(dataset, axis = 0)
def getDatasetFromCSV(fileName):
    raw_data =  open(fileName)
    header = raw_data.readline()  
    return [header, np.loadtxt(raw_data, delimiter=",")]
def generateNormalizedDataset(datasetFilePath, normalizedDatasetFilePath):
	header, dataset = getDatasetFromCSV(datasetFilePath)
	columns_normed = normalizeDatasetByColumns(dataset)
	np.savetxt(normalizedDatasetFilePath, columns_normed, header = header, delimiter=',', comments='')

generateNormalizedDataset("datasets/train_file.csv", 'datasets/normalized_train_file.csv')


