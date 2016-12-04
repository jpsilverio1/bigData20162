from   sklearn.ensemble import RandomForestClassifier
from   sklearn.ensemble import AdaBoostClassifier
from   sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import pandas as pd
import numpy as np
def classify(classifier, X,y, test):
	classifier.fit(X,y)
	classification = classifier.predict_proba(test)[:,1]
	return classification
def getFeatureColumns(dataframe):
    columns = dataframe.columns.tolist()
    return columns[:-1]
def saveClassification(classification, classificationFilePath):
	np.savetxt(classificationFilePath,classification, delimiter='\n')
trainDataframe = pd.read_csv('preprocessedData/preprocessed_train_file.csv')
featureColumns = getFeatureColumns(trainDataframe)
trainFeaturesDataframe = trainDataframe[featureColumns]
targetDataframe = trainDataframe['TARGET']
testDataframe = pd.read_csv('preprocessedData/preprocessed_test_file.csv')

'''
# bad - 75
classifier = RandomForestClassifier(n_estimators=200, n_jobs=5)
classification = classify(classifier, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results.csv')
#best so far 0.834001
classifier2  = AdaBoostClassifier(n_estimators=100)
classification = classify(classifier2, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_AdaBoost.csv') 
#second best 0.831095
classifier3  = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
classification = classify(classifier3, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_GradientBoost.csv') 
#horrible - 0.660939
classifier4  = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
classification = classify(classifier4, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_ExtraTreesClassifier.csv') '''
'''
classifier5  = svm.SVC()
classification = classify(classifier5, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_SVM.csv') 
'''



