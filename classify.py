from   sklearn.ensemble import RandomForestClassifier
from   sklearn.ensemble import AdaBoostClassifier
from   sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import svm
import pandas as pd
import numpy as np
import xgboost as xgb
def classify(classifier, X, y, testDataframe,cv=False, n_splits=2):
    if (cv == True):
        print "fazendo cross validation"
        #scores = cross_val_score(classifier,X,y,cv=20,scoring="roc_auc")
        #print scores
        kf = KFold(n_splits)
        KFold(n_splits, random_state=1, shuffle=True)

        #classification = np.zeros(len(X.index))
        auc_score = 0
        for train, test in kf.split(X,y):
            classifier.fit(X.ix[train],y.ix[train])
            classification = classifier.predict_proba(X.ix[test])[:,1]
            auc_score = auc_score + roc_auc_score(y.ix[train], classification)
        auc_score = auc_score / n_splits
        print 'auc_score'
        print auc_score
        
    print 'treinando...'
    classifier.fit(X,y)
    print 'terminei de treinar'
    classification = classifier.predict_proba(testDataframe)[:,1]
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
classifier = RandomForestClassifier(n_estimators=500, n_jobs=5)
classification = classify(classifier, trainFeaturesDataframe, targetDataframe,testDataframe, cv=True)
saveClassification(classification,'results2_randomForest.csv')

#best so far 0.834001 , 0.5 with kmeans centroids
classifier2  = AdaBoostClassifier(n_estimators=100)
classification = classify(classifier2, trainFeaturesDataframe, targetDataframe,testDataframe, cv=True)
saveClassification(classification,'results_AdaBoost2.csv')

#second best 0.831095, 0.5 with kmeans centroids
classifier3  = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
classification = classify(classifier3, trainFeaturesDataframe, targetDataframe,testDataframe, cv=True)
saveClassification(classification,'results_GradientBoost2.csv') 
'''
''''
#horrible - 0.660939
classifier4  = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
classification = classify(classifier4, trainFeaturesDataframe, targetDataframe,testDataframe, cv=True)
saveClassification(classification,'results_ExtraTreesClassifier2.csv')
''''''
classifier5  = svm.SVC(probability=True)
classification = classify(classifier5, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_SVM2.csv') 
'''
''''''
0.844381
classifier6  = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.01)
classification = classify(classifier6, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_XGBoost2.csv') 
'''''''''
classifier7  = xgb.XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.005,subsample=0.9)
classification = classify(classifier7, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_XGBoost3.csv')
#bad
classifier8  = xgb.XGBClassifier(n_estimators=400, max_depth=9, learning_rate=0.001,subsample=0.8,nthread=4)
classification = classify(classifier8, trainFeaturesDataframe, targetDataframe,testDataframe)
saveClassification(classification,'results_XGBoost4.csv')'''
'''
classifier8  = xgb.XGBClassifier(n_estimators=400, max_depth=9, learning_rate=0.005,nthread=4,subsample=0.9)
classification = classify(classifier8, trainFeaturesDataframe, targetDataframe,testDataframe, cv=True)
saveClassification(classification,'results_XGBoost7.csv')
'''



