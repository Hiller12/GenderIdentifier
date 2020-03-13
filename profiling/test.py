import filterTextsNew as filter
import featureExtractorNew as feature
import SVMmachineNew as SVMmachine
from sklearn import metrics
from sklearn import preprocessing
import numpy as npy
import pandas as pd

#filter.filter_texts("moduleTest/", "moduleTestFiltered/")
#feature.createParam("moduleTestFiltered/", "dictionary/", "moduleTestResults/", 'm')

#filter.filter_texts("texts/male/", "filteredTexts/male/")
#filter.filter_texts("texts/female/", "filteredTexts/female/")
#feature.createParam("filteredTexts/male/", "dictionary/", "analysisResults/male/", 'm')
#feature.createParam("filteredTexts/female/", "dictionary/", "analysisResults/female/", 'f')


models = SVMmachine.getModels("analysisResults/male/", "analysisResults/female/")
#filter.filter_texts("testTexts/male/", "filteredTestTexts/male/")
#feature.createParam("filteredTestTexts/male/", "dictionary/", "testAnalysis/male/", 'm')
#filter.filter_texts("testTexts/female/", "filteredTestTexts/female/")
#feature.createParam("filteredTestTexts/female/", "dictionary/", "testAnalysis/female/", 'f')
testdf1 = pd.read_csv("testAnalysis/male/Results.csv")
testdf2 = pd.read_csv("testAnalysis/female/Results.csv")
testdfSum = npy.vstack([testdf1.values, testdf2.values])
X_test = testdfSum[:, 1:7]
X_test = preprocessing.normalize(X_test)
for model in models:
    prediction = model.predict(X_test)
    print(models[model] + '\n' + metrics.classification_report(testdfSum[:, 7], prediction))
    print("_______________________________________________________________________")





