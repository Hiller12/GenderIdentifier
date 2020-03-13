import nltk
import pymorphy2
import os
import numpy as np
import pandas

def createParam(sourcePath, dicPath, destinationPath, dataGender):
    isCreatedMas = 0
    sPath = sourcePath
    dicAllnessTermsPath = dicPath
    desPath = destinationPath
    gender = dataGender
    morph = pymorphy2.MorphAnalyzer()
    dicAllnessTerms = open(dicAllnessTermsPath + "AllnessTermsDic.txt", 'r')
    AllnessTerms = nltk.word_tokenize(dicAllnessTerms.read())
    for textName in os.listdir(sPath):
        uniqueWords = []
        uniqueLongWords = []
        longWordsCnt = 0
        allnessTermsCnt = 0
        uniqueWordsCnt = 0
        negWords = 0
        nounCnt = 0
        nproCnt = 0
        file = open(sPath + textName, 'r')
        text = nltk.word_tokenize(file.read())
        textSize = len(text)
        for word in text:
            if (len(word) > 2):
                firstLetters = word[0] + word[1]
            pos = morph.parse(word)[0]
            if not (word in uniqueWords):
                uniqueWordsCnt += 1
                uniqueWords.append(word)
            if ((len(word) > 15) and not(word in uniqueLongWords)):
                longWordsCnt += 1
                uniqueLongWords.append(word)
            if word in AllnessTerms:
                allnessTermsCnt += 1
            elif ('NOUN' in pos.tag):
                nounCnt  += 1
                if (("Не" in firstLetters) or ("не"in firstLetters)):
                    negWords += 1
            elif (('ADJF' in pos.tag) or ('ADJS' in pos.tag)):
                if (("Не" in firstLetters) or ("не"in firstLetters)):
                    negWords += 1
            elif('ADVB' in pos.tag):
                if (("Не" in firstLetters) or ("не"in firstLetters)):
                    negWords += 1
            elif('PRCL' in pos.tag):
                negWords += 1
            elif('NPRO' in pos.tag):
                nproCnt += 1
        file.close()
        newRow = [(longWordsCnt / textSize), (allnessTermsCnt / textSize), (uniqueWordsCnt / textSize),
                  (negWords / textSize), (nproCnt / textSize), (nounCnt / textSize), gender]
        if (isCreatedMas == 0):
            dataArray = np.array(newRow)
            isCreatedMas = 1
        else:
            dataArray = np.vstack([dataArray, newRow])
        print(textName + " is processed \n")
    resultsDF = pandas.DataFrame(dataArray, index=None, columns=['longWordsCnt', 'allnessTermsCnt', 'uniqueWordsCnt',
                                                                 'negWords', 'nproCnt', 'nounCnt', 'gender'])
    #print(resultsDF)
    resultsDF.to_csv(desPath + "Results.csv")
    return 1
