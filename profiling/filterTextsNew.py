import nltk
import pymorphy2
import string as str
import os


def filter_texts(sourcePath, destinationPath):
    sPath = sourcePath
    dPath = destinationPath
    morph = pymorphy2.MorphAnalyzer()
    for textName in os.listdir(sPath):
        f = open(sPath + textName, 'r')
        outF = open(destinationPath + textName.replace(".txt", "F.txt"), 'w');
        textBuffer = f.read()
        for punctSymbol in str.punctuation:
            if punctSymbol in textBuffer:
                textBuffer = textBuffer.replace(punctSymbol, " ")
        for digits in str.digits:
            if digits in textBuffer:
                textBuffer = textBuffer.replace(digits, " ")
        for englishLetter in str.ascii_letters:
            if englishLetter in textBuffer:
                textBuffer = textBuffer.replace(englishLetter, " ")
        tokenizedText = nltk.word_tokenize(textBuffer)
        for word in tokenizedText:
            pos = morph.parse(word)[0]
            if ((pos.tag.POS == 'PREP')
                    or (pos.tag.POS == 'CONJ') or (pos.tag.POS == 'INTJ')
                    or (pos.tag.POS == 'NUMR') or (pos.tag.POS == 'COMP')):
                continue
            if ((pos.tag.POS == 'PRCL') and (word != "не") and (word != "ни")
                    and (word != "Не") and (word != "Ни")):
                continue
            outF.write(pos.normal_form + ' ')
        f.close()
        outF.close()
        print(textName + " is filtered \n")
    return 1
