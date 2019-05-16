# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:19:35 2019

@author: Ananya Mukherjee
"""
'''
The collected domain terms document from previous acts as reference.
For a given test document, compare the list of terms obtained from the previous program, 
with all the unique domain terms.
Based upon the maximum count of the matched words, the final domain label can be retrieved.
'''
import pandas as pd
import numpy as np
import re
import os 
import nltk
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
#*************************************************************************************************#
#Function to remove punctuation
def TextCleaner(text):   
    text=re.sub(r'(\d+)',r'',text)    
    text=text.replace(u'%',' ')   
    text=text.replace(u',',' ')
    text=text.replace(u'"',' ')
    text=text.replace(u'(',' ')
    text=text.replace(u')',' ')
    text=text.replace(u'"',' ')
    text=text.replace(u'“',' ')
    text=text.replace(u'”',' ')      
    text=text.replace(u':',' ')
    text=text.replace(u"'",' ')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",' ')
    text=text.replace(u"-",' ')
    text=text.replace(u"=",' ')
    text=text.replace(u"+",' ')
    text=text.replace(u"]",' ')
    text=text.replace(u"[",' ')
    text=text.replace(u"{",' ')
    text=text.replace(u"}",' ')
    text=text.replace(u"<br/>",'')
    text=text.replace(u"\n",'')
    text=text.replace(u"\\",' ')
    text=text.replace(u"/",' ')
    return text

def readTermFile(fpath,fname):
 
    f =  open( os.path.join( fpath, fname ) ,"r")
    vocab = TextCleaner(f.read())
    tokens = nltk.word_tokenize(vocab)
    unigram = list(set(tokens))
    bigram  = list(nltk.bigrams(tokens))
    
    for i in range(len(bigram)):
        bigram[i] = ' '.join(bigram[i])
        
    trigram = list(nltk.trigrams(tokens))
    
    for i in range(len(trigram)):
        trigram[i] = ' '.join(trigram[i])
    
    return [unigram, bigram, trigram]
   
    

def WordMatchCount(TestVocab,DomainWordsList):
    matches = []
    for i in range(len(TestVocab)):
        #check for the word match 
        matches += [item for item in TestVocab[i] if item in DomainWordsList[i]]
    return matches
#**************************************************************************************************#
#Dictionary for domain terms.
#**************************************************************************************************#
DicPath = 'Terms/'  
testDocPath = '../Dataset/TestDocuments/'
domainDiclist = ['bio','chem','math','phy']
result = [] #List to hold final result containing file name, domain label, contributing terms
VocabList = {} 
#**************************************************************************************************#
#Create the words list (Vocab Dictionary) for each domain.
for domain in domainDiclist:
        fname = domain + 'DicTerms.pkl'
        VocabList[domain] = pickle.load(open( os.path.join( DicPath, fname ) ,"rb"))
             
#Process each test file document (which has only the word phrases as N/AN/NN/ANN/NAN/NNN)
for testfolder in os.listdir(testDocPath):
    path = testDocPath+testfolder+'/'
    for testfile in os.listdir(path):
        try:
            testWordsList = readTermFile(path,testfile) #Retrieve the words list for test document    
            
            maxMatchCount = 0 
            finalDomain = ''
            ContributingTerms = ''
            #Compare test document with vocab list of all the domains.
            for domain in domainDiclist:
                MatchWords = WordMatchCount(testWordsList,VocabList[domain])
                
                if len(MatchWords) > maxMatchCount:
                    maxMatchCount = len(MatchWords)
                    finalDomain = domain 
                    ContributingTerms = MatchWords
             
            result.append([testfile,testfolder,finalDomain,ContributingTerms])
        except:
            print(testfile)

df = pd.DataFrame(result,columns = ['File Name','Actual','Prediction','Contributing Terms'])
trueLabels = list(df['Actual'])
predictedLabels = list(df['Prediction'])
#*****************Classifier Measures*******************************************
print('****************** Confusion Matrix ********************** \nbio chm mat phy')
CM = confusion_matrix(trueLabels, predictedLabels, labels=['bio','chem','math','phy'])
print(CM)
print('*****************Classification Report *******************')
print(classification_report(trueLabels, predictedLabels)) 
CMarray = np.array(CM)
print('Overall Accuracy:', accuracy_score(trueLabels, predictedLabels)) 
print('***********Classwise Accuracy**********')
for row in range(CMarray.shape[0]):
    den = np.sum(CMarray[row])
    num = CMarray[row][row]
    accuracy = num/den
    print(domainDiclist[row],' Accuracy:',accuracy)
        
#*******************************************************************************