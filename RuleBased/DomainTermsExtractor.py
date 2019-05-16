# -*- coding: utf-8 -*-
"""
Created on Tue Apr 5 17:52:29 2019

@author: Ananya Mukherjee

"""


'''
********************************************************************************************************
Domain terms identifier.
********************************************************************************************************
Creating Domain Dictionary:
Input :
    This program scans the various domain documents as input.
Algorithm:
    Scan through various documents and selects the Noun phrases, NN, AN, NAN,ANN,NNN etc.,
    For each domain, forms a 6 dimension list, where each dimension is again a list of N/NN/AN/NAN/NNN/ANN terms
    Now frame the domain terms by subtracting the terms of all other domains.
    PhysicsDomainTerms = PhysicsTerms - BioTerms - MathTerms - ChemistryTerms
Output :
    It outputs a file for each domain which contains the domain terms.

********************************************************************************************************    
    
    '''
import pandas as pd
import numpy as np
import re
import os 
import nltk
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
ignored_words = nltk.corpus.stopwords.words('english')
#******************************************************************************************
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


#Function to identify all the words that are noun in a given sentence
def getN(tagged):
    ls =[]
    for pair in tagged:
        if pair[1] == 'NN' or pair[1] == 'NNP' or pair[1] == 'NNS' or pair[1] == 'NNPS':
            ls.append(pair[0])
    return ls

#Function to identify all the words that are noun-noun in a given sentence
def getNN(tagged):
    ls=[]
    for i in range(0,len(tagged),2):   
        if (i+1)<len(tagged):
            if ((tagged[i][1] == 'NN' or tagged[i][1] == 'NNP' or 
                tagged[i][1] == 'NNS' or tagged[i][1] == 'NNPS')
            and (tagged[i+1][1] == 'NN' or tagged[i+1][1] == 'NNP'
                 or tagged[i+1][1] == 'NNS' or tagged[i+1][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0])
    return ls

#Function to identify all the words that are adjective-noun in a given sentence
def getAN(tagged):
    ls=[]
    for i in range(0,len(tagged),2):   
        if (i+1)<len(tagged):
            if ((tagged[i][1] == 'JJ')              
            and (tagged[i+1][1] == 'NN' or tagged[i+1][1] == 'NNP'
                 or tagged[i+1][1] == 'NNS' or tagged[i+1][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0])
    return ls

#Function to identify all the words that are noun-noun-noun in a given sentence
def getNNN(tagged):
    ls=[]
    for i in range(0,len(tagged),3):   
        if (i+2)<len(tagged):
            if ((tagged[i][1] == 'NN' or tagged[i][1] == 'NNP' or 
                tagged[i][1] == 'NNS' or tagged[i][1] == 'NNPS')             
            and (tagged[i+1][1] == 'NN' or tagged[i+1][1] == 'NNP'
                 or tagged[i+1][1] == 'NNS' or tagged[i+1][1] == 'NNPS')
            and (tagged[i+2][1] == 'NN' or tagged[i+2][1] == 'NNP'
                 or tagged[i+2][1] == 'NNS' or tagged[i+2][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0]+' '+tagged[i+2][0])
    return ls

#Function to identify all the words that are Adjective-noun-noun in a given sentence    
def getANN(tagged):
    ls=[]
    for i in range(0,len(tagged),3):   
        if (i+2)<len(tagged):
            if ((tagged[i][1] == 'JJ')             
            and (tagged[i+1][1] == 'NN' or tagged[i+1][1] == 'NNP'
                 or tagged[i+1][1] == 'NNS' or tagged[i+1][1] == 'NNPS')
            and (tagged[i+2][1] == 'NN' or tagged[i+2][1] == 'NNP'
                 or tagged[i+2][1] == 'NNS' or tagged[i+2][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0]+' '+tagged[i+2][0])
    return ls

#Function to identify all the words that are noun-Adjective-noun in a given sentence
def getNAN(tagged):
    ls=[]
    for i in range(0,len(tagged),3):   
        if (i+2)<len(tagged):
            if ((tagged[i][1] == 'NN' or tagged[i][1] == 'NNP' or 
                tagged[i][1] == 'NNS' or tagged[i][1] == 'NNPS')             
            and (tagged[i+1][1] == 'JJ')
            and (tagged[i+2][1] == 'NN' or tagged[i+2][1] == 'NNP'
                 or tagged[i+2][1] == 'NNS' or tagged[i+2][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0]+' '+tagged[i+2][0])
    return ls

#Function to identify all the words that are noun-Adjective-noun in a given sentence
def getNPN(tagged):
    ls=[]
    for i in range(0,len(tagged),3):   
        if (i+2)<len(tagged):
            if ((tagged[i][1] == 'NN' or tagged[i][1] == 'NNP' or 
                tagged[i][1] == 'NNS' or tagged[i][1] == 'NNPS')             
            and (tagged[i+1][1] == 'IN' or tagged[i+1][1] == 'TO' )
            and (tagged[i+2][1] == 'NN' or tagged[i+2][1] == 'NNP'
                 or tagged[i+2][1] == 'NNS' or tagged[i+2][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0]+' '+tagged[i+2][0])
    return ls


def readFile(fpath):
    ls_N = []  #List containing all noun words
    ls_NN = []  #List containing Noun-Noun words
    ls_AN = [] #List containing Adj-Noun words
    ls_NAN = [] #List containing Noun-Adj-Noun words
    ls_ANN = [] #List containing Adj-Noun-Noun words
    ls_NNN = [] #List containing Noun-Noun-Noun words
    ls_NPN = [] #List containing Noun-Preposition-Noun words
    doc = []
    #Read all the files in the domain folder
    for file in os.listdir(fpath):       
        try:            
            f =  open( os.path.join( fpath, file ) ,"r")
            document = f.read()
            cleanDoc = TextCleaner(document.lower()) #Clean the file            
            tokenized = sent_tokenize(cleanDoc) #Tokenizes the into sentences
            doc.append(cleanDoc)
            for i in tokenized:  
                wordsList = nltk.word_tokenize(i)  
                tagged = nltk.pos_tag(wordsList) 
                #For each file retrieve the word phrases as N, NN, AN, etc.,
                ls_N += getN(tagged) #Noun
                ls_NN += getNN(tagged) #Noun - Noun 
                ls_AN += getAN(tagged) #Adjective Noun
                ls_NAN += getNAN(tagged) #Noun - Adjective- Noune
                ls_ANN += getANN(tagged) #Adjective - Noun - Noun
                ls_NNN += getNNN(tagged) #Noun - Noun - Noun
                ls_NPN += getNPN(tagged) #Noun - Pronoun - Noun
         
        except:
            print('error:', file) 
            f.close()
            os.remove(os.path.join( fpath, file ))
            
    return [ls_N,ls_NN,ls_AN,ls_ANN,ls_NAN,ls_NNN,ls_NPN],doc

def ConvertToSingleString(ls_2D):
    ls_1D =  [s for S in ls_2D for s in S]
    return ls_1D

def getTfIdf(vocab,document):
    vocab = [w.lower() for w in vocab if len(w) > 2]
    document = [d.lower() for d in document]
    vocab = set(vocab)
    
    tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=vocab)
    tfidf_matrix =  tf.fit_transform(document)
    mean_score = np.average(np.asarray(tfidf_matrix.sum(axis=0)).ravel())

    scores = zip(tf.get_feature_names(),np.asarray(tfidf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    terms = [term for term,i in sorted_scores if i >= mean_score]
    return terms

def WordMatchCount(list1,list2):
    #check for the word match 
    matches = [item for item in list1 if item in list2]
    return matches

def generate_collocations(tokens,nphrase,n):  
    scores = []
    if n == 2:
        finderB = BigramCollocationFinder.from_words(tokens,window_size = 2)
        for b in nphrase:
            b1,b2 = b.split()
            scores.append(finderB.score_ngram(bigram_measures.pmi,b1,b2))
    if n == 3:
        finderT = TrigramCollocationFinder.from_words(tokens,window_size = 3)
        for t in nphrase:
            t1,t2,t3 = t.split()
            scores.append(finderT.score_ngram(trigram_measures.pmi,t1,t2,t3))
            
    avg = np.average(np.array(scores))
    bestWords = [nphrase[i] for i in range(len(nphrase)) if scores[i] >=avg]
       
    return bestWords
#**************************************************************************************************#   
#**************************************************************************************************#
#Defining the directory path for each domain folder.
phyPath = '../Dataset/physicsdocs'
mathPath = '../Dataset/mathdocs'
chemPath = '../Dataset/chemistrydocs'
bioPath = '../Dataset/biologydocs'
#**************************************************************************************************#
#Retrieve list of words for each domain, formed by scanning all the wikipedia domain specific docs.
#**************************************************************************************************#
physicWords,PhyDocument = readFile(phyPath)
mathWords,MathDocument = readFile(mathPath)
chemWords,ChemDocument = readFile(chemPath)
bioWords,BioDocument = readFile(bioPath)
#*********************************************************************#
#Tokenize all the domain specific documents
#*********************************************************************#
phytokens = nltk.word_tokenize((' '.join(PhyDocument)).lower())
mathtoken = nltk.word_tokenize((' '.join(MathDocument)).lower())
chemtoken = nltk.word_tokenize((' '.join(ChemDocument)).lower())
biotoken  = nltk.word_tokenize((' '.join(BioDocument)).lower())
#*********************************************************************#
#Process for N phrases
#Retrieve the TF-IDF features 
#*********************************************************************#
PhyTerms_N = getTfIdf(physicWords[0],PhyDocument)
MathTerms_N = getTfIdf(mathWords[0],MathDocument)
ChemTerms_N = getTfIdf(chemWords[0],ChemDocument)
BioTerms_N = getTfIdf(bioWords[0],BioDocument) 
#*********************************************************************#
#process for NN,AN
#Concatenate the AN,NN into a single list
#search for the best terms based on pmi
#*********************************************************************#
phy_Biterms = set(physicWords[1] + physicWords[2])
BestPhy = generate_collocations(phytokens,list(phy_Biterms),2)

math_Biterms = set(mathWords[1] + mathWords[2])
BestMath = generate_collocations(mathtoken,list(math_Biterms),2)

chem_Biterms = set(chemWords[1] + chemWords[2])
BestChem = generate_collocations(chemtoken,list(chem_Biterms),2)

bio_Biterms = set(bioWords[1] + bioWords[2])
BestBio = generate_collocations(biotoken,list(bio_Biterms),2)

#Retrieving the unique words for each domain
uniquePhy_b = set(BestPhy) - set(BestMath) - set(BestChem) - set(BestBio)
uniqueMath_b = set(BestMath) - set(BestPhy) - set(BestChem) - set(BestBio) 
uniqueChem_b= set(BestChem) - set(BestPhy) - set(BestMath) - set(BestBio) 
uniqueBio_b = set(BestBio) - set(BestPhy) - set(BestMath) - set(BestChem)
#************************************************************************#
BestPhy = []
BestMath = []
BestChem = []
BestBio = []
#***********************************************************************#
#Process for NAN,NNN,ANN,NPN
#Concatenate them into a single list
#search for the best terms based on pmi
#***********************************************************************#
phy_Triterms = set(physicWords[3] + physicWords[4] + physicWords[5] + physicWords[6])
BestPhy = generate_collocations(phytokens,list(phy_Triterms),3)

math_Triterms = set(mathWords[3] + mathWords[4] + mathWords[5] + mathWords[6])
BestMath = generate_collocations(mathtoken,list(math_Triterms),3)

chem_Triterms = set(chemWords[3] + chemWords[4] + chemWords[5] + chemWords[6])
BestChem = generate_collocations(chemtoken,list(chem_Triterms),3)

bio_Triterms = set(bioWords[3] + bioWords[4] + bioWords[5] + bioWords[6])
BestBio = generate_collocations(biotoken,list(bio_Triterms),3)

#Retrieving the unique words for each domain 
uniquePhy_t = set(BestPhy) - set(BestMath) - set(BestChem) - set(BestBio)
uniqueMath_t = set(BestMath) - set(BestPhy) - set(BestChem) - set(BestBio) 
uniqueChem_t= set(BestChem) - set(BestPhy) - set(BestMath) - set(BestBio) 
uniqueBio_t = set(BestBio) - set(BestPhy) - set(BestMath) - set(BestChem)

#Writing the domain specific vocab terms in a single file
with open('Terms/phyDicTerms.pkl','wb') as handle:  
    pickle.dump([PhyTerms_N,list(uniquePhy_b),list(uniquePhy_t)], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Terms/mathDicTerms.pkl','wb') as handle:  
    pickle.dump([MathTerms_N,list(uniqueMath_b),list(uniqueMath_t)], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Terms/chemDicTerms.pkl','wb') as handle:  
    pickle.dump([ChemTerms_N,list(uniqueChem_b),list(uniqueChem_t)], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Terms/bioDicTerms.pkl','wb') as handle:  
    pickle.dump([BioTerms_N,list(uniqueBio_b),list(uniqueBio_t)], handle, protocol=pickle.HIGHEST_PROTOCOL)

