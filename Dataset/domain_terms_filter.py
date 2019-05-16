#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import os 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


def TextCleaner(text):   
    text=re.sub(r'(\d+)',r'',text)    
    text=text.replace(u'%','')   
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u'“','')
    text=text.replace(u'”','')      
    text=text.replace(u':','')
    text=text.replace(u"'",'')
    text=text.replace(u"‘‘",'')
    text=text.replace(u"’’",'')
    text=text.replace(u"''",'')
    text=text.replace(u".",' ')
    text=text.replace(u"_",' ')
    text=text.replace(u"-",' ')
    text=text.replace(u"=",'')
    text=text.replace(u"+",'')
    text=text.replace(u"]",'')
    text=text.replace(u"[",'')
    text=text.replace(u"{",'')
    text=text.replace(u"}",'')
    text=text.replace(u"<br/>",'')
    text=text.replace(u"\n",'')
    text=text.replace(u"\\",'')
    text=text.replace(u"/",'')
    return text


# In[3]:


def getN(tagged):
    ls =[]
    for pair in tagged:
        if pair[1] == 'NN' or pair[1] == 'NNP' or pair[1] == 'NNS' or pair[1] == 'NNPS' or pair[1] == 'JJ':
            ls.append(pair[0])
    return ls


# In[4]:


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


# In[5]:


def getAN(tagged):
    ls=[]
    for i in range(0,len(tagged),2):   
        if (i+1)<len(tagged):
            if ((tagged[i][1] == 'JJ')              
            and (tagged[i+1][1] == 'NN' or tagged[i+1][1] == 'NNP'
                 or tagged[i+1][1] == 'NNS' or tagged[i+1][1] == 'NNPS')):
                ls.append(tagged[i][0]+' '+tagged[i+1][0])
    return ls


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


def readFile(fpath):
    ls_N = []  
    ls_NN = []  
    ls_AN = [] 
    ls_NAN = [] 
    ls_ANN = [] 
    ls_NNN = [] 
    ls_NPN = [] 
    doc = []
    for file in os.listdir(fpath):       
        try:            
            f =  open( os.path.join( fpath, file ) ,"r")
            document = f.read()
            cleanDoc = TextCleaner(document)          
            tokenized = sent_tokenize(cleanDoc) #Tokenizes the into sentences
            doc.append(cleanDoc)
            for i in tokenized:  
                wordsList = nltk.word_tokenize(i)  
                tagged = nltk.pos_tag(wordsList) 
                ls_N += getN(tagged) #Noun
                ls_NN += getNN(tagged) #Noun - Noun 
                ls_AN += getAN(tagged) #Adjective Noun
                ls_NAN += getNAN(tagged)
                ls_ANN += getANN(tagged)
                ls_NNN += getNNN(tagged)
                ls_NPN += getNPN(tagged)
        except:
            print('error:', file) 
            
    return [ls_N,ls_NN,ls_AN,ls_ANN,ls_NAN,ls_NNN,ls_NPN],doc


# In[11]:


def ConvertToSingleString(ls_2D):
    ls_1D =  [s for S in ls_2D for s in S]
    return ls_1D

def getTfIdfScore(Document):
    vectors = TfidfVectorizer(ngram_range = (1,1))
    DomainVectors = vectors.fit_transform(Document)
    return (vectors.get_feature_names())
    #return DomainVectors
    
def WordMatchCount(list1,list2):
    #check for the word match 
    matches = [item for item in list1 if item in list2]
    return matches


# In[12]:


phyPath = 'physicsdocs'
mathPath = 'mathdocs'
chemPath = 'chemistrydocs'
bioPath = 'biologydocs'
uniquePhy = []
uniqueMath = []
uniqueChem = []
uniqueBio = []


# In[13]:


physicWords,PhyDocument = readFile(phyPath)
mathWords,MathDocument = readFile(mathPath)
chemWords,ChemDocument = readFile(chemPath)
bioWords,BioDocument = readFile(bioPath)


# In[14]:


import pickle
with open('terms_rulebased/physics_docs.pkl', 'wb') as handle:
    pickle.dump(PhyDocument, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/math_docs.pkl', 'wb') as handle:
    pickle.dump(MathDocument, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/chemistry_docs.pkl', 'wb') as handle:
    pickle.dump(ChemDocument, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/bio_docs.pkl', 'wb') as handle:
    pickle.dump(BioDocument, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[15]:



with open('terms_rulebased/physics_terms.pkl', 'wb') as handle:
    pickle.dump(physicWords, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/math_terms.pkl', 'wb') as handle:
    pickle.dump(mathWords, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/chemistry_terms.pkl', 'wb') as handle:
    pickle.dump(chemWords, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('terms_rulebased/bio_terms.pkl', 'wb') as handle:
    pickle.dump(bioWords, handle, protocol=pickle.HIGHEST_PROTOCOL)

