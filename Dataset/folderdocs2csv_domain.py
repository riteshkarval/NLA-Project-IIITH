#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
from nltk.corpus import stopwords
from nltk import tokenize
import nltk
import pickle
import pandas as pd
from numpy.random import shuffle
import re


# In[2]:


stop = stopwords.words('english')


# In[3]:


def removepuntuations(text):
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


# In[4]:


with open('terms_rulebased/terms.pkl','rb') as fp:
    terms = pickle.load(fp)


# In[5]:


phy_terms = terms['Physics']
chem_terms = terms['chemistry']
math_terms = terms['Maths']
bio_terms = terms['Biology']


# In[6]:


print(len(phy_terms),len(chem_terms),len(math_terms),len(bio_terms))


# In[7]:


phy_files = []
for i in os.listdir('physicsdocs/'):
        phy_files.append(i)
chem_files = []
for i in os.listdir('chemistrydocs/'):
        chem_files.append(i)
math_files = []
for i in os.listdir('mathdocs/'):
        math_files.append(i)
bio_files = []
for i in os.listdir('biologydocs/'):
        bio_files.append(i)


# In[8]:


phy_sent = []
chem_sent = []
math_sent = []
bio_sent = []
phy_titles = []
chem_titles = []
math_titles = []
bio_titles = []


# In[9]:


for f in phy_files:
    name = 'physicsdocs/'+f
    with open(name,'r') as tf:
        texts = tf.read().lower()
    sentences = nltk.sent_tokenize(texts)
    sentences = [sent for sent in sentences if len(sent.split()) > 5]
    if(len(sentences) > 15):
        phy_sent += sentences
        phy_titles.append(name)
    
for f in chem_files:
    name = 'chemistrydocs/'+f
    with open(name,'r') as tf:
        texts = tf.read().lower()
    sentences = nltk.sent_tokenize(texts)
    sentences = [sent for sent in sentences if len(sent.split()) > 5]
    if(len(sentences) > 15):
        chem_sent += sentences
        chem_titles.append(name)
    
for f in math_files:
    name = 'mathdocs/'+f
    with open(name,'r') as tf:
        texts = tf.read().lower()
    sentences = nltk.sent_tokenize(texts)
    sentences = [sent for sent in sentences if len(sent.split()) > 5]
    if(len(sentences) > 15):
        math_sent += sentences
        math_titles.append(name)

for f in bio_files:
    name = 'biologydocs/'+f
    with open(name,'r') as tf:
        texts = tf.read().lower()
    sentences = nltk.sent_tokenize(texts)
    sentences = [sent for sent in sentences if len(sent.split()) > 5]
    if(len(sentences) > 15):
        bio_sent += sentences
        bio_titles.append(name)


# In[10]:


phy_terms = set(phy_terms)
chem_terms = set(chem_terms)
math_terms = set(math_terms)
bio_terms = set(bio_terms)


# In[11]:


print(len(phy_sent),len(chem_sent),len(math_sent),len(bio_sent))


# In[12]:


print(len(phy_terms),len(chem_terms),len(math_terms),len(bio_terms))


# In[14]:


shuffle(phy_sent)
shuffle(chem_sent)
shuffle(math_sent)
shuffle(bio_sent)


# In[17]:


sentences = []
count = 1
tags = []
for each_sent in phy_sent:
    sent_words = nltk.word_tokenize(removepuntuations(each_sent))
    sent_no = 'sentence: '+str(count)
    for each_word in sent_words:
        sentences.append((sent_no,each_word))
        if each_word in phy_terms:
            tags.append('phy-term')
        else:
            tags.append('O')
    count += 1
    if count%5000 == 0:
        print(count)
print(count)


# In[18]:


index = pd.MultiIndex.from_tuples(sentences)
tags = pd.Series(tags, index=index)

data = tags.reindex(sentences)

data = pd.DataFrame(data)
data.to_csv('wiki_phy_v2.csv')


# In[19]:


sentences = []
count = 1
tags = []
for each_sent in chem_sent:
    sent_words = nltk.word_tokenize(removepuntuations(each_sent))
    sent_no = 'sentence: '+str(count)
    for each_word in sent_words:
        sentences.append((sent_no,each_word))
        if each_word in chem_terms:
            tags.append('chem-term')
        else:
            tags.append('O')
    count += 1
    if count%10000 == 0:
        print(count)
print(count)
index = pd.MultiIndex.from_tuples(sentences)
tags = pd.Series(tags, index=index)

data = tags.reindex(sentences)

data = pd.DataFrame(data)
data.to_csv('wiki_chem_v2.csv')


# In[20]:


sentences = []
count = 1
tags = []
# math_terms = math_terms[:40000]
for each_sent in math_sent:
    sent_words = nltk.word_tokenize(removepuntuations(each_sent))
    sent_no = 'sentence: '+str(count)
    for each_word in sent_words:
        sentences.append((sent_no,each_word))
        if each_word in math_terms:
            tags.append('math-term')
        else:
            tags.append('O')
    count += 1
    if count%10000 == 0:
        print(count)
print(count)
index = pd.MultiIndex.from_tuples(sentences)
tags = pd.Series(tags, index=index)

data = tags.reindex(sentences)

data = pd.DataFrame(data)
data.to_csv('wiki_math_v2.csv')


# In[21]:


sentences = []
count = 1
tags = []
# bio_terms = bio_terms[:40000]
for each_sent in bio_sent:
    sent_words = nltk.word_tokenize(removepuntuations(each_sent))
    sent_no = 'sentence: '+str(count)
    for each_word in sent_words:
        sentences.append((sent_no,each_word))
        if each_word in bio_terms:
            tags.append('bio-term')
        else:
            tags.append('O')
    count += 1
    if count%10000 == 0:
        print(count)
print(count)
index = pd.MultiIndex.from_tuples(sentences)
tags = pd.Series(tags, index=index)

data = tags.reindex(sentences)

data = pd.DataFrame(data)
data.to_csv('wiki_bio_v2.csv')

