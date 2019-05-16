#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import model_from_json
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import re
import string
import os
import nltk
from random import shuffle, randint
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


def removepuntuations(text):
    text=re.sub(r'(\d+)',r'',text)    
    text=text.replace(u'%','')   
    text=text.replace(u',','')
    text=text.replace(u'"','')
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'"','')
    text=text.replace(u'“','')
    text=text.replace(u'^','')
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


if tf.test.is_gpu_available():
    BATCH_SIZE = 512  # Number of examples used in each iteration
    EPOCHS = 10  # Number of passes through entire dataset
    MAX_LEN = 75  # Max length of review (in words)
    EMBEDDING = 80  # Dimension of word embedding vector

else:
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_LEN = 75
    EMBEDDING = 20


# In[4]:


labels = {'chem-term':'Chemistry',
         'math-term': 'Mathematics',
         'phy-term':'Physics',
         'bio-term':'Biology'
          }


# In[5]:


def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 


# In[6]:


def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}


# In[7]:


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[8]:


tokenize('I am here now') 


# In[9]:


def get_prediction(sentence, model,word2idx):
    test_sentence = tokenize(sentence.lower()) 
#     print(test_sentence)
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=word2idx["PAD"], maxlen=MAX_LEN)

    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    return p[0][:len(test_sentence)], test_sentence


# In[10]:


with open("phy_model/phy_crf_model.json", 'r') as content_file:
    json_string = content_file.read()
phy_model = model_from_json(json_string,custom_objects=create_custom_objects())
phy_model.load_weights('phy_model/phy_crf_model_weights.h5')

with open(r"phy_model/word_to_index_phy.pickle", "rb") as input_file:
    phy_word2idx = pickle.load(input_file,)
with open(r"phy_model/tag_to_index_phy.pickle", "rb") as input_file:
    phy_tag2idx = pickle.load(input_file)


# In[11]:


with open("chem_model/chem_crf_model.json", 'r') as content_file:
    json_string = content_file.read()
chem_model = model_from_json(json_string,custom_objects=create_custom_objects())
chem_model.load_weights('chem_model/chem_crf_model_weights.h5')

with open(r"chem_model/word_to_index_chem.pickle", "rb") as input_file:
    chem_word2idx = pickle.load(input_file,)
with open(r"chem_model/tag_to_index_chem.pickle", "rb") as input_file:
    chem_tag2idx = pickle.load(input_file)


# In[12]:


with open("math_model/math_crf_model.json", 'r') as content_file:
    json_string = content_file.read()
math_model = model_from_json(json_string,custom_objects=create_custom_objects())
math_model.load_weights('math_model/math_crf_model_weights.h5')

with open(r"math_model/word_to_index_math.pickle", "rb") as input_file:
    math_word2idx = pickle.load(input_file,)
with open(r"math_model/tag_to_index_math.pickle", "rb") as input_file:
    math_tag2idx = pickle.load(input_file)


# In[14]:


phy_idx2tag = {phy_tag2idx[key]:key for key in phy_tag2idx.keys()}
chem_idx2tag = {chem_tag2idx[key]:key for key in chem_tag2idx.keys()}
math_idx2tag = {math_tag2idx[key]:key for key in math_tag2idx.keys()}


# In[15]:


print(phy_idx2tag)
print(chem_idx2tag)
print(math_idx2tag)


# In[ ]:


files = []
for i in os.listdir('../../Dataset/physicsdocs/'):
        files.append('../../Dataset/physicsdocs/'+i)
for i in os.listdir('../../Dataset/chemistrydocs/'):
        files.append('../../Dataset/chemistrydocs/'+i)
for i in os.listdir('../../Dataset/mathdocs/'):
        files.append('../../Dataset/mathdocs/'+i)
shuffle(files)


# In[104]:


index = randint(0, len(files)-1)
terms = []
counts = {
        'Physics':0,
        'Chemistry':0,
        'Math':0
}
tfidf = {}
# print(index)
print(files[index], index)
stop = set(stopwords.words('english')) 
with open(files[index],'r') as tf:
    texts = tf.read()
sentences = nltk.sent_tokenize(texts)
print('No. of Sentences:',len(sentences))
Tokens = tokenize(removepuntuations(texts.lower()))
phy_terms = set()
chem_terms = set()
math_terms = set()
bio_terms = set()
for each_sentence in sentences:
    phy_preds, phy_words = get_prediction(each_sentence,phy_model,phy_word2idx)
    chem_preds, chem_words = get_prediction(each_sentence,chem_model,chem_word2idx)
    math_preds, math_words = get_prediction(each_sentence,math_model,math_word2idx)
#     bio_preds, bio_words = get_prediction(each_sentence,bio_model,bio_word2idx)
    if len(phy_words) > len(phy_preds):
        l = len(phy_preds)
    else:
        l = len(phy_words)
    for i in range(l):
        if phy_idx2tag[phy_preds[i]] == 'phy-term':
            phy_terms.add(phy_words[i])
            
    if len(chem_words) > len(chem_preds):
        l = len(chem_preds)
    else:
        l = len(chem_words)
    for i in range(l):
        if chem_idx2tag[chem_preds[i]] == 'chem-term':
            chem_terms.add(chem_words[i])
            
        
    if len(math_words) > len(math_preds):
        l = len(math_preds)
    else:
        l = len(math_words)
    for i in range(l):
        if math_idx2tag[math_preds[i]] == 'math-term':
            math_terms.add(math_words[i])
if len(phy_terms) > 0:
    tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=phy_terms)
    tfidf_matrix =  tf.fit_transform(Tokens)
    tfidf['Physics'] = np.average(np.asarray(tfidf_matrix.sum(axis=0)).ravel())
else:
    tfidf['Physics'] = 0

if len(chem_terms) > 0:
    tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=chem_terms)
    tfidf_matrix =  tf.fit_transform(Tokens)
    tfidf['Chemistry'] = np.average(np.asarray(tfidf_matrix.sum(axis=0)).ravel())
else:
    tfidf['Chemistry'] = 0
    
if len(math_terms) > 0:
    tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=math_terms)
    tfidf_matrix =  tf.fit_transform(Tokens)
    tfidf['Math'] = np.average(np.asarray(tfidf_matrix.sum(axis=0)).ravel())
else:
    tfidf['Math'] = 0

print(tfidf)
print(max(tfidf, key=tfidf.get))

