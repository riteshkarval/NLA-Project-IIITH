#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import nltk
from nltk.corpus import stopwords


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


def removepuntuations(st):
    st = st.replace('\n',' ')
    st = st.replace('.',' ')
    st = st.replace(',',' ')
    punctuations = '''[]{};:\<>/@#$%^&"*~()'''
    st = ''.join([c for c in st if c not in punctuations])
    return st


# In[5]:


files = []
for i in os.listdir('../../Dataset/physicsdocs/'):
        files.append('../../Dataset/physicsdocs/'+i)
for i in os.listdir('../../Dataset/chemistrydocs/'):
        files.append('../../Dataset/chemistrydocs/'+i)
for i in os.listdir('../../Dataset/mathdocs/'):
        files.append('../../Dataset/mathdocs/'+i)
for i in os.listdir('../../Dataset/biologydocs/'):
        files.append('../../Dataset/biologydocs/'+i)
shuffle(files)


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


def get_prediction(sentence):
    test_sentence = tokenize(sentence.lower()) 
#     print(test_sentence)
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=word2idx["PAD"], maxlen=MAX_LEN)

    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    return p[0][:len(test_sentence)], test_sentence


# In[9]:


with open("lstm_crf_model.json", 'r') as content_file:
    json_string = content_file.read()
model = model_from_json(json_string,custom_objects=create_custom_objects())
model.load_weights('lstm_crf_model_weights.h5')

with open(r"word_to_index.pickle", "rb") as input_file:
    word2idx = pickle.load(input_file,)
with open(r"tag_to_index.pickle", "rb") as input_file:
    tag2idx = pickle.load(input_file)


# In[10]:


idx2tag = {tag2idx[key]:key for key in tag2idx.keys()}


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
index = randint(0, len(files)-1)
u_terms = []
bi_terms = []
print(index)
Tokens = []
stop_words = set(stopwords.words('english')) 
print(files[index])
with open(files[index],'r') as tf:
    texts = tf.read()
sentences = nltk.sent_tokenize(texts)
# print(len(sentences))
for each_sentence in sentences:
    tokens = nltk.word_tokenize(removepuntuations(each_sentence))
#     tokens = [tok for tok in tokens if tok not in stop_words]
    preds, _= get_prediction(each_sentence)
    Tokens += tokens
    i = 0
    if len(tokens) > len(preds):
        l = len(preds)
    else:
        l = len(tokens)
    while i <= l-1:
        if idx2tag[preds[i]] != 'O' and idx2tag[preds[i]] != 'PAD':
            if (i < l-2) and (preds[i] == preds[i+1]) and (tokens[i] not in stop_words) and (tokens[i+1] not in stop_words):
                bi_terms.append(tokens[i]+' '+tokens[i+1])
                i = i+2
            else:
                u_terms.append(tokens[i])
                i = i+1
        else:
            i = i+1
if len(bi_terms) > 0:
    bi_terms = list(set(bi_terms))
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(Tokens, window_size = 2)
    biscores = []
    for w in bi_terms:
        w1,w2 = w.split()
        biscores.append(finder.score_ngram(bigram_measures.pmi,w1,w2))
    scores = zip(bi_terms,biscores)
    sorted_scores = sorted(scores, key=lambda x: x[1])#, reverse = True)
#     mean_score = np.average(biscores)
    bi_terms = [w for w,i in sorted_scores if i > 0]

tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=set(u_terms))
tfidf_matrix =  tf.fit_transform([texts.lower()])
mean_score = np.average(np.asarray(tfidf_matrix.sum(axis=0)).ravel())
scores = zip(tf.get_feature_names(),np.asarray(tfidf_matrix.sum(axis=0)).ravel())
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

u_terms = [w for w,i in sorted_scores if i > 0 ]
            
            
# terms = list(set(terms))
# terms = [tok for tok in terms if tok.lower() not in stop_words and len(tok) > 2]
# print(files[index])
# print(sorted(terms, key = len, reverse = True))
print(bi_terms)
print(u_terms)


# In[38]:


# mathdocs/Graph paper


# In[ ]:




