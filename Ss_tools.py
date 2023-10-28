# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:59:02 2019

@author: Ss
"""

from nltk.tokenize import RegexpTokenizer
import codecs
from gensim import corpora
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')


#loading 
def loading_corpus(c_file):
    doc_set=[]
    corpus_lines = codecs.open(c_file,'r','utf-8')
    for line in corpus_lines:
        line = line.strip()
        line =str(line.split(' ')[1:])
        doc_set.append(line)
        
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        
        # clean and tokenize document string
        raw = i.lower()
        texts.append(tokenizer.tokenize(raw))
     
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts] #doc2bow corpus[d_id]=[(wid,feq)]
    corpus_without_feq=[]
    for doc in corpus:
        temp = [w[0] for w in doc]
        corpus_without_feq.append(temp)
#    print(corpus)
#    print(corpus_without_feq)
    return dictionary,corpus,corpus_without_feq


#def load_embedding(file_path):
#    """ Load word embeddings from file.
#        :return: a dict {word:embedding}
#    """
#    embed = {}
#    data = pd.read_csv(file_path, sep=' ', header=None, index_col=0)
#    for i in range(len(data.index)):
#        embed[str(data.index[i])] = np.asarray(data.values[i], dtype='float32')
#        
#    return embed
def load_embedding(file_path):
    """ Load word embeddings from file.
        :return: a dict {word:embedding}
    """
    embed = {}
    data = np.loadtxt(file_path,dtype = str,delimiter = ' ')
    for i in range(len(data)):
        embed[str(data[i][0])] = np.asarray(data[i][1:], dtype='float32')
    return embed

