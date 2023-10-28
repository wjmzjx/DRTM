# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:23:17 2019

@author: Ss
"""

import Ss_tools as tools
import numpy as np


def cal_co(c_file, h_pmi, flag=False):
    dictionary, corpus, _ = tools.loading_corpus(c_file)
    # PMI
    V = len(dictionary)
    dt_mat = np.ones((V, V))  # small value for log func
    for doc in corpus:
        for wfi in doc:
            for wfj in doc:
                dt_mat[int(wfi[0]), int(wfj[0])] += 1
    D1 = np.sum(dt_mat)
    SS = D1 * dt_mat  # V*V
    for k in range(V):
        SS[k] /= np.sum(dt_mat[k])  
    for k in range(V):
        SS[:, k] /= np.sum(dt_mat[:, k])  
    SS = np.log(SS)
    SS[SS < h_pmi] = 0
    del dt_mat

    print('PMI done cal CO...')
    # context overlap
    CO = np.zeros((V, V))
    for i in range(V):
        temp_M = np.zeros((V, V))
        temp_M[0:V] = SS[i]
        a_b = temp_M - SS
        a_b[a_b < 0] = 0
        CO[i] = np.sum(temp_M - a_b, axis=1)
    return CO


def cal_pmi(c_file, h_pmi=0):
    dictionary, corpus, _ = tools.loading_corpus(c_file)
    # PMI
    V = len(dictionary)
    dt_mat = np.ones((V, V))  # small value for log func
    for doc in corpus:
        for wfi in doc:
            for wfj in doc:
                dt_mat[int(wfi[0]), int(wfj[0])] += 1
    D1 = np.sum(dt_mat)
    SS = D1 * dt_mat  # V*V
    for k in range(V):
        SS[k] /= np.sum(dt_mat[k])  
    for k in range(V):
        SS[:, k] /= np.sum(dt_mat[:, k])  
    SS = np.log(SS)
    SS[SS < h_pmi] = 0
    del dt_mat
    print('PMI done')
    return SS


def cal_extra_info(c_file, e_file):
    '''
    return inner product and word-doc
    '''

    dictionary, corpus, _ = tools.loading_corpus(c_file)
    embedding = tools.load_embedding(e_file)
    d = len(embedding['time'])
    # cal inner product
    e_mat = []
    for i in dictionary.keys():
        if dictionary.get(i) not in embedding:
            embedding[dictionary.get(i)] = np.ones(d, dtype='float32')
        e_mat.append(embedding[dictionary.get(i)])
    e_mat = np.array(e_mat)
    extra_info = np.dot(e_mat, e_mat.T)
    # word-doc 
    wd_m = np.zeros((len(dictionary), len(corpus)))
    for i, d in enumerate(corpus):
        for w in d:
            wd_m[w[0], [i]] = w[1]
    extra_info[extra_info<0]=0
    return extra_info, wd_m, dictionary


def cal_d_doc(c_file, e_file):
    dictionary, corpus, _ = tools.loading_corpus(c_file)  ##doc2bow corpus[d_id]=[(wid,feq)]
    embedding = tools.load_embedding(e_file)
    dem = len(embedding['time'])

    docs = []
    for d in corpus:
        temp = []
        for w in d:
            if dictionary.get(w[0]) not in embedding:
                embedding[dictionary.get(w[0])] = np.ones(dem, dtype='float32')
            for _ in range(w[1]):
                temp.append(embedding.get(dictionary.get(w[0])))
        doc = np.mean(temp, axis=0)
        docs.append(doc)
    docs = np.array(docs)
    docs[docs<0]=0
    return docs.T