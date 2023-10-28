import re
import numpy as np

def read_docs(file_name):
    print('read documents')
    print('-'*50)
    docs = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1]) 
        arr = filter(None, arr) 
        arr = [int(idx) for idx in arr] 
        docs.append(arr)
    fp.close()
    
    return docs

def read_vocab(file_name):
    print('read vocabulary')
    print('-'*50)
    vocab = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        vocab.append(arr[0]) 
    fp.close()

    return vocab

