from nltk.text import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import re
import pickle as pkl


#helping function 1 for creating binary representation of each vocab.
def binirize(vocab):
    total=len(bin(len(vocab)))-2
    binary_vocab={}
    for v,i in zip(vocab,vocab.values()):
        temp=""
        for j in range(total-len(bin(i))+2 ):
            temp+="0"
        temp+=bin(i)[2:]
        z=np.zeros((1,total))
        for k,j in enumerate(temp):
            if j=="1":
                z[0,k]=1
        binary_vocab[v]=z
    print("inside binarize")
    pkl.dump(binary_vocab,open("E:\\word_embedding_pytorch_2.0\\binary_vocab\\binary_vocab_dict.pkl","wb"))
    return binary_vocab