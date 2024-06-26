from nltk.text import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import re


#helping function 2 for converting words tokenized in the sentence to binary rep.
def words_to_binary_data(window,vocab_binary,tokenized_word):

    x=[]
    y=[]
    for i in tokenized_word:
        temp=[]
        w=window
        middle=int(window/2)
        if len(i)>=window:
            for j in range(len(i)-window+1):
                temp=[]
                temp.append(i[j:w])
                temp=temp[0]
                y.append(temp.pop(middle))
                x.append(temp)
                w+=1
    
    x_b=[]
    y_b=[]
    for i,y in zip(x,y):
        temp=[]
        for j in i:
            temp.append(vocab_binary[j])
        x_b.append(np.array(temp).reshape((1,-1)))
        y_b.append(vocab_binary[y])
    x_b=np.array(x_b)
    y_b=np.array(y_b)
    return x_b,y_b
