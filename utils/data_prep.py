from nltk.text import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import re
from utils.assign_binary_data_to_words import words_to_binary_data
import pickle as pkl
# from utils.binarize_data import binirize
from utils.one_hot_vocab import one_hot_encode

# it for considering the lable as binary but it is not used ....
# def data_preparing_to_binary_one_hot_encoding(data_file,window_size):
    

#     #### data_file : path of the file that needs to be word embedded
#     #### window_size :size fro preparing the dataset 
#     with open(data_file,'r',encoding="utf8") as f:
#         file=f.read()

#     pattern=r"[\"\'\.,\!@#$%^&*()_\-\{\}:;]"
#     data=sent_tokenize(file)
#     word_tokenized=[word_tokenize(re.sub(pattern,"",i)) for i in data]


#     vocab={}
#     count=0
#     for i in word_tokenized:
#         for j in i:
#             try:
#                 if j not in vocab.keys():
#                     vocab[j]=count
#                     count+=1
#             except:
#                 pass

#     binary_vocab=binirize(vocab)
#     x,y=words_to_binary_data(window_size,binary_vocab,word_tokenized)
#     return x,y

def data_preparing_to_encoding(data_file,window_size):
    

    #### data_file : path of the file that needs to be word embedded
    #### window_size :size fro preparing the dataset 
    with open(data_file,'r',encoding="utf8") as f:
        file=f.read()

    pattern=r"[\"\'\.,\!@#$%^&*()_\-\{\}:;]"
    data=sent_tokenize(file)
    word_tokenized=[word_tokenize(re.sub(pattern,"",i)) for i in data]


    vocab={}
    count=0
    for i in word_tokenized:
        for j in i:
            try:
                if j not in vocab.keys():
                    vocab[j]=count
                    count+=1
            except:
                pass
    pkl.dump(vocab,open("E:\\word_embedding_pytorch_2.0\\vocab_dict\\vocab_dict.pkl","wb"))
    binary_vocab=one_hot_encode(vocab)
    x,y=words_to_binary_data(window_size,binary_vocab,word_tokenized)
    return x,y