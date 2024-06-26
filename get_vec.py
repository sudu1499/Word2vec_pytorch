import torch
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import pickle as pkl
import numpy as np

model_path="E:\\word_embedding_pytorch_2.0\\model.pkl"

vocab_path="E:\\word_embedding_pytorch_2.0\\vocab_dict\\vocab_dict.pkl"

model=torch.load(model_path)
vocab=pkl.load(open(vocab_path,"rb"))


w=model.l2.weight.detach().cpu().numpy()

man=w[vocab['man']].reshape(1,-1)
king=w[vocab['king']].reshape(1,-1)
queen=w[vocab['queen']].reshape(1,-1)
woman=w[vocab['woman']].reshape(1,-1)

r=king-man+woman

cosine_similarity(r,queen)
cosine_similarity(man,king)
cosine_similarity(man,woman)



