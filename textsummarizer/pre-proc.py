import pickle
import pandas as pd
import numpy as np
import os
import glob
import nltk
import re

nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import re

from nltk.corpus import stopwords







# data = pickle.load(f)
# data=np.array(data)
# print(data[:,3])

summ=[]
bes=[]
we=[]
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
set_stop_words = set(stopwords.words('english'))
path = 'ScrappedData\Deadpool\*.txt' #note C:
# folder_path = '/some/path/to/file'
for filename in glob.glob(os.path.join(path)):
  with open(filename, 'r') as f:

        for files in f:
            input = files.lower()
            # inputs = inputs.splitlines()
            # tok.append(input)

            inputs=re.split(r'[.]', input)


            sen = []
            for i in inputs:
                c=re.sub(r'[^\w\s]','',i)
                c = re.sub(r'[\d]', '', c)


                c=word_tokenize(c)
                # wtst.append(c)
                output = [w for w in c if not w in set_stop_words]
                sen.append(output)




            summ.append(sen)


            # text = f.read()
            # summ.append(text)
print(len((summ[0])))


# print(filtered_words)
from nltk.tokenize import word_tokenize
# token=[]
# for j in tok:
#     # print(j.split("/t"))
#     k = re.sub(r'[^\w\s]', '', j)
#     k = re.sub(r'[\d]', '', k)
#     token.append((word_tokenize(k)))
# # print(len(tok[0]))
# # print((token[0]))

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import operator
from gensim.test.utils import common_texts

# token=list(token)
# tokens=list(token)
r=0
bestsent=[]
realmov=[]
for ind in summ:
    # print(ind)
    nsen=len(ind)
    model= Word2Vec(ind,workers=4,min_count=1)
    model.train(ind,epochs=10,total_examples=nsen)


    voc=model.wv.vocab


# embeddings=[]
# for w in token[0]:
    # print(w)
#     embeddings.append(model.wv[w])
#
# print(embeddings)
    sentembed=[]
    vec=np.zeros((100,))
    realsen=[]
    for j in ind:
        # print(j)
        if (len(j) != 0):
            realsen.append(j)
            c=0
            for k,i in enumerate(j):
                # print(i)
                c+=1

                for h in voc:
                    if j[k]== h :
                        vec=vec + model.wv[i]
                vec=vec/c
            sentembed.append(vec)
    realmov.append(realsen)
    # print(len(realmov[0]),"mov")
    # print(len(realsen),"sen")
    # break

# print(len(sentembed),len(summ[0]))
    sentembed=np.array(sentembed)
# # print(sentembed)
#
# from numpy import dot
# from numpy.linalg import norm
#
# # for i in sentembed:
# # cos_sim = dot(vect1,vect2)/(norm(vect1)*norm(vect2))
# # print(cos_sim)
# from sklearn.metrics.pairwise import cosine_similarity
#     similarities = cosine_similarity(sentembed)
#     print(len(similarities))
# # print(np.amin(similarities))
    sim_mat = np.zeros([len(sentembed), len(sentembed)])
    for i in range(len(sentembed)):
        for j in range(len(sentembed)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentembed[i].reshape(1,100), sentembed[j].reshape(1,100))[0,0]
    # print(len(sim_mat))
# import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank_numpy(nx_graph)

#
    tops=max(scores.items(), key=operator.itemgetter(1))[0]
# # #
#     x = sorted(scores, key=(lambda key:scores[key]), reverse=True)
#     print(ind[tops])
    bestsent.append(realsen[tops])
    # print(bestsent)
    r=r+1
    print(r)


with open('word2vecgodfboy', 'wb') as fp:
    pickle.dump(bestsent, fp)

with open('word2vecgodfboy', 'rb') as handle:
    b = pickle.load(handle)
print(len(b))










# import torch
# import flair
# from flair.data import Sentence
# # sentence = Sentence('Blogs of Analytics Vidhya are Awesome.')
# # print(sentence)
# from flair.embeddings import WordEmbeddings
# from flair.embeddings import CharacterEmbeddings
# from flair.embeddings import StackedEmbeddings
# from flair.embeddings import FlairEmbeddings
# from flair.embeddings import BertEmbeddings
# from flair.embeddings import ELMoEmbeddings
# from flair.embeddings import FlairEmbeddings
#
# glove_embedding = WordEmbeddings('glove')
# character_embeddings = CharacterEmbeddings()
# flair_forward  = FlairEmbeddings('news-forward-fast')
# flair_backward = FlairEmbeddings('news-backward-fast')
# bert_embedding = BertEmbeddings()
# elmo_embedding = ELMoEmbeddings()
#
# stacked_embeddings = StackedEmbeddings( embeddings = [
#                                                        flair_forward,
#                                                        flair_backward
#                                                       ])
# # stacked_embeddings(summ[0][0])
# # for token in sum[0][0]:
# #   print(token.embedding)
#
#
#
# from flair.data import Sentence
# sentence = Sentence('Blogs of Analytics Vidhya are Awesome.')
# # emb=BertEmbeddings(sentence)
# # print(emb)
# stacked_embeddings(sentence)
# for token in sentence:
#   print(token.embedding)
#











