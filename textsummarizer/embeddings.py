
#%%
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

from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings


flair_embedding_forward = FlairEmbeddings('news-forward')

embedding = BertEmbeddings()

glove_embedding = WordEmbeddings('glove')
from flair.data import Sentence

summ=[]
tok=[]
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
set_stop_words = set(stopwords.words('english'))
cachedStopWords = stopwords.words("english")

path = 'ScrappedData\Deadpool\*.txt' #note C:
# folder_path = '/some/path/to/file'
embedding = ELMoEmbeddings()
v=0
for index,filename in enumerate(glob.glob(os.path.join(path))):
  print(filename)
  v=v+1
  with open(filename, 'r') as f:

        for files in f:
#            print(files)
            if len(files)!=0:
                
                input = files.lower()
            # print(input)
            # inputs = inputs.splitlines()
                tok.append(input)

                inputs=re.split(r'[.]', input)
            # print(inputs)
            


                sen = []
            
                for i in inputs:
#                print(i)
                    c =re.sub(r'[^\w\s]','',i)
                    c = re.sub(r'[\d]', '', c)
                    #st = ' '.join([word for word in c.split() if word not in cachedStopWords])
                    

#               s=c
                
                
#                sentence = Sentence(inputs[i])
#                embedding.embed(sentence)
 #               for token in sentence:
  #                  sen.append(token.embedding)
                    
                    
#                output = [w for w in c if not w in set_stop_words]
                    emp=' '
                    if c!= emp:
                        sen.append(c)
            summ.append(sen)

#for j in summ:
##print(summ[0][0])
#print(len(summ))


from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import operator
v=0
for j in range(5):
    if j == 0:
        document_embeddings = DocumentPoolEmbeddings([glove_embedding])
        filename='glovedeadp'
        
    if j ==1:
        embedding = BertEmbeddings()
        document_embeddings = DocumentPoolEmbeddings([embedding])
        filename='bertdeadp'
    if j == 2:
        embedding = ELMoEmbeddings()
        document_embeddings = DocumentPoolEmbeddings([embedding])
        filename='elmodeadp'
    if j == 3:
        embedding = CharacterEmbeddings()
        document_embeddings = DocumentPoolEmbeddings([embedding])
        filename='characterdeadp'
    if j==4:
        document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward])
        filename='flairdeadp'

        



    embeddmov=[]
    
    realmov=[]
    for fi in summ:
        
        l=len(sen)
        sentem=[]
        realsen=[]
        
        for k in fi:
        
            if len(k)!=0:
                realsen.append(k)
                
                sentence=(Sentence(k))
                document_embeddings.embed(sentence)
                tens=(sentence.get_embedding())
                sentem.append(tens.detach().numpy())
#        print(len(sentem),v)
        embeddmov.append(sentem)
        realmov.append(realsen)
#    print((realsen[0]),"realmov")
#    print(len(embeddmov),"embedd")
#    print(v,"v")
#    print(len(sentem))
    m=0
    outtop=[]
    
#            print(len(embeddmov))
    for arr in embeddmov:
    
        arr=np.array(arr)
        similarities = cosine_similarity(arr)
        nx_graph = nx.from_numpy_array(similarities)
        scores = nx.pagerank_numpy(nx_graph)


        tops=max(scores.items(), key=operator.itemgetter(1))[0]
#
        x = sorted(scores, key=(lambda key:scores[key]), reverse=True)
#        print("top sentence of review-",m,tops,"-sentence")
#        print(tops)
        outtop.append(realmov[m][tops])
    
#    print(summ[m][tops])
        m=m+1
    print(v,"done")
    v=v+1
#    print(len(outtop))
    with open(filename, 'wb') as fp:
        pickle.dump(outtop, fp)
    
    



with open('berthellboy', 'rb') as handle:
    b = pickle.load(handle)
print(b)








    

