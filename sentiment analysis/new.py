import nltk
nltk.download('punkt')
import numpy as np

#input read
with open('amazon_cells_labelled.txt', 'r') as myfile:
  input = myfile.read()
inputs=input.lower()
inputs=inputs.splitlines()

onegram={}
twogram={}
threegram={}
fourgram={}
fivegram={}

sent=[]
for i in inputs:
    sent.append(i.split("\t"))
#print(sent)

sentences , labels=[],[]
for i in sent:
    sentences.append(i[0])
    labels.append(i[1])
#
from nltk.tokenize import RegexpTokenizer
import re
from flair.embeddings import WordEmbeddings
from flair.embeddings import CharacterEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import ELMoEmbeddings

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
glove_embedding = WordEmbeddings('glove')
embedding = BertEmbeddings()
flair_embedding_forward = FlairEmbeddings('news-forward')
#document_embeddings = DocumentPoolEmbeddings([embedding])
document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                              embedding,
                                              flair_embedding_forward])
countone=0
counttwo=0
countthree=0
countfour=0
countfive=0
tokenizer = RegexpTokenizer(r'\w+')

#finding unique n-grams
embeddmov=[]
for i in sentences:
    c =re.sub(r'[^\w\s]','',i)
    c = re.sub(r'[\d]', '', c)
#     if len(c)!=0:
    sentence=(Sentence(c))
    document_embeddings.embed(sentence)
    tens=(sentence.get_embedding())
    embeddmov.append(tens.detach().numpy())

    
from sklearn.metrics import accuracy_score
#
from sklearn.metrics import confusion_matrix
# #sklearncv to comparecv
a=[i for i in range(0,len(sentences))]
shuffle(a)
embeddmov= np.array(embeddmov)
#embeddmov=embeddmov[a]
k=[1]
labels=np.array(labels)
for u in k:
#     imp,matt=inputmatrix(u)
#
#
    train_s=np.array(embeddmov[a,])
    label_s=np.array(labels[a])
#
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    clf = svm.SVC(kernel='linear')
    y_pred = cross_val_predict(clf, train_s, label_s, cv=10)
#    print(y_pred)
#
    tn, fp, fn, tp = confusion_matrix(y_pred,label_s).ravel()
#     # print(tn,fp,fn,tp)
    falsepos=(fp/(fp+tn))
    falseneg=(fn/(fn+tp))
    trueneg=(tn/(tn+fp))
    recall=(tp/(tp+fn))
    precision=(tp/(tp+fp))
    acc=(accuracy_score(label_s, y_pred))
#   print("result for ",matt," matrix")
    print(acc,":accuracy rate\n",falsepos,": false positive rate\n",falseneg,":false negative rate\n",trueneg,":true negative rate\n",precision,":precision\n",recall,":recall")



