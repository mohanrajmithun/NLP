from rouge import Rouge
import pickle
import pandas as pd
import numpy as np
import os
import glob
import nltk
import re


ref_texts = {'A': "Poor nations pressurise developed countries into granting trade subsidies.",
             'B': "Developed countries should be pressurized. Business exemptions to poor nations.",
             'C': "World's poor decide to urge developed nations for business concessions."}
ref="World's poor decide to urge developed nations for business concessions."
summary_text = "Poor nations demand trade subsidies from developed nations."
scores={}
rouge = Rouge()
scores = rouge.get_scores(summary_text, ref)
# print(scores)
# print(type(scores))
score=np.array(scores)
un=score[0]
for i in un:
    print(un[i]['f'])
# location="hellboy"
# for file in os.listdir(location):
#     l=open(location+"/"+file,"rb")
#     sumdata=pickle.load(l)
#     print(sumdata)
# c=np.zeros(30)
# i=0
# for file in os.listdir("ScrappedData"+"/"+location):
#     index=file.split("_")[0]
#     c[int(index)]=1
# print(c)
#
# objects = []
# objectsnp=[]
# uu=[]
# f=open('DFM2R.pkl',"rb")
# data = pickle.load(f)
#
# data=np.array(data)
# print(data[1,3])
# print(data.shape[0])
#
# for i in range (data.shape[0]):
#     if(c[i]==0):
#         data = np.delete(data, (i), axis=0)
#
# print(data.shape[0])
# scorelist=[]
# location="hellboynew"
# for file in os.listdir(location):
#     print(file,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
#     l=open(location+"/"+file,"rb")
#     sumdata=pickle.load(l)
#     i=0
#     for i  in range(data.shape[0]) :
#         scores=rouge.get_scores(sumdata[i],data[i][3])
#         print(sumdata[i],"===", data[i][3])
#         scorelist.append(scores)
#
#         print(scores)
#
#
# """
# with (open("DFM2R.pkl", "rb")) as openfile:
#     while True:
#         try:
#            # print(objects)
#             objects.append(pickle.load(openfile))
#             uu=pickle.load(openfile)
#         except EOFError:
#             break
# print(objects[:,3])
# print("pickle end")
# objectsnp=np.array(objects)
# objects=np.reshape(objects,(17,4))
#
# """
# print("word 2 vec scores +++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# f=open('word2vechellboy',"rb")
# w2v = pickle.load(f)
# w2vjoined=[]
# print(w2v)
# i=0
# score_list=[]
# for v in w2v:
#
#     v2=" ".join(v)
#     print(v2)
#     w2vjoined.append(str(v2))
#     if(i<data.shape[0]):
#         scores = rouge.get_scores(v2, data[i][3])
#     print((scores))
#     scorelist.append(scores)
#     scores=str(scores)
#     score_str="".join(str(x) for x in scores)
#     print(score_str)
#     list_2 = [num for num in score_str if isinstance(num, (int, float))]
#     i=i+1
#     print(list_2)
# print(score_list)



