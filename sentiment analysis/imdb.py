import nltk
nltk.download('punkt')
import numpy as np

#input read
with open('imdb_labelled.txt', 'r') as myfile:
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
print(sent)

sentences , labels=[],[]
for i in sent:
    sentences.append(i[0])
    labels.append(i[1])
#
from nltk.tokenize import RegexpTokenizer

countone=0
counttwo=0
countthree=0
countfour=0
countfive=0
tokenizer = RegexpTokenizer(r'\w+')

#finding unique n-grams
for i in sentences:
    sen=tokenizer.tokenize(i)

    for j in sen:
        if j not in onegram:
            onegram[j]=countone
            countone=countone+1

    for k in range(len(sen)-1):
        if sen[k]+sen[k+1] not in twogram:
            twogram[sen[k]+sen[k+1]]= counttwo
            counttwo=counttwo+1

    for l in range(len(sen)-2):
        # print(l,l+1,l+2)
        if sen[l] + sen[l+1] + sen[l+2] not in threegram:
            threegram[sen[l] + sen[l + 1]+ sen[l+2]] = countthree
            countthree = countthree + 1

    for f in range(len(sen)-3):
        # print(l,l+1,l+2)
        if sen[f] + sen[f+1] + sen[f+2] + sen[f+3] not in fourgram:
            fourgram[sen[f] + sen[f + 1]+ sen[f+2]+ sen[f+3]] = countfour
            countfour = countfour + 1

    for f in range(len(sen)-4):
        # print(l,l+1,l+2)
        if sen[f] + sen[f+1] + sen[f+2] + sen[f+3] + sen[f+4] not in fivegram:
            fivegram[sen[f] + sen[f + 1]+ sen[f+2]+ sen[f+3]+sen[f+4]] = countfive
            countfive = countfive + 1


f1=np.zeros((len(sentences),len(onegram)))
f2=np.zeros((len(sentences),len(twogram)))
f3=np.zeros((len(sentences),len(threegram)))
f4=np.zeros((len(sentences),len(fourgram)))
f5=np.zeros((len(sentences),len(fivegram)))


r=0
t=0

#finding features
for sente in sentences:

    sen = tokenizer.tokenize(sente)

    for key, value in onegram.items():
        for i in range(len(sen)):
            if sen[i]==key:
                # t=t+1

                f1[r,value]+=1
    # #
    for key, value in twogram.items():
        for i in range(len(sen)-1):
            if sen[i]+sen[i+1] == key:
                t=t+1

                f2[r, value] += 1
    #
    for key, value in threegram.items():
        for i in range(len(sen)-2):
            if sen[i]+sen[i+1]+sen[i+2] == key:
                t=t+1

                f3[r, value] += 1

    for key, value in fourgram.items():
        for i in range(len(sen)-3):
            if sen[i]+sen[i+1]+sen[i+2]+sen[i+3] == key:
                t=t+1

                f4[r, value] += 1

    for key, value in fivegram.items():
        for i in range(len(sen)-4):
            if sen[i]+sen[i+1]+sen[i+2]+sen[i+3]+sen[i+4] == key:
                t=t+1

                f5[r, value] += 1



    r=r+1

from sklearn.preprocessing import normalize


#
from random import shuffle

labels=np.array(labels)



from sklearn import svm
from sklearn import preprocessing
#normalize
normalized_X1 = preprocessing.normalize(f1)
normalized_X2 = preprocessing.normalize(f2)
normalized_X3 = preprocessing.normalize(f3)
normalized_X4 = preprocessing.normalize(f4)
normalized_X5 = preprocessing.normalize(f5)
#combined
f12 = np.concatenate((normalized_X1, normalized_X2), axis=1)
f123 = np.concatenate((normalized_X1, normalized_X2, normalized_X3), axis=1)
f1234 = np.concatenate((normalized_X1, normalized_X2, normalized_X3, normalized_X4), axis=1)
f12345 = np.concatenate((normalized_X1, normalized_X2, normalized_X3, normalized_X4,normalized_X5), axis=1)

def inputmatrix(i):
    if i == 1:
        mat = "f1"
        return normalized_X1,mat
        pass
    if i == 2:
        mat="f2"
        return normalized_X2,mat
        pass
    if i == 3:
        mat="f3"
        return normalized_X3,mat
        pass

    if i == 4:
        mat="f4"
        return normalized_X4,mat
        pass
    if i == 5:
        mat="f5"
        return normalized_X5,mat
        pass
    if i == 6:
        mat="f1f2"
        return f12,mat
        pass
    if i == 7:
        mat="f1f2f3"
        return f123,mat
        pass
    if i == 8:
        mat="f1f2f3f4"
        return f1234,mat
        pass
    if i == 9:
        mat="f1f2f3f4f5"
        return f12345,mat




from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
#sklearncv to compare
a=[i for i in range(0,len(sentences))]
shuffle(a)
k=[1,2,3,4,5,6,7,8,9]
labels=np.array(labels)
for u in k:
    imp,matt=inputmatrix(u)


    train_s=np.array(imp[a])
    label_s=np.array(labels[a])

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    clf = svm.SVC(kernel='linear')
    y_pred = cross_val_predict(clf, train_s, label_s, cv=10)
    # print(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_pred,label_s).ravel()
    # print(tn,fp,fn,tp)
    falsepos=(fp/(fp+tn))
    falseneg=(fn/(fn+tp))
    trueneg=(tn/(tn+fp))
    recall=(tp/(tp+fn))
    precision=(tp/(tp+fp))
    acc=(accuracy_score(label_s, y_pred))
    print("result for ",matt," matrix")
    print(acc,":accuracy rate\n",falsepos,": false positive rate\n",falseneg,":false negative rate\n",trueneg,":true negative rate\n",precision,":precision\n",recall,":recall")



#crossvalidation
for u in k:
    imp,matt=inputmatrix(u)


    train_s=np.array(imp[a])
    label_s=np.array(labels[a])
    m,n=train_s.shape

    falseneg,falsepos,trueneg,precision,recall=0,0,0,0,0
    acc=0
    for i in range(10):
        test_kfold =train_s[(i*100):(100 * (i + 1)), :]
        # print(test_kfold.shape)
        label_s=np.reshape(label_s,(1000,1))
        test_kfoldlbl =label_s[(i*100):(100*(i + 1)),:]
        train_kfold = np.empty((200,n))
        train_kfoldlbl = np.empty((200,1))


        for j in range(10):
            if(j!=i):

                temp=train_s[(j * 100):(100*(j+1)),:]

                templbl=label_s[(j * 100):(100*(j+1)),:]


                train_kfold = np.append(train_kfold,temp, 0)

                train_kfoldlbl=np.append(train_kfoldlbl,templbl, 0)
        train_kfold = train_kfold[100:1000,:]
        train_kfoldlbl = train_kfoldlbl[100:1000, :]
        # print("fold:",i,train_kfold.shape,train_kfoldlbl.shape,test_kfold.shape,test_kfoldlbl.shape)
        clf = svm.SVC(kernel='linear')
        clf.fit(train_kfold, train_kfoldlbl)
        y_pred = clf.predict(test_kfold)
        tn, fp, fn, tp = confusion_matrix(y_pred,test_kfoldlbl).ravel()
        print(tn,fp,fn,tp)
        falsepos=falsepos+(fp/(fp+tp))
        falseneg=falseneg+(fn/(fn+tn))
        trueneg=trueneg+(tn/(tn+fp))
        recall=recall+(tp/(tp+fn))
        precision=precision+(tp/(tp+fp))
        acc=acc+(accuracy_score(test_kfoldlbl, y_pred))

        print("fold:",i)
        # print(acc)

    print("result for ", matt, " matrix")
    print(acc/10,": accuracy \n",falseneg/10,": false negative rate\n",falsepos/10,": false positive rate\n",trueneg/10,": true neg rate\n",precision/10,": precision\n",recall/10,": recall")
