#!/usr/bin/env python
# coding: utf-8

# パッケージ

# jupyter nbconvert --to python tdss.ipynb
# で.pyに変換

# In[132]:


import pandas as pd
import numpy as np
import random as rnd
import math


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[240]:


train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")
combine=[train_df,test_df]


# In[14]:


print(train_df.columns.values)
train_df.describe()


# In[17]:


train_df.head()


# In[18]:


train_df.tail(n=1)


# In[19]:


train_df.info()


# In[38]:


train_df.describe(include=["O"])


# In[47]:


train_df[["Pclass","Survived"]].groupby(["Pclass"]).mean()


# In[50]:


train_df[["Sex","Survived"]].groupby(["Sex"]).mean()
train_df[["Sex","Survived","Age"]].groupby(["Sex"]).mean()


# In[51]:


train_df[["SibSp","Survived"]].groupby(["SibSp"]).mean()


# In[52]:


train_df[["Parch","Survived"]].groupby(["Parch"]).mean()


# In[54]:


g=sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Age",bins=20)


# In[63]:


g=sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Pclass",bins=20)


# In[62]:


grid=sns.FacetGrid(train_df,col="Survived",row="Pclass")
grid.map(plt.hist,"Age",bins=20)


# In[68]:


grid=sns.FacetGrid(train_df,row="Embarked")
grid.map(sns.pointplot,"Pclass","Survived","Sex")
grid.add_legend()


# In[69]:


grid=sns.FacetGrid(train_df,row="Embarked",col="Survived")
grid.map(sns.barplot,"Sex","Fare")


# In[241]:


train_df=train_df.drop(["Ticket","Cabin"],axis=1)
test_df=test_df.drop(["Ticket","Cabin"],axis=1)
combine=[train_df,test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# In[111]:


train_df["Name"].describe()
train_df.Name


# In[242]:


for dataset in combine:
    dataset["Title"]=dataset.Name.str.extract("([A-Za-z]+)\.",expand=False)

#dataset["Title"]
#print(train_df)
pd.crosstab(combine[0]["Title"],combine[0]["Sex"])
train_df["Title"].describe()


# In[247]:


print(combine[0]["Title"].unique())
print(type(combine[0]["Title"].unique()))
print((combine[1]["Title"].unique()))
raretitle=np.delete(combine[0]["Title"].unique(),[0,1,2,3],0)
raretitle=np.insert(raretitle,0,"Dona")
print(raretitle)



# In[248]:


for dataset in combine:
    dataset["Title"]=dataset["Title"].replace(raretitle,"Rare")
combine[0][["Title","Survived"]].groupby(["Title"]).mean()


# In[116]:


print(combine[0].isnull().any())


# In[249]:


#print(combine[0][combine[0]["Age"].isnull()])
#print(combine[0][combine[0]["Age"].isnull()].loc[:,"Agena"])
for dataset in combine:
    dataset["Agena"]=0
    for i in dataset.index:
        #print(i)
        #if dataset.at[i,"Age"].isnull():
        if math.isnan(dataset.at[i,"Age"]):
            dataset.at[i,"Agena"]=1
    
train_df["Agena"].describe()


# In[134]:


train_df["Embarked"].describe()


# In[250]:


for dataset in combine:
    dataset.fillna({"Embarked":"S","Age":dataset["Age"].median(),"Fare":dataset["Fare"].mean()},inplace=True)
    print(dataset.isnull().any())

print(train_df.isnull().any())
train_df.describe()


# In[199]:


print(type(train_df.at[0,"Sex"]))
#print(train_df.info())
print(type(combine))
    


# In[251]:


dummylist=[]
numlist=[]
for i in train_df.columns:
    print(type(train_df.at[0,i]))
    if type(train_df.at[0,i])==str:
        print(i)
        dummylist.append(i)
    else:
        numlist.append(i)
dummylist.remove("Name")
print(dummylist)
test_df['Survived']=np.nan
combine1=[]
for dataset in combine:
    #print(pd.get_dummies(dataset[dummylist],drop_first=True,dummy_na=False,columns=None))
    d1=pd.concat([dataset[numlist],pd.get_dummies(dataset[dummylist],drop_first=True,dummy_na=False,columns=None)],axis=1)
    combine1.append(d1)

print(dataset.describe())
print(train_df.describe())
print(combine1[1].describe())
train_df1=combine1[0]
test_df1=combine1[1]


# In[218]:


grid=sns.FacetGrid(train_df1,row="Survived",col="Agena")
grid.map(plt.hist,"Age",bins=15)


# In[222]:


print(min(train_df1["Age"]))
print(train_df1[train_df1["Age"]<=5].describe())
print(train_df1[(train_df1["Age"]<=10) & (train_df1["Age"]>5)].describe())
print(train_df1[(train_df1["Age"]<=15) & (train_df1["Age"]>10)].describe())


# In[252]:


for dataset in combine1:
    dataset["Agebin"]=15
    for i in range(15):
        dataset.loc[(dataset["Age"]<=5*(i+1)) & (dataset["Age"]>5*i),"Agebin"]=i

print(dataset["Agebin"].describe())
print(train_df1["Agebin"].describe())
print(test_df1["Agebin"].describe())


# In[227]:


train_df1.describe()


# In[228]:


pg=sns.pairplot(train_df1)
print(type(pg))


# In[253]:


for dataset in combine1:
    dataset["Family"]=dataset["SibSp"]+dataset["Parch"]
    dataset["alone"]=0
    dataset.loc[dataset["Family"]==0,"alone"]=1
    dataset["logage"]=np.log(dataset["Age"]+1)
    
print(train_df1.describe())
print(test_df1.describe())


# ここからモデルに　長かったしよくわからなかった…

# In[254]:


x_train=train_df1.drop(["Survived","PassengerId"],axis=1)
y_train=train_df1["Survived"]
x_test=test_df1.drop(["Survived","PassengerId"],axis=1)
x_train.shape,y_train.shape,x_test.shape


# In[ ]:




