#!/usr/bin/env python
# coding: utf-8

# パッケージ

# In[1]:


import pandas as pd
import numpy as np
import random as rnd


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


# In[7]:


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


# In[ ]:




