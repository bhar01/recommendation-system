#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.accuracy import rmse,mae
from surprise.model_selection import cross_validate
from collections import defaultdict


# In[12]:


df = pd.read_csv('ratings.csv')
df.drop('timestamp',axis = 1,inplace = True)


# In[13]:


df.isna().sum()


# In[17]:


nm = df['movieId'].nunique()
nu = df['userId'].nunique()


# In[18]:


av = df['rating'].count()
total = nm*nu
miss = total - av
sparsity = (miss/total)*100
print(f'Sparsity: {sparsity}')


# In[22]:


fm = df['movieId'].value_counts() > 3
fm = fm[fm].index.tolist()
fu = df['userId'].value_counts() > 3
fu = fu[fu].index.tolist()
df = df[(df['movieId'].isin(fm))&(df['userId'].isin(fu))]


# In[32]:


cols = ["userId","movieId","rating"]
reader = Reader(rating_scale = (0.5,5))
data = Dataset.load_from_df(df[cols],reader)
train = data.build_full_trainset()
anti = train.build_anti_testset()


# In[35]:


algo = SVD(n_epochs = 25, verbose = True)
cross_validate(algo,data,measures = ['RMSE','MAE'],cv = 5,verbose = True)


# In[36]:


pred = algo.test(anti)


# In[50]:


def rec(pred, n):
    top = defaultdict(list)

    for uid, iid, _, est, _ in pred:
        top[uid].append((iid,est))

    for uid, user_ratings in top.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top[uid] = user_ratings[:n]
    return(top) 
    pass
top = rec(pred, n=3)


# In[51]:


for uid, user_ratings in top.items():
    print(uid,[iid for (iid,rating) in user_ratings])


# In[ ]:




