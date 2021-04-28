#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")


# In[4]:


df.head()


# In[8]:


data=df[['name','reviews.rating','reviews.text']]


# In[9]:


data.head()


# In[11]:


len(data)


# In[12]:


len(data.dropna())


# In[14]:


data=data.dropna()


# In[16]:


for i in range(0,len(data)-1):
    if type(data.iloc[i]['reviews.text']) != str:
        data.iloc[i]['reviews.text'] = str(data.iloc[i]['reviews.text'])


# In[17]:


data = data[data['reviews.rating'] != 3]


# In[18]:


data.head()


# In[19]:


products=data


# In[20]:


products.head()


# In[22]:


def sentiment(n):
    return 1 if n >= 4 else 0
products['sentiment'] = products['reviews.rating'].apply(sentiment)
products.head()


# In[23]:


def combined_features(row):
    return row['name'] + ' '+ row['reviews.text']
products['all_features'] = products.apply(combined_features, axis=1)
products.head()


# In[24]:


X=products["all_features"]
y=products['sentiment']


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(ctmTr, y_train)


# In[28]:


y_pred_class = model.predict(X_test_dtm)


# In[29]:


accuracy_score(y_test, y_pred_class)


# In[ ]:




