#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("seattle-weather.csv")


# In[4]:


df.head()


# In[5]:


df.isna().sum()


# In[6]:


X= df.drop(columns= ['date', 'weather'])
y= df['weather']


# In[7]:


X


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)


# In[12]:


X_train


# In[13]:


y_train


# In[14]:


y_test


# In[15]:


X_test


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


model = LogisticRegression()


# In[18]:


model.fit(X_train, y_train)


# In[19]:


predictions = model.predict(X_test)


# In[20]:


predictions


# In[21]:


from sklearn.metrics import classification_report


# In[22]:


print(classification_report(y_test, predictions))


# In[24]:


test_weather ={
    'precipitation': 10 ,
    'temp_max' :20,
    'temp_min' :10,
    'wind':5.4
}

test_df = pd.DataFrame([test_weather])
test_df


# In[25]:


model.predict(test_df)


# In[ ]:





# In[ ]:





# In[ ]:




