#!/usr/bin/env python
# coding: utf-8

# In[141]:


import requests
import pandas as pd


# In[142]:


api_url="https://strokes.akli-lachi.repl.co"


# In[143]:


reponse=requests.get(api_url+ '/ping')


# In[144]:


reponse.text


# In[145]:


# importation du daset
df = pd.read_csv('https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/strokes.csv', index_col = 0)


# In[108]:





# In[146]:


df=df.drop("stroke",axis=1)


# In[147]:


df.head()


# In[148]:


#envoie (client-api)
payload=df.iloc[[0]].to_json()


# In[149]:


#reception 
pd.read_json(payload, typ='frame',orient='columns')


# In[150]:


#envoyer le paylod vers api
reponse=requests.post(api_url + '/predict',
              data=df.iloc[[0]].to_json(),
              headers={'content-type':'application/json'})


# In[151]:


reponse


# In[127]:


reponse.text


# In[140]:


def predict_api_titanic(df,idx):
    playload=df.iloc[[idx]].to_json()
    api_url="https://strokes.akli-lachi.repl.co"
    reponse=requests.post(api_url + '/predict', data=playload, headers={'content-type':'application/json'}  )
    #formater les resulats dans un dictionnaire 
    return {'Stroke':int(reponse.text)}

