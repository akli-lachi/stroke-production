#!/usr/bin/env python
# coding: utf-8

#                      ****projet Stroke****

# 
# 1. Exploratory Data Analysis and vizalisation
# 1.1 But
# 1.2 Recuperer les données
# 1.3 Visualisation
# 1.4 Commentaire
# 
# 2. Data Cleaning
# 2.1 Les nan
# 2.2 les duplicatats
# 2.3 Converting Categorical Features
# 
# 3.  Modeling and evaualtion
# 3.1 separation du dataframe en features(X) et target (y
# 3.1 Train Test Split
# 3.2 Training and Predicting
# 3.2.1 DecisionTreeClassifier
# 3.2.2 XGBClassifier                                                      
# 6.3 Evaluation
# 6.4 interprétation des résultats
# 
# 4. Oversampling 1
# 4.1 Oversampling data_train 
# 4.2 Training and Predicting
# 4.2.1 DecisionTreeClassifier
# 4.2.2 XGBClassifier                                                      
# 4.3 Evaluation
# 4.4 interprétation des résultats
# 
# 5. Oversampling2
# 5.1 Oversampling data_train & data_test 
# 5.2 Training and Predicting
# 5.2.1 DecisionTreeClassifier
# 5.2.2 XGBClassifier
# 5.3 Evaluation
# 5.4 interprétation des résultats
# 
# 6.  Rééquilibrage des classes avec smote
# 6.1 Oversampling data_train & data_test 
# 6.2 Training and Predicting
# 6.2.1 DecisionTreeClassifier
# 6.3 Evaluation
# 6.4 interprétation des résultats  
# 
# 7. Mise en eprodyction & exporter le model 

# **1.Analyse exploratrice des données** 

# *1.1BUT*

# Le travail consiste à prédire si une personne va subir un AVC
# Nous allons essayer de prédire une classification ,la variable target est "stoke".
# >stroke=0 : pas d'ataque cardiaque 
# 
# >stroke=1 : avc 

# *1.2 Recuperer les données*

# In[1]:


#importation des librariries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns


# In[2]:


# package  machine learning 
#normalisation 
from sklearn.preprocessing import RobustScaler 

#separation du dataset 
from sklearn.model_selection import train_test_split, cross_val_score 

#model 
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier


#metriques 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 

#pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[3]:


#package special :
# Sur-échantillonnage ou oversampling
from imblearn.over_sampling  import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#smote 
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

#exporter les model 
import joblib


# In[4]:


# importation du daset
df = pd.read_csv('https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/strokes.csv', index_col = 0)


# In[5]:


#aperçu du dataset 
df.head(3)


# In[6]:


df.tail(3)


# In[7]:


#dimesnion
df.shape


# In[8]:


#le type de colonnes 
df.dtypes


# *1.3 Visualisation*

# In[9]:


#visualisation des valeurs nulles
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


#repartition de la target 
sns.set_style('whitegrid')
sns.countplot(x='stroke',data=df,palette='RdBu_r')


# In[11]:


#repartion du genre entre les deux classes 
sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='gender',data=df,palette='RdBu_r')


# In[12]:


#repartion de la maladie cardiaque  entre les deux classes
sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='heart_disease',data=df,palette='rainbow')


# In[13]:


#visualisation de l'age 
df['age'].hist(bins=30,color='darkred',alpha=0.7)


# In[14]:


# visualisation d’hypertension
sns.countplot(x='hypertension',data=df)


# *1.4commentaire :*

#    
# -présence des nan dans les colonnes bmi.
# 
# -dataset déséquilibré : Le nombre de cas positif (strokes=1) 
#                        est trop faible dans le jeu de données.
#                        
# -la population testée est plutôt en très bonne santé, peu de gens atteint maladie  cardiaque ou soufrant d’hypertension, population jeune, autant de femme que d’homme).
# 
# >>risque de surraaprentissage(oversapling)

# **2.Data Cleaning**

# *2.1.les valeurs nulles nan*

# In[15]:


#detection des Nan
df.isna().sum().sort_values(ascending=False)


# In[16]:


#on repmlace par la mooyenne 
moyenne=round(df['bmi'].mean(),2)


# In[17]:


moyenne


# In[18]:


df["bmi"].fillna(moyenne,inplace=True)


# *2.2.les duplicatats*

# In[19]:


#detection des doublons
df.duplicated().sum()


# *2.3.Converting Categorical Features*

# In[20]:


df.dtypes


# In[21]:


#gender
# On récupère l'index de la ligne gender=other
indexOther = df[df['gender'] == 'Other'].index
# On supprime la ligne du dataFrame
df.drop(indexOther , inplace=True)

# On remplace les modalités 'Male' et 'Female' de la variable 'gender' par 0, 1
df.gender = df.gender.replace(['Male','Female'], [0,1])


# In[22]:


#ever_married
#On remplace les modalités 'No' et 'Yes' de la variable 'ever_married' par 0 et 1
df["ever_married"]= df["ever_married"].replace(['No','Yes'], [0,1])


# In[23]:


#Residence_type
# On remplace les modalités 'Rural' et 'Urban' de la variable 'Residence_type' par 0 et 1
df.Residence_type = df.Residence_type.replace(['Rural','Urban'], [0,1])


# In[24]:


#work_type'
#On supprime la colonne 'work_type'
df = df.drop(['work_type'], axis = 1)


# In[25]:


#smoking_status
# On sépare la colonne 'smoking_status' en plusieurs colonnes
df = pd.get_dummies(df, columns=['smoking_status'])


# In[26]:


#verfier les modification 
df.head(3)


# In[27]:


#revoir la nouvelle dimension avec les colonnes crées 
df.shape


# **3.Modeling and evaualtion**

# *3.1separation du dataframe  en features(X) et target (y)*

# In[28]:


#separation des variables explicatives dans un dataframe X et la variable cible dans y.
X = df.drop("stroke", axis = 1)
y = df["stroke"]


# In[29]:


print(" X",X.shape,"\n","y",y.shape)


# *3.2Train Test Split*

# In[30]:


#Separation du dataset en train_set et test_set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[31]:


# On affiche les dimensions des datasets après avoir appliqué la fonction 
print(" X_train" ,X_train.shape,"\n" ,"X_test",X_test.shape,"\n","y_train",y_train.shape,"\n","y_test",y_test.shape)


# *3.2.Training and Predicting*

# *3.2.1 DecisionTreeClassifier*

# In[32]:


pipeline_dt=make_pipeline(RobustScaler(),DecisionTreeClassifier())


# In[33]:


hyperparametres_dt={
"decisiontreeclassifier__max_depth":[150, 155, 160],
"decisiontreeclassifier__min_samples_split":[1, 2, 3],
"decisiontreeclassifier__min_samples_leaf":[1, 2, 3]  
}


# In[34]:


machine_dt=GridSearchCV(pipeline_dt,hyperparametres_dt ,cv=10)


# In[35]:


#entrainement 
machine_dt.fit(X_train,y_train)


# In[36]:


machine_dt.best_score_


# In[37]:


predictions_dt = machine_dt.predict(X_test)


# *3.2.2 XGBClassifier*

# In[38]:


pipeline_XG=make_pipeline(RobustScaler(),XGBClassifier())


# In[39]:


hyperparametres_XG={
"xgbclassifier__learning_rate" :[0.1],
"xgbclassifier__max_depth" :[10],
"xgbclassifier__min_child_weight" :[5],
"xgbclassifier__n_estimators" :[100], 
"xgbclassifier__subsample" :[0.45]}


# In[40]:


machine_XG=GridSearchCV(pipeline_XG,hyperparametres_XG,cv=10)


# In[41]:


#entrainement 
machine_XG.fit(X_train,y_train)


# In[42]:


machine_XG.best_score_


# In[43]:


predictions_XG = machine_XG.predict(X_test)


# *3.3 Evaluation*

# In[44]:


#Mesure de performance classification du model DecisionTreeClassifier 

#matrci de confusion
m=confusion_matrix(y_test,predictions_dt)
print("\nMatrice de Confusion:\n", m) 

#reposrting de classification XGBClassifier
reporting=classification_report(y_test,predictions_dt)
print("\nRapports de classification du modèle DecisionTreeClassifier:\n", reporting)


# In[45]:


#Mesure de performance classification XGBClassifier

#matrci de confusion
m=confusion_matrix(y_test,predictions_XG)
print("\nMatrice de Confusion:\n", m) 

#reposrting de classification 
reporting=classification_report(y_test,predictions_XG)
print("\nRapports de classification du modèle XGBClassifier :\n", reporting)


# *3.4 Interprétation des résultats*

# Les deux model indiquement clairement que la classe 1 n’est pas prédictible du fait de déséquilibre du dataset

# Classification déséquilibrée:
# 
# Le nombre de cas positif (strokes=1) est trop faible dans le jeu de données. 
# Nous allons augmenter le nombre d’observations de la classe minoritaire (oversampling). Nous privilégions le sur-échantillonnage car nous avons seulement quelques milliers de données.

# **4.Oversampling**

# *4.1 Oversampling data_train*

# In[46]:


# Sur-échantillonnage ou oversampling
rOs = RandomOverSampler()
X_rOs_train,y_rOs_train  = rOs.fit_resample(X_train, y_train)


# In[47]:


# On affiche les dimensions des datasets après avoir appliqué la fonction 
print(" X_rOs_train" ,X_rOs_train.shape,"\n" ,"X_test",X_test.shape,"\n","y_rOs_train",y_rOs_train.shape,"\n","y_test",y_test.shape)


# avant oversampling  : 
# 
#   X_train (3576, 12)   X_test (1533, 12)
#   y_train (3576,)      y_test (1533
#                            
# apres oversampling:
# 
#   X_rOs_train(6832, 12)    X_test (1533, 12) 
#   y_rOs_train (6832,)       y_test (1533,)                       
#                        

# In[48]:


#la nouvelle repartion de la target 
sns.set_style('whitegrid')
sns.countplot(x=y_rOs_train,palette='RdBu_r')
print("Total lignes:",len(y_train))
print("Lignes 1:",len(np.where(y_rOs_train==1)[0]))
print("Lignes 0:",len(np.where(y_rOs_train==0)[0]))


# *4.2Training and Predicting*

# *4.2.1 DecisionTreeClassifier*

# In[49]:


pipeline_dt2=make_pipeline(RobustScaler(),DecisionTreeClassifier())


# In[50]:


hyperparametres_dt2={
"decisiontreeclassifier__max_depth":[150, 155, 160],
"decisiontreeclassifier__min_samples_split":[1, 2, 3],
"decisiontreeclassifier__min_samples_leaf":[1, 2, 3]  
                     }


# In[51]:


machine_dt2=GridSearchCV(pipeline_dt2,hyperparametres_dt2 ,cv=10)


# In[52]:


#entrainement 
machine_dt2.fit(X_rOs_train,y_rOs_train)


# In[53]:


machine_dt2.best_score_


# In[54]:


predictions_dt2 = machine_dt2.predict(X_test)


# *4.2.2 XGBClassifier*

# In[55]:


pipeline_XG2=make_pipeline(RobustScaler(),XGBClassifier())


# In[56]:


hyperparametres_XG2={
"xgbclassifier__learning_rate" :[0.1],
"xgbclassifier__max_depth" :[10],
"xgbclassifier__min_child_weight" :[5],
"xgbclassifier__n_estimators" :[100], 
"xgbclassifier__subsample" :[0.45]}


# In[57]:


machine_XG2=GridSearchCV(pipeline_XG2,hyperparametres_XG2,cv=10)


# In[58]:


#entrainement 
machine_XG2.fit(X_rOs_train,y_rOs_train)


# In[59]:


machine_XG2.best_score_


# In[60]:


predictions_XG2 = machine_XG2.predict(X_test)


# *4.3Evaluation*

# In[61]:


#Mesure de performance classification DecisionTreeClassifier avec oversampling train 

#matrice de confusion
m=confusion_matrix(y_test,predictions_dt2)
print("\nMatrice de Confusion:\n", m) 

#reporting de classification 
reporting=classification_report(y_test,predictions_dt2)
print("\nRapports de classification du modèle DecisionTreeClassifier :\n", reporting)


# In[62]:


#Mesure de performance classification XGBClassifier avec oversampling train

#matrci de confusion
m=confusion_matrix(y_test,predictions_XG2)
print("\nMatrice de Confusion:\n", m) 

#reposrting de classification 
reporting=classification_report(y_test,predictions_XG2)
print("\nRapports de classification du modèleXGBClassifier  :\n", reporting)


# 4.4 interprétation des résultats

#  Malgré l’augmentation du nombre d'observation dans la partie train de 3676 a 6832, la prédiction de la classe 1 est toujours faible malgré une pétrie amélioration.

# **5.Oversampling2**

# *4.1 Oversampling data_train & data_test*

# In[63]:


# Sur-échantillonnage ou oversampling
from imblearn.over_sampling  import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
rOs = RandomOverSampler()
X_rOs_test,y_rOs_test  = rOs.fit_resample(X_test, y_test)


# In[64]:


# On affiche les dimensions des datasets après avoir appliqué la fonction 
print(" X_rOs_train" ,X_rOs_train.shape,"\n" ,"X_rOs_test",X_rOs_test.shape,"\n","y_rOs_train",y_rOs_train.shape,"\n","y_rOs_test",y_rOs_test.shape)


# In[65]:


#la nouvelle repartion de la target 
sns.set_style('whitegrid')
sns.countplot(x=y_rOs_test,palette='RdBu_r')
print("Lignes 1:",len(np.where(y_rOs_test==1)[0]))
print("Lignes 0:",len(np.where(y_rOs_test==0)[0]))


# *5.2 Training and Predicting*

# *5.2.1DecisionTreeClassifier*

# In[66]:


pipeline_dt20=make_pipeline(RobustScaler(),DecisionTreeClassifier())


# In[67]:


hyperparametres_dt20={
"decisiontreeclassifier__max_depth":[150, 155, 160],
"decisiontreeclassifier__min_samples_split":[1, 2, 3],
"decisiontreeclassifier__min_samples_leaf":[1, 2, 3]  
}


# In[68]:


machine_dt20=GridSearchCV(pipeline_dt20,hyperparametres_dt20 ,cv=10)


# In[69]:


#entrainement 
machine_dt20.fit(X_rOs_train,y_rOs_train)


# In[70]:


machine_dt20.best_score_


# In[71]:


predictions_dt20 = machine_dt20.predict(X_rOs_test)


# *5.2.2 XGBClassifier*

# In[72]:


predictions_XG2 = machine_XG2.predict(X_rOs_test)


# *5.3 Evaluation*

# In[73]:


#Mesure de performance classification DecisionTreeClassifier apres oversampling 2

#matrice de confusion
m=confusion_matrix(y_rOs_test,predictions_dt20)
print("\nMatrice de Confusion:\n", m) 

#reporting de classification 
reporting=classification_report(y_rOs_test,predictions_dt20)
print("\nRapports de classification du modèle DecisionTreeClassifier :\n", reporting)


# In[74]:


#Mesure de performance classification XGBClassifier apres oversampling 2

#matrci de confusion
m=confusion_matrix(y_rOs_test,predictions_XG2)
print("\nMatrice de Confusion:\n", m) 

#reposrting de classification 
reporting=classification_report(y_rOs_test,predictions_XG2)
print("\nRapports de classification du modèle  XGBClassifier:\n", reporting)


# *5.4 interprétation des résultats*

# meme avec un oversdampling des deux pârtie , a savoir train_set et test_set ,
# les resulats de la prediction de la classe 1 reste insufisant; 
# le meilluer reuslats obtrenu est 0.43

# **6. Rééquilibrage des classes avec smote** 

# In[75]:


#rééquilibrage des classes avec SMOTE
sm = SMOTE(sampling_strategy='minority')
#train-set
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
#test_set
X_test_smote, y_test_smote = sm.fit_resample(X_test, y_test)


# In[76]:


# On affiche les dimensions des datasets après avoir appliqué la fonction 
print(" X_train_smote" ,X_train_smote.shape,"\n" ,"X_test_smote",X_test_smote.shape,"\n","y_train_smote",y_train_smote.shape,"\n","y_test_smote",y_test_smote.shape)


# In[77]:


#la nouvelle repartion de la target 
sns.set_style('whitegrid')
sns.countplot(x=y_train_smote,palette='RdBu_r')
print("Lignes 1:",len(np.where(y_train_smote==1)[0]))
print("Lignes 0:",len(np.where(y_train_smote==0)[0]))


# In[78]:


#la nouvelle repartion de la target 
sns.set_style('whitegrid')
sns.countplot(x=y_train_smote,palette='RdBu_r')
print("Lignes 1:",len(np.where(y_test_smote==1)[0]))
print("Lignes 0:",len(np.where(y_test_smote==0)[0]))


# *6.2 Training and Predicting*

# In[79]:


pipeline=make_pipeline(RobustScaler(),RandomForestClassifier())


# In[80]:


hyperparametres={
"randomforestclassifier__n_estimators":[10, 50, 100],
"randomforestclassifier__criterion":['gini','entropy'],
"randomforestclassifier__min_samples_split":[1, 2, 3]  
}


# In[81]:


machine=GridSearchCV(pipeline,hyperparametres,cv=10)


# In[82]:


#entrainement 
machine.fit(X_train_smote,y_train_smote)


# In[83]:


machine.best_params_


# In[84]:


machine.best_score_


# In[85]:


predictions = machine.predict(X_test_smote)


# *6.3 Evaluation*

# In[86]:


#Mesure de performance classification DecisionTreeClassifier

#matrci de confusion
m=confusion_matrix(y_test_smote,predictions)
print("\nMatrice de Confusion:\n", m) 

#reposrting de classification 
reporting=classification_report(y_test_smote,predictions)
print("\nRapports de classification du modèle DecisionTreeClassifier :\n", reporting)


# *6.4 interprétation des résultats* 

# In[87]:


#bingo 
#meilleurs resulats en faisant smote sur l'ensemble  du dataframe (train_set et test_set) avec randomforestclassifier


# **7. mise en eprodyction & exporter le model**

# In[88]:


pipeline_finale=make_pipeline(RobustScaler(),
                              RandomForestClassifier( criterion='gini',
                                                      min_samples_split= 2,
                                                      n_estimators= 50)
                             )


# In[89]:


pipeline_finale.fit(X_train_smote,y_train_smote)


# In[90]:


prediction_finale =pipeline_finale.predict(X_test_smote)


# In[91]:


print("\nRapports de classification du modèle rf :\n",classification_report(y_test_smote,prediction_finale))


# In[92]:


#exporter le model 


# In[93]:


joblib.dump(pipeline_finale,"stroke.final")

