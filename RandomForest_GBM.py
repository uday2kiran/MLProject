#!/usr/bin/env python
# coding: utf-8

# In[34]:



import os 
import numpy as np 
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#import xgboost



# In[36]:


os.chdir("C:\DataScience\PR\AllDataSets")


# In[37]:


#chrn = pd.read_csv("Churn.csv")
chrn = pd.read_csv("Churn_Modelling.csv")


# In[38]:


chrn.head()


# In[39]:


chrn.dtypes


# In[40]:



for i in chrn.columns:
    print (chrn[i].name , chrn[i].dtype)


# In[41]:


chrn.describe()


# In[42]:


chrn["Dummy"] = 1


# In[43]:


chrn.head()


# In[44]:


chrn.head()


# In[45]:


chrn = chrn.drop( [ "RowNumber","CustomerId","Surname", "Geography", "Gender"], axis = 1)


# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[47]:


target = chrn["Exited"]
chrn = chrn.drop(["Exited"], axis=1)


# In[48]:


chrn.head()


# In[49]:


x_train, x_test, y_train, y_test = train_test_split( chrn, target, test_size=0.2, random_state=42)


# In[50]:


rf = RandomForestClassifier()


# In[51]:


rf = RandomForestClassifier(n_estimators=50, max_features= 5, max_depth=7,random_state = 123 )


# In[52]:


rf.fit( x_train, y_train )


# In[53]:


y_pred = rf.predict( x_test)


# In[54]:


from sklearn.metrics import confusion_matrix


# In[55]:


confusion_matrix( y_test, y_pred)


# In[56]:


from sklearn.metrics import classification_report


# In[57]:


classification_report (y_test, y_pred)


# In[58]:


sklearn.metrics.accuracy_score(y_test, y_pred) 


# In[59]:


from sklearn.ensemble import GradientBoostingClassifier


# In[60]:


gb = GradientBoostingClassifier(verbose=1, random_state=123, n_estimators=500, learning_rate=0.05,  max_features=6, max_depth = 5, min_samples_leaf=4)


# In[61]:


gb.fit(x_train, y_train)


# In[62]:


y_pred = gb.predict( x_test)


# In[63]:


y_pred[1:10]


# In[64]:


confusion_matrix(y_test,y_pred)


# In[65]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridsearvchCV
from sklearn.model_selection import GridSearchCV


# In[66]:


import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[67]:


cv_scores = cross_val_score(gb, x_train, y_train, cv=5, scoring="roc_auc")


# In[68]:


"Accuracy: %0.5f (+/- %0.5f)"%(cv_scores.mean(), cv_scores.std())


# In[69]:


cv_scores


# In[70]:


metrics.accuracy_score(y_test, y_pred)


# In[71]:


gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [2, 3,4,5],
              #'max_features': [1.0, 0.3, 0.1] 
              }
print(gb_grid_params)

gb_gs = GradientBoostingClassifier(n_estimators = 50)

clf = GridSearchCV(gb_gs,
                   gb_grid_params,
                   cv=2,
                   scoring='roc_auc',
                   verbose = 3, 
                   n_jobs=4);
clf.fit(x_train, y_train);


# In[72]:


dir(clf)


# In[73]:


clf.best_params_


# In[74]:


clf.best_score_


# In[75]:


clf.best_estimator_


# In[76]:


import pickle 
pickle.dump('rf',open("GBM1.pkl",'wb'))

