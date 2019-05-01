#!/usr/bin/env python
# coding: utf-8

# In[210]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


# In[211]:


Train_data = pd.read_csv('../../Data_Part1/Train_data_part1.csv').values
Test_data = pd.read_csv('../../Data_Part1/Test_data_part1.csv').values


# In[212]:


#print(Test_data)


# In[213]:


X_train = Train_data[:,:4]
Y_train = Train_data[:,4]

X_test = Test_data[:,:4]
Y_test = Test_data[:,4]


# In[214]:


# print(X_train)
# print(Y_train)
# print(X_test)


# In[225]:


# Conversion of strings into integer types

# Price and Maintenance are ordinal variables, Airbag is a categorical variable.
# Profitable is also a categorical variable.
#from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

LE1 = LabelEncoder()
LE2 = LabelEncoder()
LE3 = LabelEncoder()
LE4 = LabelEncoder()


LE1.fit(X_train[:,3])
LE2.fit(Y_train)
LE3.fit(X_train[:,0])
LE4.fit(X_train[:,1])

X_train[:,3] = LE1.transform(X_train[:,3])
X_test[:,3] = LE1.transform(X_test[:,3])
Y_train = LE2.transform(Y_train)
Y_test = LE2.transform(Y_test)

X_train[:,0] = LE3.transform(X_train[:,0])
X_train[:,0] = np.where(X_train[:,0]==0,int(3),X_train[:,0])

X_test[:,0] = LE3.transform(X_test[:,0])
X_test[:,0] = np.where(X_test[:,0]==0,int(3),X_test[:,0])

X_train[:,1] = LE4.transform(X_train[:,1])
X_train[:,1] = np.where(X_train[:,1]==0,int(3),X_train[:,1])

X_test[:,1] = LE4.transform(X_test[:,1])
X_test[:,1] = np.where(X_test[:,1]==0,int(3),X_test[:,1])


# In[216]:


# print(X_train)
# print("\n",X_test)


# In[217]:


#print(Y_train)


# In[218]:


# Using Gini index
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)
Y_pred = DT.predict(X_test)


# In[219]:


print("Predictions on test set:\n",Y_pred)


# In[220]:


Test_acc = accuracy_score(Y_test,Y_pred)
print("Test Accuracy = ",Test_acc)


# In[221]:


#print(DT.get_params())


# In[222]:


#print(DT.decision_path(X_train))


# In[202]:


#print(DT.feature_importances_)


# In[203]:


import graphviz
from sklearn import tree


# In[204]:


data = tree.export_graphviz(DT, out_file=None, feature_names=['price','maintenance','capacity','airbag'],filled=True
                           ,rounded=True,special_characters=True)
graph = graphviz.Source(data)
#graph.render("Cars_gini")


# In[205]:


#graph


# In[226]:


# Using Information Gain
# from sklearn.preprocessing import LabelEncoder

# LE1 = LabelEncoder()
# LE2 = LabelEncoder()
# LE3 = LabelEncoder()
# LE4 = LabelEncoder()


# LE1.fit(X_train[:,3])
# LE2.fit(Y_train)
# LE3.fit(X_train[:,0])
# LE4.fit(X_train[:,1])

# X_train[:,3] = LE1.transform(X_train[:,3])
# X_test[:,3] = LE1.transform(X_test[:,3])
# Y_train = LE2.transform(Y_train)
# Y_test = LE2.transform(Y_test)

# X_train[:,0] = LE3.transform(X_train[:,0])
# X_train[:,0] = np.where(X_train[:,0]==0,int(3),X_train[:,0])

# X_test[:,0] = LE3.transform(X_test[:,0])
# X_test[:,0] = np.where(X_test[:,0]==0,int(3),X_test[:,0])

# X_train[:,1] = LE4.transform(X_train[:,1])
# X_train[:,1] = np.where(X_train[:,1]==0,int(3),X_train[:,1])

# X_test[:,1] = LE4.transform(X_test[:,1])
# X_test[:,1] = np.where(X_test[:,1]==0,int(3),X_test[:,1])


DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,Y_train)
Y_pred = DT.predict(X_test)


# In[227]:


print("Predictions on test set:\n",Y_pred)
Test_acc = accuracy_score(Y_test,Y_pred)
print("Test Accuracy = ",Test_acc)


# In[209]:


data = tree.export_graphviz(DT, out_file=None, feature_names=['price','maintenance','capacity','airbag'],filled=True
                           ,rounded=True,special_characters=True)
graph = graphviz.Source(data)
#graph.render("Cars_entropy")


# In[188]:


#graph


# In[ ]:




