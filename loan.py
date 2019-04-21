# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:34:22 2018

@author: swastik
"""
#importing liabraies
import pandas as pd 
import numpy as np
import matplotlib as plt
import seaborn as sns

#importing dataset

test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

X_train = train.iloc[:,1:12]
Y_train = train.iloc[:,-1]
X_test = test.iloc[:,1:12]


X_train.iloc[:,2] = X_train.iloc[:,2].replace({'3+' : 3})
X_test.iloc[:,2] = X_test.iloc[:,2].replace({'3+' : 3})


X_train.isnull().sum()
X_train['Loan_Amount_Term'].unique()
X_test['Loan_Amount_Term'].unique()

X_train['Gender'] = X_train['Gender'].fillna('Male')
X_train['Married'] = X_train['Married'].fillna('Yes')
X_train['Education'] = X_train['Education'].fillna('Graduate')
X_train['Dependents'] = X_train['Dependents'].fillna('0')
X_train['Self_Employed'] = X_train['Self_Employed'].fillna('No')
X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna('360')
X_train['Credit_History'] = X_train['Credit_History'].fillna('1')

 
X_test['Gender'] = X_test['Gender'].fillna('Male')
X_test['Married'] = X_test['Married'].fillna('Yes')
X_test['Education'] = X_test['Education'].fillna('Graduate')
X_test['Dependents'] = X_test['Dependents'].fillna('0')
X_test['Self_Employed'] = X_test['Self_Employed'].fillna('No')   
X_test['Credit_History'] = X_test['Credit_History'].fillna('1')
X_test['LoanAmount'] = X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())
X_test['Loan_Amount_Term'] = X_test['Loan_Amount_Term'].fillna('360')
X_test['Credit_History'] = X_test['Credit_History'].fillna('1')

Var_Corr = X_train.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

X_train.iloc[:,0] = labelencoder.fit_transform(X_train.iloc[:,0])
X_train.iloc[:,1] = labelencoder.fit_transform(X_train.iloc[:,1])
X_train.iloc[:,3] = labelencoder.fit_transform(X_train.iloc[:,3])
X_train.iloc[:,4] = labelencoder.fit_transform(X_train.iloc[:,4])
X_train.iloc[:,-1] = labelencoder.fit_transform(X_train.iloc[:,-1])

X_test.iloc[:,0] = labelencoder.fit_transform(X_test.iloc[:,0])
X_test.iloc[:,1] = labelencoder.fit_transform(X_test.iloc[:,1])
X_test.iloc[:,3] = labelencoder.fit_transform(X_test.iloc[:,3])
X_test.iloc[:,4] = labelencoder.fit_transform(X_test.iloc[:,4])
X_test.iloc[:,-1] = labelencoder.fit_transform(X_test.iloc[:,-1])


onehot = OneHotEncoder(categorical_features= [0,1,2,4,10])
X_test = onehot.fit_transform(X_test).toarray()
X_train = onehot.fit_transform(X_train).toarray()


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l1',C=0.5,verbose=0.001)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)

Y_train[:] = Y_train[:].replace({'Y':1,'N':0})

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(14, kernel_initializer='uniform', activation= 'relu', input_shape = (19,))) 
classifier.add(Dense(14, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(14, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation= 'sigmoid'))
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y_train, batch_size= 10, epochs= 10)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

"""import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=10,activation='relu',input_dim=19))
classifier.add(Dense(units=10,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,Y_train,nb_epoch=10)
Y_pred = classifier.predict(X_test)"""


from sklearn.model_selection import cross_val_score
cv = cross_val_score(classifier,X_train,Y_train,scoring='accuracy')
cv.mean()

from sklearn.model_selection import GridSearchCV

param = [{'batch_size':[10, 20, 40, 60, 80, 100],'epochs':[10, 50, 100],'learn_rate':[0.001, 0.01, 0.1, 0.2, 0.3],'weight_constraint':[1, 2, 3, 4, 5],'neurons':[1, 5, 10, 15, 20, 25, 30]}]
grid = GridSearchCV(estimator = classifier,param_grid=param,scoring='accuracy')
grid.fit(X_train,Y_train)
parameter = grid.best_params_
score = grid.best_score_

np.savetxt('Sample_Submission_ZAuTl8O_FK3zQHh',Y_pred,fmt = '%s')






