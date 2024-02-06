# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:56:03 2024

@author: Ashish Chincholikar
Decision Tree
"""
import pandas as pd

df = pd.read_csv("C:/Supervised_ML/Decision_Tree/Data_Set/salaries.csv")
df.head()
inputs = df.drop('salary_more_then_100k' ,  axis = 'columns')
target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company' , 'job' , 'degree'] , axis = 'columns')
target

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n , target)
#is salary of Google , Computer Engineering , Bachelors degree > 100k 
model.predict([[2,1,0]])
#is salary of google , computer engineering , Masters degree > 100k
model.predict([[2,1,1]])

#op array[0] -->salary not greater than 100k
#op array[1] -->salary greater than 100k

















