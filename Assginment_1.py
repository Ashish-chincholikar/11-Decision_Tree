# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:44:08 2024

@author: Ashish Chincholikar
Decision Tree Assignment1
"""

"""
1. Business Problem
    For a Cloth Manufacturing company , it is an important of them to know about
    what are the factors that are responsible for high sales for their product.
    so that they can take care of these aspect and develope their product 
    accordingly
    
1.1 what is business objective?
    ~To understand the factors that are leading to high sales of the product and 
    maintain them
    ~
    
1.2 Are there any constraints
    ~
    ~

2. Create a Data Dictionary 
name of feature 
description 
type 
relevance

1. CompPrice , Component Price , ~ , Relavent data
2. Income , Income associated with that product , ~ , Relavent data
3. Advertising , Advertising cost associated with that product , ~  , Relevant data
4. Population , Population to which that product is accesseble ,~ , irrelevant
5. Price , Price of that product , ~ , irreleavant
6. ShelveLoc , Occupation of an individual , ~ , relevant
7. Age , Age group for which that product is targeted to , ~ , irrelevant
8. Education , ~ , ~ , irrelevant
9. Urban  , ~ , ~ , irrelevant
10. US , ~ , ~ , relevant
"""

""" 
3. Data Preprocessing
3.1 Data Cleaning , feature engineering ,etc
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read the csv file
df = pd.read_csv("C:/Supervised_ML/Decision_Tree/Data_Set/Company_Data.csv")

#display the top 5 records
df.head(5)

#columns of the dataframe
df.columns

#shape of the datframe
df.shape

#5 number summary of the dataframe
df.describe

# check for null values
df.isnull()


# False
df.isnull().sum()
# 0 no null values


# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)

df.isnull().sum()
df.dropna()
df.columns

# boxplot
# boxplot on Income column
sns.boxplot(df.Sales)
# In Sales column 2 outliers 


sns.boxplot(df.CompPrice)
# In CompPrice column 2 outliers

# boxplot on df dataframe
sns.boxplot(df)
# There is outliers in almost all columns

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# histplot - show distributions of datasets
sns.histplot(df['Sales'],kde=True)
# normal distribution

sns.histplot(df['CompPrice'],kde=True)
# normal distribution

sns.histplot(df,kde=True)


#converting the Target variable to categorical data by rounding it 
Y = round(df['Sales'])

#Seperating the Target variable from the main dataframe
X = df.drop('Sales' ,  axis = 'columns')

"""
#Normalization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#Now let us apply this function to the dataframe
X = norm_func(X.iloc[: , :10])
"""
#now we use the labelEncoder in order to convert the numeric data to 
#categorical data
from sklearn.preprocessing import LabelEncoder
#le_shelveLoc = LabelEncoder()
#le_Urban = LabelEncoder()
#le_US = LabelEncoder()

enc = LabelEncoder()

X.columns

X['CompPrice_n'] = enc.fit_transform(X['CompPrice'])
X['Income'] = enc.fit_transform(X['Income'])
X['Advertising_n'] = enc.fit_transform(X['Advertising'])
X['Population_n'] = enc.fit_transform(X['Population'])
X['Price_n'] = enc.fit_transform(X['Price'])
X['ShelveLoc_n'] = enc.fit_transform(X['ShelveLoc'])
X['Age_n'] = enc.fit_transform(X['Age'])
X['Education_n'] = enc.fit_transform(X['Education'])
X['Urban_n'] = enc.fit_transform(X['Urban'])
X['US_n'] = enc.fit_transform(X['US'])


X_n = X.drop(['CompPrice', 'Income', 'Advertising', 'Population', 'Price','ShelveLoc', 'Age', 'Education', 'Urban', 'US'] , axis = 'columns')
Y

"""from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X_n , Y , test_size=0.2)


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train , Y_train)

#let us check the model
pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(pred, Y_test))
"""


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_n , Y)
model.predict([[49,11,141,54,0,17,7,1,1]])
#array([10.]
model.predict([[22,16,129,18,1,40,0,1,1]])
#array([11.])

