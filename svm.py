# importing required libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pickle 
from os import path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import secml


## Import Dataset
data = pd.read_csv('datasets/UNSW_NB15.csv')
print("Total data shape:{}".format(data.shape))


#Clean up data
data[data['service']=='-']
data['service'].replace('-',np.nan,inplace=True)
data.dropna(inplace=True)
print("After cleanup, shape:{}".format(data.shape))

print(data.head(n=5))
print("Attack Categories")
print(data['attack_cat'].value_counts())
