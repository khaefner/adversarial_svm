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

def clean_data(data):
    ## Clean up data
    data[data['service']=='-']
    data['service'].replace('-',np.nan,inplace=True)
    data.dropna(inplace=True)
    print("After cleanup, shape:{}".format(data.shape))
    print("Attack Categories")
    print(data['attack_cat'].value_counts())
    return data
    


def one_hot_encoding(data):
    num_col = data.select_dtypes(include='number').columns
    # selecting categorical data attributes
    cat_col = data.columns.difference(num_col)
    # creating a dataframe with only categorical attributes
    data_cat = data[cat_col].copy()
    data_cat = pd.get_dummies(data_cat,columns=cat_col)
    data = pd.concat([data, data_cat],axis=1)
    data.drop(columns=cat_col,inplace=True)
    #print(data.shape)
    return data

def data_normalization(data):
    # selecting numeric attributes columns from data
    num_col = list(data.select_dtypes(include='number').columns)
    try:
        num_col.remove('id')
        num_col.remove('label')
    except:
        pass
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for i in num_col:
        arr = data[i]
        arr = np.array(arr)
        data[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
    #print(data.head())
    return data

def binary_labels(data):
    # changing attack labels into two categories 'normal' and 'abnormal'
    bin_label = pd.DataFrame(data.label.map(lambda x:'normal' if x==0 else 'abnormal'))
    # creating a dataframe with binary labels (normal,abnormal)
    bin_data = data.copy()
    bin_data['label'] = bin_label
   # label encoding (0,1) binary labels
    le1 = preprocessing.LabelEncoder()
    enc_label = bin_label.apply(le1.fit_transform)
    bin_data['label'] = enc_label 
    np.save("le1_classes.npy",le1.classes_,allow_pickle=True)
    num_col = list(data.select_dtypes(include='number').columns)
    corr_bin = bin_data[num_col].corr()
    corr_ybin = abs(corr_bin['label'])
    highest_corr_bin = corr_ybin[corr_ybin >0.3]
    highest_corr_bin.sort_values(ascending=True)
    bin_cols = highest_corr_bin.index
    bin_data = bin_data[bin_cols].copy()
    #print(bin_data)
    bin_data.to_csv('./datasets/bin_data.csv')
    return bin_data 

def multi_labels(data):
    multi_data = data.copy()
    multi_label = pd.DataFrame(multi_data.attack_cat)
    multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])
    le2 = preprocessing.LabelEncoder()
    enc_label = multi_label.apply(le2.fit_transform)
    multi_data['label'] = enc_label
    return multi_data

def svm_multi_fit(data):
    X = data.drop(columns=['label'],axis=1)
    Y = data['label']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30, random_state=100)
    lsvm_multi = SVC(C=1.0, kernel="rbf")
    lsvm_multi.fit(X_train,y_train) 
    y_pred = lsvm_multi.predict(X_test) 
    print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
    print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

    
def linear_regression(bin_data):
    X = bin_data.drop(columns=['label'],axis=1)
    Y = bin_data['label']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=50)
    lr_bin = LinearRegression()
    lr_bin.fit(X_train, y_train)
    y_pred = lr_bin.predict(X_test)
    round = lambda x:1 if x>0.6 else 0
    vfunc = np.vectorize(round)
    y_pred = vfunc(y_pred)
    print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
    print("Accuracy - ",accuracy_score(y_test,y_pred)*100)

def multi_linear_regression(data):
    X = data.drop(columns=['label'],axis=1)
    Y = data['label']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=50)
    lr_bin = LinearRegression()
    lr_bin.fit(X_train, y_train)
    y_pred = lr_bin.predict(X_test)
    round = lambda x:1 if x>0.6 else 0
    vfunc = np.vectorize(round)
    y_pred = vfunc(y_pred)
    print("Mean Absolute Error - " , metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error - " , metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error - " , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2 Score - " , metrics.explained_variance_score(y_test, y_pred)*100)
    print("Accuracy - ",accuracy_score(y_test,y_pred)*100)


def multi_adversarial_clean_svm(data):
    #X = data.drop(columns=['attack_cat_DoS','attack_cat_Exploits','attack_cat_Fuzzers','attack_cat_Generic','attack_cat_Normal','attack_cat_Reconnaissance','attack_Cat_Worms'])
    Y = data[['attack_cat_DoS','attack_cat_Exploits','attack_cat_Fuzzers','attack_cat_Generic','attack_cat_Normal','attack_cat_Reconnaissance','attack_cat_Worms','attack_cat_Analysis','attack_cat_Backdoor']]
    X = data.drop(columns=['label','attack_cat_DoS','attack_cat_Exploits','attack_cat_Fuzzers','attack_cat_Generic','attack_cat_Normal','attack_cat_Reconnaissance','attack_cat_Worms','attack_cat_Analysis','attack_cat_Backdoor'])
    X = data_normalization(X)
    print(X.head())
    random_state = 999
    from art.estimators.classification import SklearnClassifier
    #clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    #clf = SVC(kernel='linear',gamma='auto')
    #model = SVC(kernel='linear',gamma='auto')
    model = SVC(C=1.0, kernel="rbf")
    clf = SklearnClassifier(model=model)
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=random_state)
    print(X_train.shape,y_train.shape)
    print(X_train)
    print(y_train)
    print(list(X_train))
    clf.fit(X_train,y_train)
    # Compute predictions on a test set
    y_pred = clf.predict(X_test)
    #N = y_test.shape
    #accuracy = (y_test == y_pred).sum() / N
    #print(accuracy)
    accuracy = np.sum(np.argmax(y_pred) == np.argmax(y_test)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))


def adversarial_clean_svm(data):
    X = data.drop(columns=['label'],axis=1)
    Y = data['label']
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=50)
    random_state = 999
    from art.estimators.classification import SklearnClassifier
    #clf = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    #clf = SVC(kernel='linear',gamma='auto')
    #model = SVC(kernel='linear',gamma='auto')
    model = SVC(C=1.0, kernel="rbf")
    clf = SklearnClassifier(model=model)
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=random_state)
    print(X_train.shape,y_train.shape)
    print(X_train)
    print(y_train)
    print(list(X_train))
    clf.fit(X_train,y_train)
    # Compute predictions on a test set
    y_pred = clf.predict(X_test)
    #N = y_test.shape
    #accuracy = (y_test == y_pred).sum() / N
    #print(accuracy)
    accuracy = np.sum(np.argmax(y_pred) == np.argmax(y_test)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

def plot_features():
    features = pd.read_csv('datasets/UNSW_NB15_features.csv')

    #Clean up Features
    features['Type '] = features['Type '].str.lower()

    # selecting column names of all data types
    nominal_names = features['Name'][features['Type ']=='nominal']
    integer_names = features['Name'][features['Type ']=='integer']
    binary_names = features['Name'][features['Type ']=='binary']
    float_names = features['Name'][features['Type ']=='float']

    # selecting common column names from dataset and feature dataset
    cols = data.columns
    nominal_names = cols.intersection(nominal_names)
    integer_names = cols.intersection(integer_names)
    binary_names = cols.intersection(binary_names)
    float_names = cols.intersection(float_names)
    plt.figure(figsize=(8,8))
    plt.pie(data.label.value_counts(),labels=['normal','abnormal'],autopct='%0.2f%%')
    plt.title("Pie chart distribution of normal and abnormal labels",fontsize=16)
    plt.legend()
    plt.savefig('plots/Pie_chart_binary.png')
    #plt.show()
    plt.figure(figsize=(8,8))
    plt.pie(data.attack_cat.value_counts(),labels=data.attack_cat.unique(),autopct='%0.2f%%')
    plt.title('Pie chart distribution of multi-class labels')
    plt.legend(loc='best')
    plt.savefig('plots/Pie_chart_multi.png')
    #plt.show()


# Import Dataset
data = pd.read_csv('datasets/UNSW_NB15.csv')
print("Total data shape:{}".format(data.shape))
data = clean_data(data)
print(data)
data = one_hot_encoding(data)
print(list(data))
#adversarial_clean_svm(data)




#svm_multi_fit(data)
#multi_linear_regression(data)
#multi_adversarial_clean_svm(data)

