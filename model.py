import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import math

dataset = pd.read_csv("studentdata.csv", delimiter=",")

# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def addData(request):
    global dataset
    formData = pd.json_normalize(request)
    dates = [date for date in formData.Date]
    for i in range(len(dataset.Date)):
        date = dataset.Date.iloc[i]
        for j in range(len(dates)):
            if(date == dates[j]):
                dataset.iloc[i] = formData.iloc[j]
                formData = formData.drop([j], axis=0)
                break
    if(len(formData) > 0):
        dataset = dataset.append(formData, ignore_index=True)
    dataset.to_csv("studentdata.csv", index=False)

X = dataset[["Work", "School", "Life", "Exercise"]].values
y = dataset["Happiness"].values

def run():
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # what are scores for the features
    idx = 0
    total = 0
    most_important = max(fs.scores_)
    for i in range(len(fs.scores_)):
        print(dataset.columns[i + 1], fs.scores_[i])
        if(fs.scores_[i] == most_important):
            idx = i
        total += fs.scores_[i]
    
    most_important = round(most_important, 2)
    return(f"{dataset.columns[idx + 1]} tasks contributed the most to your happiness, with a score of {most_important}!")
    






