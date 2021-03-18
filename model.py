import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv("studentdata.csv", delimiter=",")

def addData(request):
    global data
    formData = pd.json_normalize(request)
    idx = 0
    for entry in formData.values:
        if data.loc[data.Date == entry[0]].empty:
            newData = pd.DataFrame([entry], columns=data.columns)
            data = data.append(newData, ignore_index=True)
        else:
            data.loc[data.Date == entry[0]] = entry
    data.to_csv("studentdata.csv", index=False)
    
def run(request):
    addData(request)
    global data
    features = ["Work", "School", "Life", "Exercise"]
    X = data[features].values
    y = data["Happiness"].values
    model = LinearRegression()
    model.fit(X, y)
    importance = model.coef_

    most_important = ("", 0)
    most_harmful = ("", 10000)
    
    for i, j in enumerate(importance):
        if j > most_important[1]:
            most_important = (features[i], j)
        if j < most_harmful[1]:
            most_harmful = (features[i], j)

    return(
        f"{most_important[0]} tasks contributed the most to your happiness and {most_harmful[0].lower()} tasks took the largest toll on it"
    )





