import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import json
    
def run(request):
    data = pd.json_normalize(request)
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
        json.dumps({most_important[0]: round(most_important[1], 2), most_harmful[0]: round(most_harmful[1], 2)})
    )





