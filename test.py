import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

df=load_boston()

dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names

## Independent features and dependent features
X=dataset
y=df.target

## train test split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


## standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


regression=LinearRegression()
regression.fit(X_train,y_train)
mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
##prediction 
reg_pred=regression.predict(X_test)
print("Prediction------>",reg_pred)

score=r2_score(reg_pred,y_test)
print("r2_score",score)