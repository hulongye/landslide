import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


data = pd.read_excel('origin.xlsx')
data = data.iloc[:, :].astype('float').values

X1 = data[:,0:14]
X2 = data[:, 14:16]
y = data[:,16:]

min_max_scaler = MinMaxScaler()
enc = OneHotEncoder()
X1 = min_max_scaler.fit_transform(X1)
X2 = enc.fit_transform(X2).toarray()
X = np.concatenate((X1, X2), axis=1)
y = np.array(y).flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
print(rf.score(X_test, y_test))