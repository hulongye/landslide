import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from cnn import cnn


data = pd.read_excel('data.xlsx')
data = data.iloc[:, :].astype('float').values
X1 = data[:,0:11]
X2 = data[:, 11:13]
y = data[:,13:]

min_max_scaler = MinMaxScaler()
enc = OneHotEncoder()
X1 = min_max_scaler.fit_transform(X1)
X2 = enc.fit_transform(X2).toarray()
X = np.concatenate((X1, X2), axis=1)
y = np.array(y).flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


cnn = cnn()
history = cnn.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=100,batch_size=32)

y_pre_cnn = cnn.predict(X_test).ravel()
fpr_cnn, tpr_cnn, threshold_cnn = roc_curve(y_test, y_pre_cnn)
auc_cnn = auc(fpr_cnn, tpr_cnn)

plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_cnn, tpr_cnn, label = '1D CNN(AUC = {:.3f})'.format(auc_cnn))
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc='best')
plt.show()
