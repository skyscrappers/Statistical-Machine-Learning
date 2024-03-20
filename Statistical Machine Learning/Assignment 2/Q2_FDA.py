import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df = pd.read_csv('Heart.csv')
t = np.array(df)
dataX = []
for i in t:
    if i[-3] in [0, 1, 2]:
        dataX.append(i[1:])
dataX = np.array(dataX)
dataY = np.array([1 if i[-1] == 'Yes' else 0 for i in dataX])

#Encoding the data(replacing strings with numerical values)
dataX[:, 2] = le.fit_transform(dataX[:, 2])
dataX[:, -2] = le.fit_transform(dataX[:, -2])
dataX[:, len(dataX[0])-1] = le.fit_transform(dataX[:, len(dataX[0])-1])

#Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

#Implementing FDA from scratch
data_zero = []
data_one = []
for i in X_train:
    if i[-1]==0:
        data_zero.append(i[0:len(i)-1])
    else:
        data_one.append(i[0:len(i)-1])
data_zero = np.array(data_zero)
data_one = np.array(data_one)

mean0 = np.mean(data_zero.T,axis=1)
mean1 = np.mean(data_one.T,axis=1)

S0 = np.zeros((13, 13), dtype=np.float64)
for i in data_zero:
    a = np.array(i-mean0).reshape(13,1)
    S0+=np.dot(a,a.T).astype(np.float64)
S1 = np.zeros((13, 13), dtype=np.float64)
for i in data_one:
    a = np.array(i-mean1).reshape(13,1)
    S1+=np.dot(a,a.T).astype(np.float64)
n0 = len(data_zero)
n1 = len(data_one)
S0/=n0-1
S1/=n1-1

w = np.dot(np.linalg.inv(n0*S0 + n1*S1),mean0-mean1)
# print("Direction :: ",w)

#Finding the projected data
projected_data_train = np.array(np.dot(X_train[:,:-1],w.T)).reshape(-1,1)
projected_data_train  = np.hstack((np.ones((projected_data_train.shape[0], 1)), projected_data_train))
projected_data_test = np.array(np.dot(X_test[:,:-1],w.T)).reshape(-1,1)
projected_data_test  = np.hstack((np.ones((projected_data_test.shape[0], 1)), projected_data_test))

#implementing Logistic regression from scratch
max_iter = 50000
learning_rate = 0.1
weight = np.zeros(2).reshape(-1,1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
projected_data_train = np.array(projected_data_train,dtype=np.float64)
for _ in range(max_iter):
    p = sigmoid(np.dot(projected_data_train, weight))
    weight = weight - (learning_rate/(n1+n0))*np.dot(projected_data_train.T, (p-y_train.reshape(-1,1)))
print("Weights: ",weight)


# Predicting Outputs
t = np.dot(projected_data_test,weight)
y_pred = []
for i in t:
    y_pred.append(sigmoid(i[0]))
d = []
for i in y_pred:
    if(i<0.5):
        d.append(0)
    else:
        d.append(1)
d = np.array(d)

accuracy = np.mean(d == y_test)
print("Accuracy with FDA: ",accuracy*100,"%")