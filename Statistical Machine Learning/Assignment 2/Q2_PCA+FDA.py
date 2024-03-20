import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
df = pd.read_csv('Heart.csv')
t = np.array(df)
dataX = []
for i in t:
    if i[-3] in [0, 1, 2]:
        dataX.append(i[1:])
dataX = np.array(dataX)
# Encoding labels
dataY = np.array([1 if i[-1] == 'Yes' else 0 for i in dataX])

# Encoding the data
dataX[:, 2] = le.fit_transform(dataX[:, 2])
dataX[:, -2] = le.fit_transform(dataX[:, -2])
dataX[:, len(dataX[0]) - 1] = le.fit_transform(dataX[:, len(dataX[0]) - 1])

scaler = StandardScaler()
dataX_scaled = scaler.fit_transform(dataX[:, :-1])

pca = PCA(n_components=5)
#Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataX_scaled, dataY, test_size=0.3, random_state=100)

#Applying PCA
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

zeros = X_pca_train[y_train == 0]
ones = X_pca_train[y_train == 1]

#implementing FDA from scratch
mean0 = np.mean(zeros, axis=0)
mean1 = np.mean(ones, axis=0)

S0 = np.zeros((5, 5), dtype=np.float64)
# print(S0.shape)
for i in zeros:
    a = np.array(i-mean0).reshape(5,1)
    S0+=np.dot(a,a.T).astype(np.float64)
S1 = np.zeros((5, 5), dtype=np.float64)
for i in ones:
    a = np.array(i-mean1).reshape(5,1)
    S1+=np.dot(a,a.T).astype(np.float64)
n0 = len(zeros)
n1 = len(ones)
S0/=n0-1
S1/=n1-1
w = np.dot(np.linalg.inv(n0*S0+n1*S1), mean0 - mean1)

# Finding proected data
projected_data_train = np.dot(X_pca_train, w)
projected_data_test = np.dot(X_pca_test, w)

#implementing Logistic regression
def sigmoid(x): return 1 / (1 + np.exp(-x))

max_iter = 500000
learning_rate = 0.1
weight = np.zeros(1)

for _ in range(max_iter):
    p = sigmoid(projected_data_train * weight)
    weight = weight - (learning_rate / len(y_train)) * np.dot(projected_data_train, (p - y_train))


y_pred = sigmoid(projected_data_test * weight)

y_pred_label = np.where(y_pred < 0.5, 0, 1)

accuracy = np.mean(y_pred_label == y_test)
print("Accuracy with PCA+FDA: ", accuracy * 100, "%")