import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LogisticRegression:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def compute_cost(X, y, w):
        z = np.dot(X, w)
        y_pred = LogisticRegression.sigmoid(z)
        eps = 1e-8
        cost = -np.mean(y * np.log(y_pred+eps) + (1-y) * np.log(1-y_pred+eps))
        return cost

    @staticmethod
    def logistic_regression(X, y, learning_rate, num_iterations):
        # Normalize features
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        
        # Initialize weights to zero
        w = np.zeros(X_norm.shape[1])
        costs = []
        
        # Gradient descent
        for i in range(num_iterations):
            z = np.dot(X_norm, w)
            y_pred = LogisticRegression.sigmoid(z)
            dw = (1/X_norm.shape[0]) * np.dot(X_norm.T, (y_pred - y))
            w -= learning_rate * dw
            cost = LogisticRegression.compute_cost(X_norm, y, w)
            costs.append(cost)
            if i % 100 == 0:
                print(f'Iteration {i}, Cost : {cost}')
            
        return w, costs

# create a LabelEncoder object
le = LabelEncoder()

# load data
df = pd.read_csv('Heart.csv')

# preprocess data
dataX = []
for i in df.values:
    if i[-3] in [0, 1, 2]:
        dataX.append(i[1:-1])
dataX = np.array(dataX)
dataY = np.array([1 if i[-1] == 'Yes' else 0 for i in df.values if i[-3] in [0, 1, 2]])
dataX[:, 2] = le.fit_transform(dataX[:, 2])
dataX[:, -1] = le.fit_transform(dataX[:, -1])
dataX = np.hstack((np.ones((dataX.shape[0], 1)), dataX))

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0)
# train logistic regression model
learning_rate = 0.1
num_iterations = 1000
w, costs = LogisticRegression.logistic_regression(x_train, y_train, learning_rate, num_iterations)

# make predictions on testing data
z = np.dot(StandardScaler().fit_transform(x_test), w)
y_pred = LogisticRegression.sigmoid(z)
y_pred_class = np.round(y_pred)

# evaluate model
accuracy = np.mean(y_pred_class == y_test)
print('Accuracy:', accuracy*100,"%")


plt.plot(costs)
plt.title("Cost over iterations")
plt.xlabel("Iteration number")
plt.ylabel("Cost")
plt.show()