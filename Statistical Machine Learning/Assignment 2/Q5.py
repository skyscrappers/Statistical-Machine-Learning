#We can apply logistic regression on multi-class classification problem 
# by using 1 vs all approach, for each label we will treat that label as
# 0 and others as 1 and we will do this for each label and we would calculate 
# Probabilities on all new labels generated and then predict the labels
# according to the calculated probabilities
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target
#creating labels 1 vs all
y1 = [] #{0,{1,2}}
y2 = [] #{1,{0,2}}
y3 = [] #{2,{0,1}}
for i in y:
    if(i==0):
        y1.append(0)
    else:
        y1.append(1)
for i in y:
    if(i==1):
        y2.append(0)
    else:
        y2.append(1)
for i in y:
    if(i==2):
        y3.append(0)
    else:
        y3.append(1)
y1 = np.array(y1)
#splitting data into training and testing
X_train, X_test, y_train_1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42)
X_train, X_test, y_train_2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
X_train, X_test, y_train_3, y_test3 = train_test_split(X, y3, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_set = list(set(y))
# print(label_set)
class LogisticRegression:
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def gradient_descent(X,y,n,l):
        weight = np.zeros(5).reshape(-1,1)
        for _ in range(n):
            p = LogisticRegression.sigmoid(np.dot(X_train, weight))
            weight = weight - (l/(len(X)+len(y)))*np.dot(X.T, (p-np.array(y).reshape(-1,1)))
        return weight
learning_rate = 0.1
num_iterations = 1000
#Adding intercept term in training and testing data
X_train  = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test  = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


# Predicting Output from y1
w = LogisticRegression.gradient_descent(X_train,y_train_1,num_iterations,learning_rate)
t = np.dot(X_test,w)
y_pred = []
for i in t:
    y_pred.append(LogisticRegression.sigmoid(i[0]))
p = []
for i in y_pred:
    if(i<0.5):
        p.append(0)
    else:
        p.append(1)
d = np.array(y_pred)
accuracy = np.mean(p == np.array(y_test1))
print("Accuracy with {0,{1,2}}: ",accuracy*100,"%")

# Predicting Output from y2
w1 = LogisticRegression.gradient_descent(X_train,y_train_2,num_iterations,learning_rate)
t1 = np.dot(X_test,w1)
y_pred1 = []
for i in t1:
    y_pred1.append(LogisticRegression.sigmoid(i[0]))
p1 = []
for i in y_pred1:
    if(i<0.5):
        p1.append(0)
    else:
        p1.append(1)
d1 = np.array(y_pred1)
accuracy1 = np.mean(p1 == np.array(y_test2))
print("Accuracy with {1,{0,2}}: ",accuracy1*100,"%")


# Predicting Output from y3
w2 = LogisticRegression.gradient_descent(X_train,y_train_3,num_iterations,learning_rate)
t2 = np.dot(X_test,w2)
y_pred2 = []
for i in t2:
    y_pred2.append(LogisticRegression.sigmoid(i[0]))

p3 = []
for i in y_pred2:
    if(i<0.5):
        p3.append(0)
    else:
        p3.append(1)
d2 = np.array(y_pred2)

accuracy2 = np.mean(p3 == np.array(y_test3))
print("Accuracy with {2,{0,1}}: ",accuracy2*100,"%")

final_labels = []
for i in range(len(y_test)):
    x = min([d[i],d1[i],d2[i]])
    if(x==d[i]):
        final_labels.append(0)
    elif x==d1[i]:
        final_labels.append(1)
    else:
        final_labels.append(2)
print("Predicted labels: ",np.array(final_labels))
print("Actual Labels: ",y_test)
accuracy = np.mean(np.array(final_labels)==y_test)
print("Final Accuracy: ",accuracy*100,"%")


# logistic regression provides better result than LDA in Q4
# beacause in Q4 if we take 2 top-eigenvectors or change 
# hyperparameters thn accuracy dropes
# to 96.6667% however in logistic regression accuracy remains 100% 
# even for varying hyperparameters it predicts very accurate labels.