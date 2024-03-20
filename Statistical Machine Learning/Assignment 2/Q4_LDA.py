import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
labels = iris.target
data, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=75)

unique_labels, label_indices = np.unique(y_train, return_inverse=True)
grouped_data = [data[label_indices == i] for i in range(len(unique_labels))]

means = []
for i in grouped_data:
    means.append(list(np.mean(i.T,axis=1)))

M = np.mean(data.T,axis=1)

class LDA:
    def in_scatter_matrix(j):
        x = np.mean(j.T, axis=1)
        s = np.zeros((len(j[0]), len(j[0])))
        for i in j:
            a = np.array(i-x).reshape(len(j[0]),1)
            s += np.dot(a,a.T)
        return s

    def between_scatter_matrix(d):
        M = np.mean(data.T,axis=1)
        M.reshape(4,1)
        t = np.zeros((len(data[0]), len(data[0])))
        for j in d:
            m = np.mean(j.T,axis=1).reshape(4,1)
            a = m-M.reshape(4,1)
            t += len(j) * np.dot(a,a.T)
        return t

Sw = np.zeros((len(data[0]), len(data[0])))
for i in grouped_data:
    x = LDA.in_scatter_matrix(i)
    Sw += x
Sb = LDA.between_scatter_matrix(grouped_data)
print("Sw (Within Class scatter Matrix) is: \n",Sw)
print("Sb (Between Class scatter Matrix) is: \n",Sb)
k =  int(input("K: "))
Sw_inv = np.linalg.inv(Sw)
M = np.dot(Sw_inv, Sb)
eigenvalues, eigenvectors = np.linalg.eigh(M)
idx = np.argsort(eigenvalues)[::-1][:k] # Get the indices of the top k eigenvalues
topk_eigenvectors = np.real_if_close(eigenvectors[:,idx]) # Convert the eigenvectors to real values
# print(topk_eigenvectors)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with only kNN:", accuracy*100,"%")


X_train = np.dot(data, topk_eigenvectors)
# print(X_train)
eigenvalues, eigenvectors = np.linalg.eigh(M)
X_test = np.dot(X_test, topk_eigenvectors)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print("Accuracy with LDA and kNN:", accuracy*100,"%")