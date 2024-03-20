import numpy as np
import math
import os
from PIL import Image
import matplotlib.pyplot as plt
def loadImages(Foldername):
    count = 0
    images = []
    for img in os.listdir(Foldername):
        count+=1
        images.append(list(np.asarray(Image.open(os.path.join(Foldername, img))).flatten()))
    return images
train_images = []
for i in range(10):
    a = list(loadImages(f'/home/akash/Downloads/WhatSie/SML/trainingSet/trainingSet/{i}'))
    train_images+=a
class PCA:
    def mean(x):
        return sum(x)/len(x)
    def standard_deviation(numbers):
        n = len(numbers)
        if n < 2:
            return 0.0

        mean = sum(numbers) / n
        variance = sum((x - mean) ** 2 for x in numbers) / (n - 1)
        std_dev = variance ** 0.5
        return std_dev
    def cov_matrix(S):
        return np.cov(S.T,ddof=0)
    def eigenvalues(S):
        return np.linalg.eigvals(S)
    def eigenvectors(S):
        return np.linalg.eig(S)
    def ratio(l,i):
        return sum(l[:i])/(sum(l))
# data = np.array([[1,5,3,1],[4,2,6,3],[1,4,3,2],[4,4,1,1],[5,5,2,3]])
data = np.array(train_images)
means = []
sds = []
for i in data.T:
    means.append(PCA.mean(i))
    sds.append(PCA.standard_deviation(i))
print(means)
print(sds)
Z = data.T.tolist()

for i in range(len(Z)):
    p1 = PCA.mean(Z[i])
    p2 = PCA.standard_deviation(Z[i])
    for j in range(len(Z[i])):
        if(p2!=0):
            Z[i][j] = (Z[i][j]-p1)/p2

Z1 = np.array(Z).T
print('Standardized matrix\n',Z1)
A = PCA.cov_matrix(Z1)
print('Covariance Matrix\n',A)
eigenvalues, eig_vectors = np.linalg.eig(A)
print('Eigenvalues are: \n',eigenvalues)
print('Eigenvectors are: \n',eig_vectors.T)
indices = np.argsort(eigenvalues)
indices = indices[::-1]
sorted_eigenvectors = eig_vectors[:, indices].T
print(sorted_eigenvectors)
eigenvalues[::-1].sort()
print(eigenvalues)
l = [int(i) for i in range(len(data.T)+1)]
ratios = []
for i in l:
    ratios.append(PCA.ratio(eigenvalues,i))
print(ratios)
plt.bar(l,ratios)
plt.plot(l,[0.8]*len(l),color = 'red')
plt.show()
print(ratios)
print(l)
pcs = 0
for i in range(len(l)):
    if(ratios[i]>0.8):
        pcs+=1
print(f"We need select {pcs} PCs to cover at least 80% of the variance")
k = int(input("k : "))
P = sorted_eigenvectors[:k]
P = P.T
# PC = np.matmul(Z1,P)
print(P)
# [[ 0.29520559 -0.56103699  0.55255128  0.54108986]
#  [-0.7810274  -0.26768116  0.4622677  -0.32349894]]
PC = np.matmul(Z1,P).T
for i in range(k):
    print(f"PC{i+1} is {np.array(PC[i])}")
#print(PC)
