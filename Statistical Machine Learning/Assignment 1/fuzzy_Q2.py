import random
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
class Fuzzy:
    def calculate_fuzzy_membership(x, centroids, m):
        c = centroids.shape[0] # number of centroids
        # Calculate distances between data point and centroids
        distances = np.linalg.norm(x - centroids, axis=1)
        
        # Calculate membership values
        membership = np.zeros(c)
        for j in range(c):
            sumofdistances = np.sum((distances / distances[j]) ** (2 / (m - 1)))
            membership[j] = 1 / sumofdistances
        
        return membership
    def calculate_fuzzy_centroids(X, U, m):
        n = X.shape[0] # number of data points
        d = X.shape[1] # number of features
        c = U.shape[1] # number of clusters
        
        # Calculate fuzzy centroids matrix
        V = np.zeros((c, d))
        for j in range(c):
            numerator = np.sum((U[:,j]**m)[:, np.newaxis] * X, axis=0)
            denominator = np.sum(U[:,j]**m)
            V[j,:] = numerator / denominator
        
        return V
    def calculate_objective_function(X, U, V, m):
        n = X.shape[0] # number of data points
        k = U.shape[1] # number of clusters
        # Calculate distances between data points and fuzzy centroids
        distances = np.linalg.norm(X[:,:,np.newaxis] - V.T[np.newaxis,:,:], axis=1)
        # Calculate objective function value
        J = np.sum(U**m * distances**2)
        return J

X = np.array(np.load('kmeans_data.npy'))
c = 3 # number of clusters
m = 2 # fuzziness parameter
max_iterations = 100000 # maximum number of iterations
beta = 0.3 # termination condition

U = np.random.rand(X.shape[0], c)
U = U / np.sum(U, axis=1)[:, np.newaxis]

J = np.inf
i=0
while i <(max_iterations):
    V = Fuzzy.calculate_fuzzy_centroids(X, U, m)
    for j in range(c):
        for n in range(X.shape[0]):
            U[n,j] = Fuzzy.calculate_fuzzy_membership(X[n], V, m)[j]
    U = U / np.sum(U, axis=1)[:, np.newaxis]
    J_new = Fuzzy.calculate_objective_function(X, U, V, m)
    if J_new < J:
        J = J_new
        best_U = U.copy()
        best_V = V.copy()
    if np.linalg.norm(U - best_U) < beta:
        break
    i+=1
print("Minimum value of objective function = {}".format(J))

