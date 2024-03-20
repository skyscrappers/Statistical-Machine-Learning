import pandas as pd
import numpy as np
import math
# from pprint import pprint
import matplotlib.pyplot as plt
df = pd.read_csv('glass.csv')
# print(df)
data = []
for i in df.values:
    data.append(i[0:-1])
data = np.array(data)
# print(data)
class Mahalanobis:
    def covariance(X):
        return np.cov(X.T)
    def m_distance(p,X):
        m = np.mean(X.T,axis=1)
        x1 = np.array(p-m).reshape(9,1)
        sigma = Mahalanobis.covariance(X)
        d = math.sqrt(np.dot(np.dot(x1.T,np.linalg.inv(sigma)),x1))
        return d

distances = []

for i in data:
    distances.append(Mahalanobis.m_distance(i,data))
plt.hist(distances, bins=10)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Mahalanobis Distance Histogram')
plt.show()

hist, bin_edges = np.histogram(distances, bins=10)
print("Frequency: ",hist)
print("Edges: ",bin_edges)

# Calculate the total number of distances
total = len(distances)
max_variance = 19919191
threshold = 0

# Iterate over each bin
for i in range(1, 10):
    
    # Calculate the weights for the two classes (foreground and background)
    w0 = np.sum(hist[:i]) / total
    w1 = np.sum(hist[i:]) / total
    
    # Calculate the mean values for the two classes
    u0=0
    u1=0
    sig0=0
    sig1=0
    for j in range(i):
        u0 += hist[j]*j
    u0 /= sum(hist[:i])

    for j in range(i, 10):
        u1 += hist[j]*j
    u1 /= sum(hist[i:])
    
    for j in range(i):
        sig0+=((j-u0)**2)*hist[j]
    sig0/=sum(hist[:i])
    for j in range(i,10):
        sig1+=((j-u1)**2)*hist[j]
    sig1/=sum(hist[i:])
    variance = w0*sig0 + w1*sig1
    
    # # Check if the variance is greater than the current maximum variance
    if variance < max_variance:
        max_variance = variance
        threshold = bin_edges[i]
        
# Print the threshold value
print("Otsu's threshold:", threshold)

# Plot the histogram and the threshold value
plt.hist(distances, bins=10)
plt.axvline(x=threshold, color='r', linestyle='--', label="Otsu's threshold")
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Mahalanobis Distance Histogram with Otsu Threshold')
plt.legend()
plt.show()

In=[]
out=[]
for i in distances:
    if(i<threshold):
        In.append(i)
    else:
        out.append(i)
print("Inliers: ",len(In))
print("Outliers: ",len(out))