import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('glass.csv')
data = []
for i in df.values:
    data.append(i[0:-1])
data = np.array(data)
class LOF:
    def manhattan_distance(p1, p2):
        return np.sum(np.abs(p1 - p2))
    def k_nearest_neighbor(dataset, query_point, k):
        distances = np.sum(np.abs(dataset - query_point), axis=1)
        sorted_distances_indices = np.argsort(distances)
        k_nearest_indices = sorted_distances_indices[1:k+1] # Exclude the query point itself
        k_nearest_neighbors = dataset[k_nearest_indices]
        kth_nearest_index = sorted_distances_indices[k]
        return k_nearest_neighbors, kth_nearest_index
    def kth_nearest_distance(dataset, p, k):
        x,y = LOF.k_nearest_neighbor(dataset, p, k)
        d = dataset[y]
        return LOF.manhattan_distance(p,d)
    def Reachability_distance(p1,p2,k):
        d1 = LOF.manhattan_distance(p1,p2)
        d2 = LOF.kth_nearest_distance(data,p2,k)
        return max(d1,d2)
    def Local_reachability_distance(p,data,k):
        x,y = LOF.k_nearest_neighbor(data,p,k)
        s=0
        for i in x:
            s+=LOF.Reachability_distance(p,i,k)
        t = s/len(x)
        return 1/t
    def LOF_distance(point,dataset,k):
        x,y = LOF.k_nearest_neighbor(dataset,point,k)
        a=0
        for i in x:
            a+=LOF.Local_reachability_distance(i,dataset,k)
        b = LOF.Local_reachability_distance(point,dataset,k)
        return a/(len(x)*b)
LOFs = []
minpts = int(input("Minpts: "))
for i in data:
    LOFs.append(LOF.LOF_distance(i,data,minpts))
# print(LOFs)
plt.hist(LOFs, bins=10)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('LOF score Histogram')
plt.show()

hist, bin_edges = np.histogram(LOFs, bins=10)
print("Frequency: ",hist)
print("Edges: ",bin_edges)

# Calculate the total number of distances
total = len(LOFs)

# Initialize variables for the maximum variance and the threshold value
def Otsu_threshold(hist, bin_edges):
    max_variance = 19919191
    threshold = 0
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
        
        if variance < max_variance:
            max_variance = variance
            threshold = bin_edges[i]
    return threshold
# Print the threshold value
threshold = Otsu_threshold(hist, bin_edges)
print("Otsu's threshold for LOF:", threshold)

# Plot the histogram and the threshold value
plt.hist(LOFs, bins=10)
plt.axvline(x=threshold, color='r', linestyle='--', label="Otsu's threshold")
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('LOF score Histogram with Otsu Threshold')
plt.legend()
plt.show()

In1=[]
out1=[]
for i in LOFs:
    if(i<threshold):
        In1.append(i)
    else:
        out1.append(i)
print("Inliers for LOFs: ",len(In1))
print("Outliers for LOFs: ",len(out1))
