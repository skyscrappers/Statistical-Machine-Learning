import random
import math
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, K, max_iterations=100):
        self.K = K
        self.max_iterations = max_iterations

    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum([(point1[i] - point2[i])**2 for i in range(len(point1))]))

    def random_centroids(self, data):
        centroids = data[np.random.choice(range(len(data)), self.K), :]
        return centroids

    def assign_clusters(self, data, centroids):
        clusters = [[] for i in range(self.K)]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        return clusters

    def calculate_centroids(self, clusters):
        centroids = []
        for cluster in clusters:
            if cluster:
                centroid = [sum(point[i] for point in cluster) / len(cluster) for i in range(len(cluster[0]))]
                centroids.append(centroid)
        return np.array(centroids)

    def fit(self, data):
        centroids = self.random_centroids(data)
        for i in range(self.max_iterations):
            clusters = self.assign_clusters(data, centroids)
            new_centroids = self.calculate_centroids(clusters)
            if np.array_equal(new_centroids, centroids):
                break
            else:
                centroids = new_centroids
        return clusters, centroids

    def plot(self, data):
        clusters, centroids = self.fit(data)
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cluster in enumerate(clusters):
            color = plt.cm.tab10(i)
            for point in cluster:
                ax.scatter(point[0], point[1], color=color, alpha=0.5)
            ax.scatter(centroids[i][0], centroids[i][1], marker='o', color='k', s=100, linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('K-Means Clustering')
        plt.show()

    import numpy as np

    def distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def silhouette_score(self,X, labels):
        dtype=None
        n_clusters = len(set(labels))
        cluster_centers = np.array([np.mean(X[labels == i], axis=0) for i in range(n_clusters)])
        scores = np.zeros(len(X))

        for i in range(len(X)):
            a_i = np.mean([KMeans.distance(X[i], X[j]) for j in range(len(X)) if labels[j] == labels[i] and i != j])
            b_i = np.min([np.mean([KMeans.distance(X[i], X[j]) for j in range(len(X)) if labels[j] == k]) for k in range(n_clusters) if k != labels[i]])
            scores[i] = (b_i - a_i) / max(b_i, a_i) if max(b_i, a_i) > 0 else 0
        if np.isnan(scores).all():
            return np.nan
        return np.mean(scores, dtype=np.float64)


 
data = np.array(np.load('kmeans_data.npy'))
# # data = np.array([[1,1],[2,2],[4,4],[5,7],[1,3],[5,5]])
scores = []
for t in range(2,11):
#x = int(input("K : "))
    model = KMeans(K=t)
    clusters, centroids = model.fit(data)
    def convert_3d_array_to_list(arr):
        if isinstance(arr, np.ndarray):
            return [convert_3d_array_to_list(subarr) for subarr in arr]
        elif isinstance(arr, list):
            return [convert_3d_array_to_list(subarr) for subarr in arr]
        else:
            return arr
    clusters = convert_3d_array_to_list(clusters)

    def find_labels(cluster, data):
        labels = []
        for i in range(len(data)):
            point = tuple(data[i])
            for j in range(len(cluster)):
                if point in [tuple(x) for x in cluster[j]]:
                    labels.append(j)
                    break
        return labels

    labels = find_labels(clusters, data)
    model.plot(data)
    sa = model.silhouette_score(data, labels)
    print(f'Silhouette score for K = {t} is: ', sa)
    scores.append(sa)
print(scores)
for i in range(2,11):
    if(scores[i-2]==max(scores)):
        print(f"Optimal value of K is {i} and Sillhouette score at K = {i} is {scores[i-2]}")
        break
# Silhouette score is:  0.663492972101041
# Silhouette score is:  0.7234705964942477
# Silhouette score is:  0.7059019291990838
# Silhouette score is:  0.7276465307467833
# Silhouette score is:  0.644085154977051
# Silhouette score is:  0.547295514204624
# Silhouette score is:  0.5767165340598887
# Silhouette score is:  0.38389206191589953
# Silhouette score is:  0.5343218907304323
S = [0.663492972101041, 0.7234705964942477, 0.7059019291990838, 0.7276465307467833, 0.644085154977051, 0.547295514204624, 0.5767165340598887, 0.38389206191589953, 0.5343218907304323]
k = [i for i in range(2,11)]
plt.plot(k,scores)
plt.xlabel("K")
plt.ylabel("Silhouette scores")
plt.show()