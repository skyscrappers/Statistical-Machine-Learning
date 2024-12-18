from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('train.csv')
train_data = np.array(train_data)
train_data = train_data[:, 1:]
labels = train_data.T[-1]
encoder = LabelEncoder()

# fit the encoder to the list of strings
encoder.fit(labels)

# transform the list of strings into encoded values
encoded_labels = encoder.transform(labels)

train_data = train_data[:,:-1]
rawData = train_data
rawData = np.hstack((rawData, encoded_labels.reshape(-1, 1)))
rawData = pd.DataFrame(rawData)
# print(rawData)

test_data = pd.read_csv('test.csv')
test_data = np.array(test_data)
ids = test_data.T[0]
ids = ids.astype(int)
print(type(ids[0]))
# print("Ruko jara sabar karo")
# print(ids)
test_data = test_data[:, 1:]
#Apply lof to remove outliers
lof = LocalOutlierFactor(n_neighbors=2)
outliers = lof.fit_predict(rawData)
id = 0
for out in outliers:
    if out == -1:
        rawData.drop(id,axis=0,inplace=True) 
    id += 1
encoded_labels = np.array(rawData, dtype=int).T[-1]
train_data = np.array(rawData)[:,:-1]
# print(train_data)


kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data)
kmeans_labels = kmeans.predict(train_data)
kmeans.fit(test_data)
kmeans_labels1 = kmeans.predict(test_data)

kmeans1 = KMeans(n_clusters=10)
kmeans1.fit(train_data)
kmeans_labels2 = kmeans1.predict(train_data)
kmeans1.fit(test_data)
kmeans_labels22 = kmeans1.predict(test_data)

kmeans3 = KMeans(n_clusters=15)
kmeans3.fit(train_data)
kmeans_labels3 = kmeans3.predict(train_data)
kmeans3.fit(test_data)
kmeans_labels33 = kmeans3.predict(test_data)

test_data = np.column_stack((test_data, kmeans_labels1, kmeans_labels22, kmeans_labels33))
train_data = np.column_stack((train_data, kmeans_labels, kmeans_labels2, kmeans_labels3))

pca = PCA(n_components=480)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)


lda = LDA(n_components=19)
train_data = lda.fit_transform(train_data, encoded_labels)
test_data = lda.transform(test_data)

lr = LogisticRegression(max_iter=40000,C=1)

lr.fit(train_data, encoded_labels)
y_pred = lr.predict(test_data)

decoded_label = encoder.inverse_transform(y_pred)

matrix = np.column_stack((ids, decoded_label))
df = pd.DataFrame(data=matrix, columns=['id', 'Category'])

# define the filename for the CSV file
filename = 'Final_Submission.csv'

# save the dataframe as a CSV file
df.to_csv(filename, index=False)
