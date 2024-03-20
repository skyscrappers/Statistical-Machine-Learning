#Define Data
import math
age = [25,35,45,20,35,52,23,40,60,48,33]
loan = [40,60,80,20,120,18,95,62,100,220,150]
HPI = [135,256,231,267,139,150,127,216,139,250,264]
BHK = [2,3,3,4,4,2,2,4,2,3,4]
def euclidean_distance(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
pair = []
for i in range(len(age)):
    pair.append([[age[i],loan[i]],[HPI[i], BHK[i]]])
new_age = 37
new_loan = 142
k = int(input())
distances = []
for i in pair:
    x = euclidean_distance(new_age,i[0][0],new_loan,i[0][1])
    distances.append(x)
print(distances)
distances.sort()
closest = []
for i in range(len(pair)):
    if euclidean_distance(pair[i][0][0],new_age,pair[i][0][1],new_loan) in distances[:k]:
        closest.append(pair[i])
print(closest)
# import numpy as np

# # Define the training data
# age = [25,35,45,20,35,52,23,40,60,48,33]
# loan = [40,60,80,20,120,18,95,62,100,220,150]
# HPI = [135,256,231,267,139,150,127,216,139,250,264]
# BHK = [2,3,3,4,4,2,2,4,2,3,4]

# # Define the test instance
# test_instance = [37, 142]

# # Combine the features into a single numpy array
# X = np.column_stack((age, loan))

# # Define the labels for HPI and BHK
# y_hpi = np.array(HPI)
# y_bhk = np.array(BHK)

# # Define the value of k for kNN
# k = 4

# # Calculate the Euclidean distance between the test instance and each training instance
# distances = np.sqrt(np.sum((X - test_instance)**2, axis=1))

# # Get the indices of the k nearest neighbors
# nearest_indices = np.argsort(distances)[:k]

# # Use the labels of the k nearest neighbors to make a prediction
# prediction_hpi = np.mean(y_hpi[nearest_indices])
# prediction_bhk = np.round(np.mean(y_bhk[nearest_indices]))

# # Print the predictions for HPI and BHK
# print(f"For k={k}, the predicted HPI is {prediction_hpi:.2f} and the predicted BHK is {prediction_bhk}")
