import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Load the dataset
df = pd.read_csv('Real estate.csv')

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare the training data
train_data = train_df.values
train_features = train_data[:, 1:-1]
train_labels = train_data[:, -1]

# Add intercept term to the features
train_X = np.concatenate([np.ones((train_features.shape[0], 1)), train_features], axis=1)

# Calculate the theta using the normal equation
theta = np.dot(np.linalg.inv(np.dot(train_X.T, train_X)), np.dot(train_X.T, train_labels)).reshape(-1, 1)
test_data = test_df.values
test_features = test_data[:, 1:-1]
test_labels = test_data[:, -1]

test_X = np.concatenate([np.ones((test_features.shape[0], 1)), test_features], axis=1)
predicted_output = np.dot(test_X, theta)
# Print the final equation
print("Final equation for Linear regression is:")
print(f"Y = {theta[0][0]:.4f} + ", end="")
for i in range(1, len(theta)):
    if i == len(theta) - 1:
        print(f"x{i}*{theta[i][0]:.4f} ", end='')
    else:
        print(f"x{i}*{theta[i][0]:.4f} + ", end='')
RSS = 0
for i in range(len(predicted_output)):
    RSS+=(test_labels[i]-predicted_output[i])**2
print("\nRSS = ",float(RSS))
RMSE = RSS/len(predicted_output)
print("RMSE: ",float(RMSE**0.5))
TSS=0
y_m = np.mean(predicted_output)
for i in test_labels:
    TSS+=(i-y_m)**2
print("TSS = ",TSS)
R_squared = 1-RSS/TSS
print("R_squared = ",float(R_squared))



# The limitations of the normal equation approach for linear regression are:

# It can be computationally expensive for large datasets.
# It can overfit the data if the input features are highly correlated or if there are too many features relative to the number of examples.
# It may be numerically unstable if the input features are linearly dependent.
# It can only model linear relationships between variables.
# It is sensitive to outliers, which can lead to overfitting and poor generalization to new data.