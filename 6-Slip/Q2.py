import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('employees.csv')
data = data.dropna()  # Preprocess data by removing null values
X = data[['income']]  # Adjust to the appropriate columns

# K-means clustering
kmeans = KMeans(n_clusters=4)
data['Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(data['income'], range(len(data)), c=data['Cluster'])
plt.xlabel('Income')
plt.title('Employee Income Clustering')
plt.show()
