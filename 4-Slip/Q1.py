import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('mall_customers.csv')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Plot results
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=data['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-means Clustering of Mall Customers')
plt.show()
