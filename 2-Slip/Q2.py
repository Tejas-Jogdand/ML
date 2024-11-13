import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Load dataset
data = pd.read_csv('wholesale_customers.csv')
clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering.fit_predict(data[['annual_spending']])  # Adjust columns as necessary
data['Cluster'] = labels
print(data[['annual_spending', 'Cluster']])
