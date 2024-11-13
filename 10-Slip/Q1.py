import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('iris.csv')
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.show()
