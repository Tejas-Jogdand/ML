import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('iris.csv')
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Scatter Plot
plt.scatter(data['sepal_length'], data['petal_length'], c=data['species'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatter Plot of Iris Dataset with Numeric Species')
plt.show()
