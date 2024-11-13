import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('boston_housing.csv')
X = data[['RM']]  # Number of rooms
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)
print(predictions)
