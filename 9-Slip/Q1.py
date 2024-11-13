import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

# Load dataset
data = pd.read_csv('boston_houses.csv')
X = data[['RM']]  # Number of rooms
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict([[5]])  # Predict for 5 rooms
print("Ridge Prediction for 5 rooms:", ridge_predictions)

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict([[5]])  # Predict for 5 rooms
print("Lasso Prediction for 5 rooms:", lasso_predictions)
