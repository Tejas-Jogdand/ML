import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('salary_pos.csv')
X = data[['level']]
y = data['salary']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict salaries for level 11 and 12
prediction = model.predict([[11], [12]])
print("Predicted Salaries for Level 11 and 12:", prediction)
 