import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('UniversalBank.csv')
X = data.drop(columns=['Personal Loan'])  # Features
y = data['Personal Loan']  # Target: Loan Approval (1/0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Linear SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
 