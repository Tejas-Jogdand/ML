import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = pd.read_csv('news_data.csv')
X = data['text']  # News text
y = data['category']  # News categories

# Vectorize text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=0)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
