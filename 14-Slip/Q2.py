import pandas as pd

# Create or load a dataset
data = pd.DataFrame({
    'Feature1': [1, 2, None, 4, 5],
    'Feature2': [None, 'B', 'C', 'D', 'E']
})

# Remove rows with null values
data_cleaned = data.dropna()
print(data_cleaned)
