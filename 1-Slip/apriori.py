from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Load the groceries dataset
data = pd.read_csv('grocery.csv')
encoded_data = pd.get_dummies(data)
frequent_items = apriori(encoded_data, min_support=0.25, use_colnames=True)
print(frequent_items)
 