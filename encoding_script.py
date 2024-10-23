import pandas as pd
import numpy as np

# Function for Ordinal Encoding
def ordinal_encode(df, column):
    
    unique_values = df[column].unique()
    ordinal_map = {val: idx for idx, val in enumerate(unique_values)}
    df[column] = df[column].map(ordinal_map)
    return df, ordinal_map

# Function for One-Hot Encoding
def one_hot_encode(df, column):
    
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df.drop(column, axis=1), one_hot], axis=1)
    return df

# Sample Data
data = {'City': ['New York', 'Paris', 'Tokyo', 'Paris', 'New York', 'Tokyo'],
        'Age': [24, 28, 22, 32, 35, 29]}

# Create DataFrame
df = pd.DataFrame(data)

# Apply Ordinal Encoding
print("Original DataFrame:")
print(df)

df_ordinal, ordinal_map = ordinal_encode(df.copy(), 'City')
print("\nOrdinal Encoded DataFrame:")
print(df_ordinal)
print("Ordinal Map:", ordinal_map)

# Apply One-Hot Encoding
df_one_hot = one_hot_encode(df.copy(), 'City')
print("\nOne-Hot Encoded DataFrame:")
print(df_one_hot)
