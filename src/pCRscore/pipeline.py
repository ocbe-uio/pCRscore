import pandas as pd

# Function to normalize cell fractions and remove outliers
def normalize_data(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    for i in range(data.shape[1]):
        data.iloc[:, i] /= data.iloc[:, i].quantile(0.99)
    return data
