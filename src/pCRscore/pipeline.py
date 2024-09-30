import pandas

# Function to normalize cell fractions and remove outliers
def normalize_data(data, cols=None):
    if cols is not None:
        data = data.iloc[:, cols]
    data = data.apply(pandas.to_numeric, errors='coerce')
    for i in range(data.shape[1]):
        data.iloc[:, i] /= data.iloc[:, i].quantile(0.999)
    return data
