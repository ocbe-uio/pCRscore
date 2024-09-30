import pandas

# Function to normalize cell fractions and remove outliers
def normalize_data(data, cols=None):
    # TODO: drop columns not corresponding to numbers?
    if cols is not None:
        data = data.iloc[:, cols]
    for i in range(data.shape[1]):
        data.iloc[:, i] /= data.iloc[:, i].quantile(0.999)
    return data
