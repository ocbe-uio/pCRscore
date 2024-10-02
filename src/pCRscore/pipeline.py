import pandas


# Function to normalize cell fractions and remove outliers
def normalize_data(data, cols=None):
    # TODO: drop columns not corresponding to numbers?
    if cols is not None:
        data = data.iloc[:, cols]
    for i in range(data.shape[1]):
        data.iloc[:, i] /= data.iloc[:, i].quantile(0.99)
        # Convert values above 1 to NA
        data.iloc[:, i] = data.iloc[:, i].apply(lambda x: x if x <= 1 else None)
    return data


# Function to combine normalized data and SHAP values
def combine_fractions_shap(data_norm, shap):
    data_melted = pandas.melt(data_norm)
    shap_melted = pandas.melt(shap)
    all_pat_1 = pandas.concat([data_melted, shap_melted['value']], axis=1)
    all_pat_1.columns = ['Feature', 'Fraction', 'SHAP value']
    all_pat_1_clean = all_pat_1.dropna()
    return all_pat_1_clean
