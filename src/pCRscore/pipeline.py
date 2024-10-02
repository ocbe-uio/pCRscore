import pandas


# Function to drop non-float columns
def drop_non_float(data, extra_cols=None):
    data = data.select_dtypes(include='float')
    if extra_cols is not None:
        data = data.drop(data.columns[extra_cols], axis=1)
    return data


# Function to normalize cell fractions and remove outliers
def normalize_data(data):
    for i in range(data.shape[1]):
        data.iloc[:, i] /= data.iloc[:, i].quantile(0.99)
        # Convert values above 1 to NA
        data.iloc[:, i] = data.iloc[:, i].where(data.iloc[:, i] <= 1, None)
    return data


# Function to combine normalized data and SHAP values
def combine_fractions_shap(data_norm, shap):
    data_melted = pandas.melt(data_norm)
    shap_melted = pandas.melt(shap)
    all_pat_1 = pandas.concat([data_melted, shap_melted['value']], axis=1)
    all_pat_1.columns = ['Feature', 'Fraction', 'SHAP value']
    all_pat_1_clean = all_pat_1.dropna()
    return all_pat_1_clean
