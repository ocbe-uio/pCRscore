import pandas
from sklearn import preprocessing

# TODO: add code from
# https://github.com/YounessAzimzade/XML-TME-NAC-BC/blob/main/Discovery%20SVM.ipynb

def preprocess(data):
    # Mapping the values in the 'Response' column to binary values 0 and 1
    resp = {'pCR': 1, 'RD': 0}
    data.Response = [resp[item] for item in data.Response]

    # Mapping the values in the 'ER' column to binary values 0 and 1
    er = {'Positive': 1, 'Negative': 0}
    data.ER = [er[item] for item in data.ER]

    # Creating dummy variables for the categorical column 'PAM50'
    categorical_cols = ['PAM50']
    data = pandas.get_dummies(data, columns=categorical_cols)

    # Selecting validation cohort data
    valid_cohort = [
        'E-MTAB-4439', 'GSE18728', 'GSE19697', 'GSE20194', 'GSE20271',
        'GSE22093', 'GSE22358', 'GSE42822', 'GSE22513'
    ]
    data = data[data['Trial'].isin(valid_cohort)]

    return data

def extract_features(data):
    # Extract the features (independent variables) and create a DataFrame 'X'
    # Drop columns 'Trial', 'Mixture', 'Response', and 'Cohort' to get features
    dropped_columns = ['Trial', 'Mixture', 'Response', 'Cohort']
    X = pandas.DataFrame(data.drop(dropped_columns, axis = 1))
    d3 = pandas.DataFrame(data.drop(dropped_columns, axis = 1))

    # P# Extract the target variable 'y' (dependent variable)
    y = data['Response']

    # Standardize the features using the StandardScaler from sklearn
    # This step scales the features to have mean 0 and standard deviation 1
    # This is important for some machine learning algorithms that
    # are sensitive to feature scales
    X = pandas.DataFrame(
        preprocessing.StandardScaler().fit(X).transform(X),
        columns = d3.columns
    )

    return X, y
