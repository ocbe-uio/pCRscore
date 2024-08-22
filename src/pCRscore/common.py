
import pandas

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
