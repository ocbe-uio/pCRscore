import pandas
import numpy
from sklearn import preprocessing
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.svm import SVC

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

def grid_search(X, y, n_cores = 1):
    # TODO: profile (with cProfile?) to possibly add progress bar
    # TODO: allow multithreading?
    # Defining the parameter range for the hyperparameter grid search
    param_grid = {
        'C': numpy.exp(numpy.linspace(-12, 3, num = 50)),
        'gamma': numpy.exp(numpy.linspace(-12, 1, num = 50)),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Define a custom scoring dictionary that includes F1 score and accuracy
    scoring = {
        'F1': make_scorer(f1_score),
        'Accuracy': make_scorer(accuracy_score)
    }

    # Create a StratifiedKFold object with 5 splits for cross-validation
    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3333) # TODO: replace with 1 / 3?

    # Create a GridSearchCV object with the SVC classifier, parameter grid,
    # custom scoring, refit based on F1 score, 10-fold cross-validation, and
    # no verbosity
    grid = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid, scoring = scoring, refit = 'F1', verbose = 0, cv = 10,
        n_jobs = n_cores
    )

    # Fit the model for grid search using the training data
    grid.fit(X_train, y_train)

    # Print the best parameters found during grid search
    print(grid.best_params_)

    # Print the best estimator (model) found during grid search
    print(grid.best_estimator_)
