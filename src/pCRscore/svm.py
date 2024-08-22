import pandas
import numpy
import shap
from sklearn import preprocessing
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import \
    GridSearchCV, train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification

def preprocess(data, svm_type = "discovery"):
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
    if svm_type == "discovery":
        valid_cohort = [
            'E-MTAB-4439', 'GSE18728', 'GSE19697', 'GSE20194', 'GSE20271',
            'GSE22093', 'GSE22358', 'GSE42822', 'GSE22513'
        ]
    elif svm_type == "validation":
        valid_cohort = [
            'GSE25066', 'GSE32603', 'GSE32646', 'GSE37946', 'GSE50948',
            'GSE23988'
        ]

    data = data[data['Trial'].isin(valid_cohort)]

    return data

def extract_features(data):
    # Extract the features (independent variables) and create a DataFrame 'X'
    # Drop columns 'Trial', 'Mixture', 'Response', and 'Cohort' to get features
    dropped_columns = ['Trial', 'Mixture', 'Response', 'Cohort']
    X = data.drop(dropped_columns, axis = 1)
    d3 = data.drop(dropped_columns, axis = 1)

    # P# Extract the target variable 'y' (dependent variable)
    y = data['Response']

    # Standardize the features using the StandardScaler from sklearn
    # This step scales the features to have mean 0 and standard deviation 1
    # This is important for some machine learning algorithms that
    # are sensitive to feature scales
    X = pandas.DataFrame(
        preprocessing.StandardScaler().fit(X).transform(X),
        index = d3.index, columns = d3.columns
    )

    return X, y

def grid_search(X, y, n_cores = 1, verbose = 0):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 / 3)

    # Create a GridSearchCV object with the SVC classifier, parameter grid,
    # custom scoring, refit based on F1 score, 10-fold cross-validation, and
    # no verbosity
    grid = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid, scoring = scoring, refit = 'F1', cv = 10, n_jobs = n_cores,
        verbose = verbose
    )

    # Fit the model for grid search using the training data
    grid.fit(X_train, y_train)

    return grid

def evaluate_model(X, y, verbose = False):
    # We normally start with the model that has the best performance and
    # fine tune the parameters to find the best model.
    # Here, the following model found to have the best performance
    # based on combined score

    # Create model
    model = fit_svc()

    # It should be noted that SHAP values calculated using these two models are
    # very similar, particularly for features with high correlation to response.

    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    # evaluate model
    Acc_score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-2)
    f1_score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-2)
    roc_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-2)

    # report performance
    if verbose:
        print('Accuracy: %.3f (%.3f)\nf1 score: %.3f (%.3f)\nAUC: %.3f (%.3f)' %
            (numpy.mean(Acc_score) * 100, numpy.std(Acc_score) * 100,
            numpy.mean(f1_score), numpy.std(f1_score),
            numpy.mean(roc_auc), numpy.std(roc_auc))
        )

    return {'Accuracy': Acc_score, 'f1 score': f1_score, 'AUC': roc_auc}

def fit_svc():
    return SVC(
        C = 1, gamma = 0.1, kernel = 'rbf', probability = True,
        class_weight = 'balanced'
    )

def shap_analysis(X, y, pandas_out = False):
    # Create model and fit to discovery data
    clf = fit_svc()
    clf.fit(X, y)

    # creating the explainer using the model and X as the background
    svm_explainer = shap.KernelExplainer(clf.predict, X)

    # calculating SHAP values for X using the explainer
    # For 1000 samples it takes 50 hours on a single core of 8gen intel CPU
    svm_shap_values = svm_explainer.shap_values(X)

    # Convert the SHAP values to a pandas DataFrame
    if pandas_out:
        svm_shap_values = pandas.DataFrame(
            svm_shap_values, columns=X.columns, index=X.index
        )

    return svm_shap_values

def shap_plot(shap_values, X, type = 'dot'):
    shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type=type)
