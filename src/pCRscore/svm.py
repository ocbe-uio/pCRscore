import pandas
import numpy
import shap
from sklearn import preprocessing
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import \
    GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from .misc import _binary_encode


def preprocess(data, split_var='Cohort', bin_vars='auto', cat_vars='auto'):
    # Mapping the values in the 'Response' column to binary values 0 and 1
    resp = {'pCR': 1, 'RD': 0}
    data.Response = [resp[item] for item in data.Response]

    # Recode variables listed under bin_vars to {-1, 1}
    if bin_vars == 'auto':
        # Sweep all columns. If coded as {Neg*, Pos*}, recode to {-1, 1}
        for col in data.columns:
            if len(data[col].unique()) == 2 and \
                pandas.api.types.is_string_dtype(data[col]) and \
                    set(data[col].str[:3].unique()) in [{'Neg', 'Pos'}]:
                data = _binary_encode(data, col, out_values=[-1, 1])
    else:
        for col in bin_vars:
            data = _binary_encode(data, col, out_values=[-1, 1])

    # Creating dummy variables for the categorical variables
    if cat_vars == 'auto':
        # replace cat_vars with vars that have between 5 and 4 unique values
        cat_vars = data.select_dtypes('object').columns
        for col in cat_vars:
            if len(data[col].unique()) > 5 or len(data[col].unique()) < 3:
                cat_vars = cat_vars.drop(col)
        # Remove the 'Response' column from cat_vars
        cat_vars = [col for col in cat_vars if col != 'Response']
        # If no categorical variables are found, set cat_vars to None
        if len(cat_vars) > 0:
            data = pandas.get_dummies(data, columns=cat_vars)
    else:
        data = pandas.get_dummies(data, columns=cat_vars)

    # If split_var is None, randomly split the data
    if split_var is None:
        data_disc, data_valid = train_test_split(data, test_size=0.5)
        return data_disc, data_valid

    # Split data into discovery and validation cohorts based on split_var
    data_disc = data[data[split_var] == 'Discovery']
    data_valid = data[data[split_var] == 'Validation']

    return data_disc, data_valid


def extract_features(data, y_name='Response'):
    # Extract the target variable 'y' (dependent variable)
    y = data[y_name]

    # Extract the features (independent variables) and create a DataFrame 'X'
    # Drop columns that are not numerical
    X = data.select_dtypes(exclude='object')
    X = X.drop(columns=[y_name])

    # Standardize the features using the StandardScaler from sklearn
    # This step scales the features to have mean 0 and standard deviation 1
    # This is important for some machine learning algorithms that
    # are sensitive to feature scales
    X = pandas.DataFrame(
        preprocessing.StandardScaler().fit(X).transform(X),
        index=X.index, columns=X.columns
    )

    return X, y


def grid_search(X, y, n_cores=-2, verbose=0):
    # Defining the parameter range for the hyperparameter grid search
    param_grid = {
        'C': numpy.exp(numpy.linspace(-12, 3, num=50)),
        'gamma': numpy.exp(numpy.linspace(-12, 1, num=50)),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Define a custom scoring dictionary that includes F1 score and accuracy
    scoring = {
        'F1': make_scorer(f1_score),
        'Accuracy': make_scorer(accuracy_score)
    }

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3)

    # Create a GridSearchCV object with the SVC classifier, parameter grid,
    # custom scoring, refit based on F1 score, 10-fold cross-validation, and
    # no verbosity
    grid = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid, scoring=scoring, refit='F1', cv=10, n_jobs=n_cores,
        verbose=verbose
    )

    # Fit the model for grid search using the training data
    grid.fit(X_train, y_train)

    return grid


def evaluate_model(X, y, verbose=False):
    # We normally start with the model that has the best performance and
    # fine tune the parameters to find the best model.
    # Here, the following model found to have the best performance
    # based on combined score

    # Create model
    model = fit_svc()

    # It should be noted that SHAP values calculated using these two models are
    # very similar, particularly for features with high correlation to response

    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    # evaluate model
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-2)
    f1_score = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-2)
    roc_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-2)

    # report performance
    if verbose:
        print(
            'Accuracy: %.3f (%.3f)\nf1 score: %.3f (%.3f)\nAUC: %.3f (%.3f)' %
            (numpy.mean(acc) * 100, numpy.std(acc) * 100,
             numpy.mean(f1_score), numpy.std(f1_score),
             numpy.mean(roc_auc), numpy.std(roc_auc))
        )

    return {'Accuracy': acc, 'f1 score': f1_score, 'AUC': roc_auc}


def fit_svc():
    return SVC(
        C=1, gamma=0.1, kernel='rbf', probability=True,
        class_weight='balanced'
    )


def shap_analysis(X, y, nsamples='auto', l1_reg='auto', pandas_out=False):
    # Create model and fit to discovery data
    clf = fit_svc()
    clf.fit(X, y)

    # creating the explainer using the model and X as the background
    svm_explainer = shap.KernelExplainer(clf.predict, X)

    # calculating SHAP values for X using the explainer
    # For 1000 samples it takes 50 hours on a single core of 8gen intel CPU
    svm_shap_values = svm_explainer.shap_values(
        X, nsamples=nsamples, l1_reg=l1_reg
    )

    # Convert the SHAP values to a pandas DataFrame
    if pandas_out:
        svm_shap_values = pandas.DataFrame(
            svm_shap_values, columns=X.columns, index=X.index
        )

    return svm_shap_values


def shap_plot(shap_values, X, type='dot'):
    shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type=type)
