import pandas
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn


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


# Function to fit a line to SHAP vs Fraction for each cell type
def fit_line(data, split_ci=False):
    result = []
    grouped = data.groupby('Feature')
    for name, group in grouped:
        X = statsmodels.api.add_constant(group['Fraction'])
        y = group['SHAP value']
        model = statsmodels.api.OLS(y, X).fit()
        coef = model.params['Fraction']
        ci = model.conf_int(alpha=0.001).loc['Fraction']
        if split_ci:
            result.append(
                {'Feature': name, 'Coef': coef, 'LI': ci[0], 'HI': ci[1]}
            )
        else:
            result.append(
               {'Feature': name, 'Coef': coef, 'CI': ci[0] * ci[1]}
            )
    return pandas.DataFrame(result)


def combine_discovery_validation(data_disc, data_valid, fit_disc, fit_valid):
    # Combine fit results
    fit_combined = pandas.merge(
        fit_valid, fit_disc, on="Feature", suffixes=("_Valid", "_Discv")
    )
    # Select cell types that show a clear association in both cohorts
    fit_combined = fit_combined[
        (fit_combined['Coef_Discv'] * fit_combined['Coef_Valid'] > 0) &
        (fit_combined['CI_Valid'] > 0) &
        (fit_combined['CI_Discv'] > 0)
    ]
    # All data put together
    all_pat = pandas.concat([data_disc, data_valid], axis=0)

    # Select only the cell types that pass the validation
    all_pat = all_pat[all_pat['Feature'].isin(fit_combined['Feature'])]
    return all_pat


def _prep_plot_fit(fit):
    # Preparations to plot Fig2
    fit['Sign'] = fit['Coef'].apply(lambda x: 1 if x > 0 else -1)

    fit = fit.sort_values(by='Coef', ascending=False)

    fit['Feature'] = fit['Feature'].astype(str)
    fit['Feature'] = pandas.Categorical(
        fit['Feature'], categories=fit['Feature'], ordered=True
    )
    fit.columns = ["Cell Type", "Coef", "LI", "HI", "Sign"]

    # Remove major cell types. By keeping them, we get suppl Fig 13 b.
    major_cell_types = [
        "CAFs", "PVLs", "TCells", "Normal.Epi", "BCells", "Endothelials",
        "Myeloids", "Cancer.Cells"
    ]
    fit = fit[~fit['Cell Type'].isin(major_cell_types)]

    fit['pCR Score'] = fit['Coef'] / fit['Coef'].quantile(0.95)
    fit['pCR Score'] = fit['pCR Score'].clip(-1, 1)

    # Update the CI scales
    fit['LI'] = fit['LI'] * (fit['pCR Score'] / fit['Coef'])
    fit['HI'] = fit['HI'] * (fit['pCR Score'] / fit['Coef'])

    return fit


def plot_fit(fit):
    fit = _prep_plot_fit(fit)
    plt.figure(figsize=(12, 8))
    seaborn.barplot(
        x='Cell Type', y='pCR Score', hue='Sign', data=fit, dodge=False,
        palette=['yellow', 'purple']
    )
    plt.errorbar(
        x=range(len(fit)), y=fit['pCR Score'],
        yerr=[fit['pCR Score'] - fit['LI'], fit['HI'] - fit['pCR Score']],
        fmt='none', c='black', capsize=5
    )
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Cell Type')
    plt.ylabel('pCR Score', fontsize=14)
    plt.annotate('pCR', xy=(-0.5, 1.15), fontsize=14, color='black')
    plt.annotate('RD', xy=(-0.5, -1.13), fontsize=14, color='black')
    plt.legend().set_visible(False)
    plt.show()
