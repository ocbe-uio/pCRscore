from pCRscore import pipeline
import pandas
import math
import os
import numpy


def mock_data(shap=False):
    columns = [
        'Unnamed: 0', 'Trial', 'Mixture', 'B.cells.Memory',
        'B.cells.Naive', 'CAFs.MSC.iCAF.like', 'CAFs.myCAF.like', 'DCs',
        'Endothelial.ACKR1', 'Endothelial.CXCL12', 'Endothelial.LYVE1',
        'Endothelial.RGS5', 'GenMod1', 'GenMod2', 'GenMod3', 'GenMod4',
        'GenMod5', 'GenMod6', 'GenMod7', 'Luminal.Progenitors',
        'Macrophage', 'Mature.Luminal', 'Monocyte', 'Myoepithelial',
        'NK.cells', 'NKT.cells', 'Plasmablasts', 'PVL.Differentiated',
        'PVL.Immature', 'T.cells.CD4.', 'T.cells.CD8.', 'Cancer.Cells',
        'Normal.Epi', 'TCells', 'Myeloids', 'BCells', 'CAFs', 'PVLs',
        'Endothelials', 'ER', 'Response', 'Cohort', 'PAM50_Basal',
        'PAM50_Her2', 'PAM50_LumA', 'PAM50_LumB', 'PAM50_Normal'
    ]
    num_rows = 1009
    if shap:
        n_floats = 44
    else:
        n_floats = 36
    df = pandas.DataFrame(columns=columns, index=numpy.arange(num_rows))
    df['Unnamed: 0'] = 0
    df['Trial'] = 'GSE22093'
    df['Mixture'] = 'GSM549230'
    for i in range(n_floats):
        df[columns[i + 3]] = numpy.random.uniform(0, 1, num_rows)
    if not shap:
        df['ER'] = numpy.random.choice([0, 1], num_rows)
    df['Response'] = numpy.random.choice([0, 1], num_rows)
    df['Cohort'] = 'Discovery'
    if not shap:
        df['PAM50_Basal'] = numpy.random.choice([True, False], num_rows)
        df['PAM50_Her2'] = numpy.random.choice([True, False], num_rows)
        df['PAM50_LumA'] = numpy.random.choice([True, False], num_rows)
        df['PAM50_LumB'] = numpy.random.choice([True, False], num_rows)
        df['PAM50_Normal'] = numpy.random.choice([True, False], num_rows)
    # Dropping columns not present in SHAP data
    if shap:
        df = df.drop(columns=['Trial', 'Mixture', 'Response', 'Cohort'])
    return df


# Import DiscoveryData.csv and drop invalid columns
if os.path.exists(".meta"):
    # Running locally
    local = True
    data_raw = pandas.read_csv(".meta/DiscoveryData.csv")
    shap_raw = pandas.read_csv(".meta/DiscoverySHAP.csv")
else:
    # On GitHub Actions
    local = False
    data_raw = mock_data()
    shap_raw = mock_data(shap=True)


data = pipeline.drop_non_float(data_raw)
shap = pipeline.drop_non_float(shap_raw, extra_cols=range(36, 42))


def test_drop_non_float_and_unnamed():
    assert data.shape == data_raw.iloc[:, 3:39].shape
    assert shap.shape == shap_raw.iloc[:, 1:37].shape


# Normalize data
data_norm = pipeline.normalize_data(data.copy())


# Test
def test_normalize_data():
    assert data_norm.shape == data.shape
    if local:
        tol = 1e-3
        assert math.isclose(
            data['B.cells.Memory'].iloc[0], 0.01789, rel_tol=tol
        )
        assert math.isclose(
            data_norm['B.cells.Memory'].iloc[0], 0.34186, rel_tol=tol
        )
        assert math.isclose(
            shap['Endothelials'].iloc[0], 0.037792, rel_tol=tol
        )


# Combine data and SHAP values
data_shap = pipeline.combine_fractions_shap(data_norm, shap)


def test_combine_fractions_shape():
    assert data_shap.shape == (35928, 3)

# Fit lines for Discovery and Validation cohorts
fit_discovery = pipeline.fit_line(data_shap)


def test_fit_line():
    assert fit_discovery.shape[0] == len(data_shap['Feature'].unique())
    assert 'Feature' in fit_discovery.columns
    assert 'Coef' in fit_discovery.columns
    assert 'CI' in fit_discovery.columns
    if local:
        line = fit_discovery['Feature'] == 'B.cells.Memory'
        assert math.isclose(
            fit_discovery.loc[line, 'Coef'].values[0], -0.0198, rel_tol=1e-2
        )
        assert math.isclose(
            fit_discovery.loc[line, 'CI'].values[0], 0.0000193, rel_tol=1e-2
        )
