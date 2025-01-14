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
    disc_data_raw = pandas.read_csv(".meta/DiscoveryData.csv")
    disc_shap_raw = pandas.read_csv(".meta/DiscoverySHAP.csv")
    valid_data_raw = pandas.read_csv(".meta/ValidationData.csv")
    valid_shap_raw = pandas.read_csv(".meta/ValidationSHAP.csv")
else:
    # On GitHub Actions
    local = False
    disc_data_raw = mock_data()
    disc_shap_raw = mock_data(shap=True)


disc_data = pipeline.drop_non_float(disc_data_raw)
disc_shap = pipeline.drop_non_float(disc_shap_raw, extra_cols=range(36, 42))
valid_data = pipeline.drop_non_float(valid_data_raw)
valid_shap = pipeline.drop_non_float(valid_shap_raw, extra_cols=range(36, 42))


def test_drop_non_float_and_unnamed():
    assert disc_data.shape == disc_data_raw.iloc[:, 3:39].shape
    assert disc_shap.shape == disc_shap_raw.iloc[:, 1:37].shape


# Normalize data
disc_data_norm = pipeline.normalize_data(disc_data.copy())
valid_data_norm = pipeline.normalize_data(valid_data.copy())


# Test
def test_normalize_data():
    assert disc_data_norm.shape == disc_data.shape
    if local:
        tol = 1e-3
        assert math.isclose(
            disc_data['B.cells.Memory'].iloc[0], 0.01789, rel_tol=tol
        )
        assert math.isclose(
            disc_data_norm['B.cells.Memory'].iloc[0], 0.34186, rel_tol=tol
        )
        assert math.isclose(
            disc_shap['Endothelials'].iloc[0], 0.037792, rel_tol=tol
        )


# Combine data and SHAP values
disc_data_shap = pipeline.combine_fractions_shap(disc_data_norm, disc_shap)
valid_data_shap = pipeline.combine_fractions_shap(valid_data_norm, valid_shap)


def test_combine_fractions_shape():
    assert disc_data_shap.shape == (35928, 3)


# Fit lines for Discovery and Validation cohorts
fit_disc = pipeline.fit_line(disc_data_shap)
fit_valid = pipeline.fit_line(valid_data_shap)


def test_fit_line():
    assert fit_disc.shape[0] == len(disc_data_shap['Feature'].unique())
    assert 'Feature' in fit_disc.columns
    assert 'Coef' in fit_disc.columns
    assert 'CI' in fit_disc.columns
    if local:
        line = fit_disc['Feature'] == 'B.cells.Memory'
        assert math.isclose(
            fit_disc.loc[line, 'Coef'].values[0], -0.0198, rel_tol=1e-2
        )
        assert math.isclose(
            fit_disc.loc[line, 'CI'].values[0], 0.0000193, rel_tol=1e-2
        )

all_pat = pipeline.combine_discovery_validation(
    disc_data_shap, valid_data_shap, fit_disc, fit_valid
)
def test_combine_discovery_validation():
    assert 'Feature' in all_pat.columns
    assert 'Fraction' in all_pat.columns
    assert 'SHAP value' in all_pat.columns
    if local:
        all_pat.shape == (15888, 3)
