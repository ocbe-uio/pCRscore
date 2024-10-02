from pCRscore import pipeline
import pandas
import math

# Import DiscoveryData.csv and drop invalid columns
# FIXME: replace local data with mock data (after finishing the pipeline)
data_raw = pandas.read_csv(".meta/DiscoveryData.csv")
shap_raw = pandas.read_csv(".meta/DiscoverySHAP.csv")
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
    tol = 1e-3
    assert math.isclose(data['B.cells.Memory'].iloc[0], 0.01789, rel_tol=tol)
    assert math.isclose(
        data_norm['B.cells.Memory'].iloc[0], 0.34186, rel_tol=tol
    )
    assert math.isclose(shap['Endothelials'].iloc[0], 0.037792, rel_tol=tol)


# Combine data and SHAP values
data_shap = pipeline.combine_fractions_shap(data_norm, shap)


def test_combine_fractions_shape():
    assert data_shap.shape == (35928, 3)
