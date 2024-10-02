from pCRscore import pipeline
import pandas
import math

# Import DiscoveryData.csv and keep columns 4 to 39
data = pandas.read_csv(".meta/DiscoveryData.csv")
# TODO: incorporate filtering in pipeline?
shap = pandas.read_csv(".meta/DiscoverySHAP.csv").iloc[:, 1:37]

# Normalize data
data_norm = pipeline.normalize_data(data.copy(), range(3, 39))


# Test
def test_normalize_data():
    assert data_norm.shape == (data.shape[0], data.shape[1] - 11)
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
