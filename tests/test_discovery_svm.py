from pCRscore import discovery_svm
import pandas as pd
from unittest import mock
import pytest
import numpy as np

@pytest.fixture
def mock_data():
  # Define the column names
  columns = [
    "Trial", "Mixture", "B.cells.Memory", "B.cells.Naive", "CAFs.MSC.iCAF.like",
    "CAFs.myCAF.like", "DCs", "Endothelial.ACKR1", "Endothelial.CXCL12",
    "Endothelial.LYVE1", "Endothelial.RGS5", "GenMod1", "GenMod2", "GenMod3",
    "GenMod4", "GenMod5", "GenMod6", "GenMod7", "Luminal.Progenitors",
    "Macrophage", "Mature.Luminal", "Monocyte", "Myoepithelial", "NK.cells",
    "NKT.cells", "Plasmablasts", "PVL.Differentiated", "PVL.Immature",
    "T.cells.CD4.", "T.cells.CD8.", "Cancer.Cells", "Normal.Epi", "TCells",
    "Myeloids", "BCells", "CAFs", "PVLs", "Endothelials", "ER", "Response",
    "Cohort", "PAM50", "PAM50_Normal", "PAM50_LumA", "PAM50_Her2", "PAM50_LumB",
    "PAM50_Basal"
  ]

  # Number of rows you want in your DataFrame
  num_rows = 100  # Adjust this based on how many rows you need

  # Create a dataframe with random values in (0, 1) for the numerical columns
  n_numerical_columns = 7
  df = pd.DataFrame(
    np.random.rand(num_rows, len(columns) - n_numerical_columns),
    columns = columns[:-n_numerical_columns]
  )

  # Adding the non-numerical columns manually
  df['Trial'] = 'E-MTAB-4439'
  df['Mixture'] = 'Mixture1'
  df['Cohort'] = 'Discovery'
  df['Response'] = np.random.choice(['pCR', 'RD'], num_rows)
  df['ER'] = np.random.choice(['Positive', 'Negative'], num_rows)
  df['PAM50'] = np.random.choice(['LumA', 'Basal'], num_rows)
  df['PAM50_Normal'] = np.random.choice([True, False], num_rows)
  df['PAM50_LumA'] = np.random.choice([True, False], num_rows)
  df['PAM50_Her2'] = np.random.choice([True, False], num_rows)
  df['PAM50_LumB'] = np.random.choice([True, False], num_rows)
  df['PAM50_Basal'] = np.random.choice([True, False], num_rows)

  return df

@mock.patch('pandas.read_csv')
def test_preprocess(mock_read_csv, mock_data):
   # Configure the mock to return your predefined DataFrame
    mock_read_csv.return_value = mock_data

    data = pd.read_csv("Data NAC cohort _1_.csv") # returns mock data instead
    data = discovery_svm.preprocess(data)
    assert data.shape == (100, 48)

    X, y = discovery_svm.extract_features(data)
    assert X.shape == (100, 44)

@pytest.mark.slow
def test_grid_search():
    X = pd.DataFrame(np.random.randn(100, 44))
    y = np.random.choice([0, 1], 100)
    grid = discovery_svm.grid_search(X, y)
    assert isinstance(grid, discovery_svm.GridSearchCV)
    assert hasattr(grid, 'best_params_')
    assert hasattr(grid, 'best_score_')

def test_evaluate_model():
    X = np.random.randn(100, 44)
    y = np.random.choice([0, 1], 100)
    stats = discovery_svm.evaluate_model(X, y)
    assert isinstance(stats, dict)
    assert len(stats) == 3
    for i in stats:
        assert len(stats[i]) == 5

def test_shapley():
    X = pd.DataFrame(np.random.randn(100, 44))
    y = np.random.choice([0, 1], 100)
    shapl = discovery_svm.shap_analysis(X, y)
    assert isinstance(shapl, np.ndarray)
    assert shapl.shape == (100, 44)
    discovery_svm.shap_plot(shapl, X)
