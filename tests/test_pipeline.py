from pCRscore import pipeline
import pytest
import pandas
import math

# Import DiscoveryData.csv and keep columns 4 to 39
data = pandas.read_csv(".meta/DiscoveryData.csv")
data = data.iloc[:, 3:38]

# Normalize data
data_norm = pipeline.normalize_data(data.copy())

# Test
def test_normalize_data():
  assert data_norm.shape == data.shape
  tol = 1e-3
  assert math.isclose(data['B.cells.Memory'].iloc[0], 0.01789, rel_tol=tol)
  assert math.isclose(data_norm['B.cells.Memory'].iloc[0], 0.34186, rel_tol=tol)
