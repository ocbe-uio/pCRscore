from pCRscore import discovery_svm
import pandas as pd

def test_preprocess():
    data = pd.read_csv("Data NAC cohort _1_.csv")
    data = discovery_svm.preprocess(data)
    assert data.shape == (1009, 46)

    X, y = discovery_svm.extract_features(data)
    assert X.shape == (1009, 42)
