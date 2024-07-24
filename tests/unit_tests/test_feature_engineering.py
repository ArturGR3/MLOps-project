import pytest
import pandas as pd
import numpy as np
import os 
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
project_path = os.getenv("PROJECT_PATH")
print(f"project path {project_path}")
if project_path not in sys.path:
    sys.path.append(project_path)
    
from modules.feature_engineering import FeatureEnginering

@pytest.fixture
def feature_engineering():
    return FeatureEnginering("test_competition", "target")

@pytest.fixture
def sample_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.randint(0, 100, 1000),
        'target': np.random.rand(1000)
    })
    return data

def test_init(feature_engineering):
    assert feature_engineering.target_column == "target"
    assert "test_competition" in feature_engineering.preprocessed_data
    assert "test_competition" in feature_engineering.feature_eng_data
    assert os.path.exists(feature_engineering.feature_eng_data)

def test_stratified_sample(feature_engineering, sample_data):
    X_train, y_train = feature_engineering.stratified_sample(sample_data)
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    
    assert len(X_train) <= 1000 * 15  # max possible size
    assert len(y_train) == len(X_train)
    assert set(X_train.columns) == set(sample_data.columns) - {'target', 'bins'}

def test_openfe_fit(feature_engineering, sample_data):
    ofe = feature_engineering.openfe_fit(sample_data, number_of_features=2)
    
    assert hasattr(ofe, 'new_features_list')
    assert len(ofe.new_features_list) >= 2

def test_openfe_transform(feature_engineering, sample_data):
    train = sample_data.iloc[:800]
    test = sample_data.iloc[800:]
    
    train_transformed, test_transformed = feature_engineering.openfe_transform(train, test, number_of_features=2)
    
    assert isinstance(train_transformed, pd.DataFrame)
    assert isinstance(test_transformed, pd.DataFrame)
    assert len(train_transformed.columns) > len(train.columns)
    assert len(test_transformed.columns) > len(test.columns)
    assert os.path.exists(os.path.join(feature_engineering.feature_eng_data, "train_transformed.pkl"))
    assert os.path.exists(os.path.join(feature_engineering.feature_eng_data, "test_transformed.pkl"))