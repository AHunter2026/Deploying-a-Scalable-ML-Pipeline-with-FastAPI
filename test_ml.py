import pytest
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data 

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test that the train_model function returns a trained RandomForestClassifier
    """
    #Creating training data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    #train model
    model = train_model(X_train, y_train)

    #check model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)

    #check model has been fitted
    assert hasattr(model, 'classes_')


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns valid metrics between 0 and 1
    """
    #create test data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])

    #compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    #check metrics are between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    Test that inference returns predictions of the correct shape and type
    """
    #create and train model
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    model = train_model(X_train, y_train)

    #create test data
    X_test = np.random.rand(20, 10)

    #run inference
    preds = inference(model, X_test)

    #check predictions have right shape
    assert len(preds) == len(X_test)

    #check predictions are binary
    assert set(preds).issubset({0, 1})
