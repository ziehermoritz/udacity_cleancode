import pytest
import os
import logging
import pathlib as pl
import sys
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
# -
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import churn_library_solution as cls

from churn_library import import_data
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models
from churn_library import Directories

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
    Test data import.

    This example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: "
                      "The file doesn't appear to have rows and columns")
        raise err

    pytest.df = df


def test_eda():
    """test perform eda function."""
    df_bank = pytest.df
    df_bank['Churn'] = df_bank['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # setup and initialize all necessary directories
    pytest.dirs = Directories(pl.Path(os.getcwd()))
    try:
        perform_eda(df_bank, pytest.dirs)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        assert df_bank.shape[0] > 0
        assert df_bank.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing test_eda: "
                      "The file doesn't appear to have rows and columns")
        raise err

    pytest.df = df_bank

def test_encoder_helper():
    """test encoder helper."""
    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    shape_0 = pytest.df.shape[0]
    shape_1 = pytest.df.shape[1]
    try:
        df_bank = encoder_helper(pytest.df, cat_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err
    try:
        assert df_bank.shape[0] == shape_0
        assert df_bank.shape[1] > shape_1
        logging.error("Testing encoder_helper: Appending categorical values "
                      "SUCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: "
                      "Appending categorical values failed")
        raise err

    pytest.df = df_bank


def test_perform_feature_engineering():
    """	test perform_feature_engineering."""
    try:
        train_X, test_X, train_y, test_y = perform_feature_engineering(pytest.df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_feature_engineering: The file wasn't found")
        raise err

    try:
        assert len(train_X.shape) == 2
        logging.error("Testing encoder_helper: train_X got right shape.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: train_X got wrong shape.")
        raise err

    try:
        assert len(train_y.shape) == 1
        logging.error("Testing encoder_helper: train_y got right shape.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: train_y got wrong shape.")
        raise err

    try:
        assert len(test_X.shape) == 2
        logging.error("Testing encoder_helper: test_X got right shape.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: test_X got wrong shape.")
        raise err

    try:
        assert len(test_y.shape) == 1
        logging.error("Testing encoder_helper: train_y got right shape.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: train_y got wrong shape.")
        raise err

    pytest.train_X = train_X
    pytest.train_y = train_y
    pytest.test_X = test_X
    pytest.test_y = test_y

def test_train_models():
    """test train_models."""

    try:
        cv_rfc, lrc = train_models(pytest.train_X, pytest.test_X, pytest.train_y, pytest.test_y, pytest.dirs)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file wasn't found")
        raise err

    try:
        isinstance(cv_rfc, GridSearchCV)
        logging.info("Testing train_models: Got trained grid search model.")
    except AssertionError as err:
        logging.error("Testing train_models: ""trained grid search model failed.")
        raise err

    try:
        isinstance(lrc, LogisticRegression)
        logging.info("Testing train_models: ""Got trained logistic regression model.")
    except AssertionError as err:
        logging.error("Testing train_models: ""trained  logistic regression model failed.")
        raise err







