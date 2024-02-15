#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:31:48 2024

author: moritz

pytest - script for churn_library.py

commandline possibilities:

    1.) python test_churn_library.py
    2.) pytest test_churn_library.py
"""
import os
import pathlib as pl
import logging
import pytest
# -
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from churn_library import import_data
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models
from churn_library import Directories

# configure logger with baseline logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# get mylogger as logging reference
mylogger = logging.getLogger()


def test_import():
    """Test data import."""
    try:
        test_df = import_data("./data/bank_data.csv")
        mylogger.info("SUCCESS - Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR - Testing import_eda: The file wasn't found")
        raise err

    try:
        assert test_df.shape[0] > 0
        assert test_df.shape[1] > 0
    except AssertionError as err:
        mylogger.error("ERROR - Testing import_data: "
                       "The file doesn't appear to have rows and columns")
        raise err

    pytest.df = test_df


def test_eda():
    """test perform eda function."""
    df_bank = pytest.df
    df_bank['Churn'] = df_bank['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # setup and initialize all necessary directories
    pytest.dirs = Directories(pl.Path(os.getcwd()))
    try:
        perform_eda(df_bank, pytest.dirs)
        mylogger.info("SUCCESS - Testing perform_eda.")
    except FileNotFoundError as err:
        mylogger.error("ERROR - Testing perform_eda: The file wasn't found")
        raise err

    try:
        assert df_bank.shape[0] > 0
        assert df_bank.shape[1] > 0
    except AssertionError as err:
        mylogger.error("ERROR - Testing test_eda: "
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
        mylogger.info("SUCCESS - Testing encoder_helper.")
    except FileNotFoundError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "The file wasn't found.")
        raise err
    try:
        assert df_bank.shape[0] == shape_0
        assert df_bank.shape[1] > shape_1
        mylogger.info(
            "SUCCESS - Testing encoder_helper: "
            "Appending categorical values.")
    except AssertionError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "Appending categorical values failed.")
        raise err

    pytest.df = df_bank


def test_perform_feature_engineering():
    """	test perform_feature_engineering."""
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            pytest.df)
        mylogger.info("SUCCESS - Testing perform_feature_engineering.")
    except FileNotFoundError as err:
        mylogger.error(
            "ERROR - Testing perform_feature_engineering: "
            "The file wasn't found.")
        raise err

    try:
        assert len(x_train.shape) == 2
        mylogger.info(
            "SUCCESS - Testing encoder_helper: "
            "x_train got right shape.")
    except AssertionError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "x_train got wrong shape.")
        raise err

    try:
        assert len(y_train.shape) == 1
        mylogger.info(
            "SUCCESS - Testing encoder_helper: "
            "y_train got right shape.")
    except AssertionError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "y_train got wrong shape.")
        raise err

    try:
        assert len(x_test.shape) == 2
        mylogger.info(
            "SUCCESS - Testing encoder_helper: "
            "x_test got right shape.")
    except AssertionError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "x_test got wrong shape.")
        raise err

    try:
        assert len(y_test.shape) == 1
        mylogger.info(
            "SUCCESS - Testing encoder_helper: "
            "y_test got right shape.")
    except AssertionError as err:
        mylogger.error(
            "ERROR - Testing encoder_helper: "
            "y_test got wrong shape.")
        raise err

    pytest.x_train = x_train
    pytest.y_train = y_train
    pytest.x_test = x_test
    pytest.y_test = y_test


def test_train_models():
    """test train_models."""

    try:
        cv_rfc, lrc = train_models(
            pytest.x_train,
            pytest.x_test,
            pytest.train_y,
            pytest.y_test,
            pytest.dirs)

        mylogger.info("SUCCESS - Testing train_models.")
    except FileNotFoundError as err:
        mylogger.error("ERROR - Testing train_models: The file wasn't found.")
        raise err

    if not isinstance(cv_rfc, GridSearchCV):
        mylogger.error(
            "ERROR - Testing train_models: "
            "Trained grid search model failed.")
        mylogger.error(
            "ERROR - Expected type GridSearchCV, "
            f"but got {type(cv_rfc)}.")
        raise ValueError

    mylogger.info(
        "SUCCESS - Testing train_models: "
        "Got trained grid search model.")

    if not isinstance(lrc, LogisticRegression):
        mylogger.error(
            "ERROR - Testing train_models: "
            "Trained  logistic regression model failed.")
        mylogger.error(
            "ERROR - Expected type LogisticRegression, "
            f"but got {type(lrc)}.")
        raise ValueError

    mylogger.info(
        "SUCCESS - Testing train_models: "
        "Got trained logistic regression model.")


if __name__ == '__main__':
    mylogger.info(' About to start the tests ')
    pytest.main(args=[os.path.abspath(__file__)])
    mylogger.info(' Done executing the tests ')
