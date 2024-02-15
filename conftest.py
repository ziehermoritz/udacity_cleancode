#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 07:36:41 2024

@author: mo

conftest.py file for pytest namespace definitions.
works in combination with test_churn_library.py
"""
import pytest

def df_plugin():
    return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    pytest.df = df_plugin()
    pytest.path = df_plugin()
    pytest.dirs = df_plugin()

    pytest.x_train = df_plugin()
    pytest.y_train = df_plugin()
    pytest.x_test = df_plugin()
    pytest.y_test = df_plugin()

