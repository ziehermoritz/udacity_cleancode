#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 07:36:41 2024

@author: mo
"""
import pytest

def df_plugin():
    return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    pytest.df = df_plugin()
    pytest.path = df_plugin()
    pytest.dirs = df_plugin()

    pytest.train_X = df_plugin()
    pytest.train_y = df_plugin()
    pytest.test_X = df_plugin()
    pytest.test_y = df_plugin()

