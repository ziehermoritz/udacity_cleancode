#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:31:48 2024

author: moritz
"""
# import libraries
import os
import sys
import pathlib as pl
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
# from sklearn.preprocessing import normalize


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    """
    Return dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(dataframe, directories):
    """
    Perform eda on dataframe and save figures to images folder.

    input:
            dataframe: pandas dataframe

    output:
            None
    """
    print(f'{dataframe.head()}\n')  # print first rows of dataframe
    # print the shape of dataframe
    print(f'Read in data frame is of shape: {dataframe.shape}\n')
    print(f'{dataframe.isnull().sum()}\n')  # get an idea of empty rows/columns
    # pritn statistical information of dataframe
    print(f'{dataframe.describe()}\n')

    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]


    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(directories.eda_dir.joinpath(
        'churn_histogram.png').as_posix(), dpi=600)
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(directories.eda_dir.joinpath(
        'customerage_histogram.png').as_posix(), dpi=600)
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(directories.eda_dir.joinpath(
        'normalized_martialstatus.png').as_posix(), dpi=600)
    plt.close()

    # plot histogram of Total Transaction Count (Last 12 months)
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(dataframe['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(directories.eda_dir.joinpath(
        'total_trans_ct_last12m.png').as_posix(), dpi=600)
    plt.close()

    # plot the heatmap of the correlation matrix of dataframe
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe[quant_columns].corr(), annot=False, cmap='Dark2_r',
                linewidths=2)
    # plt.show()
    plt.savefig(directories.eda_dir.joinpath(
        'dataframe_correlation.png').as_posix(), dpi=600)
    plt.close()


def encoder_helper(dataframe, category_lst, response=None):
    """
    Utilitiy function to turn each categorical column into a new column.

    Propotion of churn for each category - associated with cell 15 from the
    notebook.

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or
             index y column]

    output:
            dataframe: pandas dataframe with new columns for
    """
    if response:
        pass
    else:
        response = 'Churn'

    for cat in category_lst:
        tmp_lst = []
        cat_groups = dataframe.groupby(cat)[response].mean()
        for val in dataframe[cat]:
            tmp_lst.append(cat_groups.loc[val])
        dataframe[f'{cat}_{response}'] = tmp_lst

    return dataframe


def perform_feature_engineering(dataframe, response=None):
    """
    Utilitiy function to extract features.

    input:
              dataframe: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables
               or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    if response:
        pass
    else:
        response = 'Churn'

    data_y = dataframe[response]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    data_x = pd.DataFrame()
    data_x[keep_cols] = dataframe[keep_cols]

    print(data_x.head())

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_data_dict):
    """
    Classification report for training and testing results.

    Stores report as image in images folder.

    input:
        y_data_dict - (dictionary) - with entries:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # scores
    print('random forest results')
    print('test results')
    print(
        classification_report(
            y_data_dict["y_test"],
            y_data_dict["y_test_preds_rf"]))
    print('train results')
    print(
        classification_report(
            y_data_dict["y_train"],
            y_data_dict["y_train_preds_rf"]))

    print('logistic regression results')
    print('test results')
    print(
        classification_report(
            y_data_dict["y_test"],
            y_data_dict["y_test_preds_lr"]))
    print('train results')
    print(
        classification_report(
            y_data_dict["y_train"],
            y_data_dict["y_train_preds_lr"]))


def feature_importance_plot(model, x_data, output_pth):
    """
    Create and store the feature importances in pth.

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(pl.Path(output_pth).joinpath(
        'feature_importance.png'), format='png', dpi=600)


def train_models(x_train, x_test, y_train, y_test, directories):
    """
    Train, store model results: images + scores, and store models.

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfcl = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrcl = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfcl = GridSearchCV(estimator=rfcl, param_grid=param_grid, cv=5)
    cv_rfcl.fit(x_train, y_train)

    lrcl.fit(x_train, y_train)

    y_train_preds_rf = cv_rfcl.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfcl.best_estimator_.predict(x_test)

    y_train_preds_lr = lrcl.predict(x_train)
    y_test_preds_lr = lrcl.predict(x_test)

    # plots
    plt.figure(figsize=(15, 8))
    # ax = plt.gca()
    # rfc_disp =
    RocCurveDisplay.from_estimator(
        cv_rfcl.best_estimator_, x_test, y_test, ax=plt.gca(), alpha=0.8)
    # lrc_plot =
    RocCurveDisplay.from_estimator(
        lrcl, x_test, y_test, ax=plt.gca(), alpha=0.8)
    # plt.show()
    plt.savefig(directories.results_dir.joinpath(
        'results.png').as_posix(), dpi=600)
    plt.close()

    # save best model
    joblib.dump(cv_rfcl.best_estimator_,
                directories.model_dir.joinpath('rfc_model.pkl'))
    joblib.dump(lrcl, directories.model_dir.joinpath('logistic_model.pkl'))

    explainer = shap.TreeExplainer(cv_rfcl.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig(directories.results_dir.joinpath(
        'summary.png').as_posix(), dpi=600)
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')

    # approach improved by OP -> monospace!
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')

    # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(directories.results_dir.joinpath(
        'randomforest_train.png').as_posix(), dpi=600)
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(directories.results_dir.joinpath(
        'logistic_regression_train.png').as_posix(), dpi=600)
    plt.close()

    return cv_rfcl, lrcl


class Directories:
    """
    A class to store all needed directories.

    Attributes
    ----------
    working_dir - (pathlib.Path) - current working directory serving as root
    image_dir - (pathlib.Path) - toplevel directory of figures and images
    eda_dir - (pathlib.Path) - destination directory for eda figures
    results_dir - (pathlib.Path) - destination directory for result figures
    model_dir - (pathlib.Path) - destination directory for best models
    log_dir - (pathlib.Path) - destination directory for logging data
    """

    def __init__(self, working_dir):
        """Initialize all directories specified."""
        self.working_dir = working_dir

        self.image_dir = self.working_dir.joinpath('images')
        self.eda_dir = self.image_dir.joinpath('eda')
        self.results_dir = self.image_dir.joinpath('results')
        self.model_dir = self.working_dir.joinpath('models')
        self.log_dir = self.working_dir.joinpath('logs')
        self.avail_dirs = [
            self.image_dir,
            self.eda_dir,
            self.results_dir,
            self.model_dir,
            self.log_dir]

    def init(self):
        """Initialize all directories specified."""
        try:
            self.image_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.image_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.image_dir} already exists!')

        try:
            self.eda_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.eda_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.eda_dir} already exists!')

        try:
            self.results_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.results_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.results_dir} already exists!')

        try:
            self.model_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.model_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.model_dir} already exists!')

        try:
            self.log_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.log_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.log_dir} already exists!')

    def print_available_directories(self):
        """Print out all available directories with absolute path."""
        for dir_tmp in self.avail_dirs:
            print(f'{dir_tmp}')


if __name__ == "__main__":

    curr_dir = pl.Path(os.getcwd())
    try:
        curr_dir.mkdir(parents=False, exist_ok=False)
        print(f'{curr_dir} created!')
    except FileNotFoundError:
        print(f'Parent directory of {curr_dir} missing!')
        sys.exit(15)
    except FileExistsError:
        print(f'{curr_dir} already exists!')

    # setup and initialize all necessary directories
    dirs = Directories(curr_dir)
    dirs.init()

    input_pth = dirs.working_dir.joinpath('data/bank_data.csv')
    df_bank = import_data(input_pth)

    # get dummy variable for closed accounts
    # existing customers are not relevant in that case
    # could do this with dataframe.get_dummy_variables() ?
    df_bank['Churn'] = df_bank['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df_bank, dirs)

    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    df_bank = encoder_helper(df_bank, cat_lst)

    train_X, test_X, train_y, test_y = perform_feature_engineering(df_bank)

    cv_rfc, lrc = train_models(train_X, test_X, train_y, test_y, dirs)

    train_y_preds_rf = cv_rfc.best_estimator_.predict(train_X)
    test_y_preds_rf = cv_rfc.best_estimator_.predict(train_X)

    train_y_preds_lr = lrc.predict(train_X)
    test_y_preds_lr = lrc.predict(train_X)

    y_dict = {
        "train_y": train_y,
        "test_y": test_y,
        "train_y_preds_lr": train_y_preds_lr,
        "train_y_preds_rf": train_y_preds_rf,
        "test_y_preds_lr": test_y_preds_lr,
        "test_y_preds_rf": test_y_preds_rf}

    classification_report_image(y_dict)
