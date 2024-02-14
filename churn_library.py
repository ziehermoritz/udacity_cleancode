# library doc string


# import libraries
import os
import sys
import pathlib as pl
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)    
    return df


def perform_eda(df, directories):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    print(f'{df.head()}\n') # print first rows of df
    print(f'Read in data frame is of shape: {df.shape}\n') # print the shape of df
    print(f'{df.isnull().sum()}\n') # get an idea of empty rows/columns
    print(f'{df.describe()}\n') # pritn statistical information of df
    
    # define list of categorical column names
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]
    
    # define list of quantitative column names
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
    
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.savefig(directories.eda_dir.joinpath('churn_histogram.png').as_posix(), dpi=600)
    plt.close()
    
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    plt.savefig(directories.eda_dir.joinpath('customerage_histogram.png').as_posix(), dpi=600)
    plt.close()
    
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.savefig(directories.eda_dir.joinpath('normalized_martialstatus.png').as_posix(), dpi=600)
    plt.close()

    # plot histogram of Total Transaction Count (Last 12 months)
    plt.figure(figsize=(20,10)) 
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    plt.savefig(directories.eda_dir.joinpath('total_trans_ct_last12m.png').as_posix(), dpi=600)
    plt.close()
    
    # plot the heatmap of the correlation matrix of df
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    # plt.show()
    plt.savefig(directories.eda_dir.joinpath('df_correlation.png').as_posix(), dpi=600)
    plt.close()
    


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    
    if response:
        pass
    else:
        response = 'Churn'
        
    y = df[response]
    
    for cat in category_lst:
        tmp_lst = []
        cat_groups = df.groupby(cat).mean()[response]
        for val in df[cat]:
            tmp_lst.append(cat_groups.loc[val])
        df[f'{cat}_{response}'] = tmp_lst
        
    return df      
        

def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    if response:
        pass
    else:
        response = 'Churn'
        
    y = df[response]
    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    
    print(X.head())
    
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
    

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(pl.Path(output_pth).joinpath('feature_importance.png'), format='png', dpi=600)

def train_models(X_train, X_test, y_train, y_test, directories):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    # plt.show()
    plt.savefig(directories.results_dir.joinpath('results.png').as_posix(), dpi=600)
    plt.close()
    
    # save best model
    joblib.dump(cv_rfc.best_estimator_, directories.model_dir.joinpath('rfc_model.pkl'))
    joblib.dump(lrc, directories.model_dir.joinpath('logistic_model.pkl'))
    
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    fig = shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(directories.results_dir.joinpath('summary.png').as_posix(), dpi=600)
    plt.close()
    
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig(directories.results_dir.joinpath('randomforest_train.png').as_posix(), dpi=600)
    plt.close()
    
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig(directories.results_dir.joinpath('logistic_regression_train.png').as_posix(), dpi=600)
    plt.close()
    
    return cv_rfc, lrc
    
class Directories:
    '''
    A class to store all needed directories.
    
    Attributes
    ----------
    working_dir - (pathlib.Path) - current working directory serving as root
    image_dir - (pathlib.Path) - toplevel directory of figures and images
    eda_dir - (pathlib.Path) - destination directory for eda figures
    results_dir - (pathlib.Path) - destination directory for result figures
    model_dir - (pathlib.Path) - destination directory for best models
    log_dir - (pathlib.Path) - destination directory for logging data
    '''
    def __init__(self, working_dir):
        '''
        Setup all directories specified
        '''
        
        self.working_dir = working_dir
        self.image_dir = working_dir.joinpath('images')
        try:
            self.image_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.image_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.image_dir} already exists!')

        self.eda_dir = self.image_dir.joinpath('eda')
        try:
            self.eda_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.eda_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.eda_dir} already exists!')

        self.results_dir = self.image_dir.joinpath('results')
        try:
            self.results_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.results_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.results_dir} already exists!')

        self.model_dir = self.working_dir.joinpath('models')
        try:
            self.model_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.model_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.model_dir} already exists!')

        self.log_dir = self.working_dir.joinpath('logs')
        try:
            self.log_dir.mkdir(parents=False, exist_ok=False)
            print(f'{self.log_dir} created!')
        except FileNotFoundError:
            print(f'Parent directory {self.working_dir} missing!')
            sys.exit(15)
        except FileExistsError:
            print(f'{self.log_dir} already exists!')    
    
if __name__ == "__main__":
    
    working_dir = pl.Path(os.getcwd())
    try:
        working_dir.mkdir(parents=False, exist_ok=False)
        print(f'{working_dir} created!')
    except FileNotFoundError:
        print(f'Parent directory of {working_dir} missing!')
        sys.exit(15)
    except FileExistsError:
        print(f'{working_dir} already exists!')
        
    directories = Directories(working_dir)
    
    input_pth = directories.working_dir.joinpath('data/bank_data.csv')
    df = import_data(input_pth)
    
    # get dummy variable for closed accounts
    # existing customers are not relevant in that case
    # could do this with df.get_dummy_variables() ?
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)  
    
    # perform_eda(df, directories)
    
    
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = encoder_helper(df, category_lst)
    
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    
    cv_rfc, lrc = train_models(X_train, X_test, y_train, y_test, directories)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    