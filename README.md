# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

- ### Table of Contents

1. [Project Description](#projectdescription)
2. [Files and data description](#files)
3. [Installation](#installation)
4. [Running Files](#running)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Description <a name="projectdescription"></a>
Framework to identify credit card customers that are most likely to churn.
This is a project aims to train and to implement best coding practices.

## Files and data description <a name="files"></a>

| File | Description |
| --- | --- |
| churn_notebook.ipynb | Jupyter notebook with original unfractured code |
| Guide.ipynb | Instructions for the project tasks |
| requirements_py3.8.txt | Necessary Python packages with used versions|
| churn_library.py | Finalized Framework for project following best coding practice|
| test_churn_library | Implementation of framework testing by pyest|
| conftest.py | Helper script to define pytest-namespace |
| data/bank_data.csv | dataset to be analyzed|

| Directory | Description |
| --- | --- |
| data | contains dataset |
| images | contains figures and images of results |
| models | contains best models |
| logs | testing log resulting from pytest |

## Installation <a name="installation"></a>

The code has been developed by the use of Anaconda distribution of Python.
  - conda Version: 23.7.4
  - Python Version: 3.8.18

## Running Files <a name="running"></a>

The main script churn_library.py is simple run by the command:

```ruby
   python churn_library.py
```

The script will perform data analysis and stores resulting figures and models in the necessary directories.

To test the implemented function an additional testing-script is provided via test_churn_library.py.
The test can either be started by:
```ruby
   python test_churn_library.py
```
or
```ruby
   pytest test_churn_library.py
```
These tests will log the performance and results in the logs-directory.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Kaggle for the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available
[here](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code).
Additionally, the baseline code was provided by the Udacity-Team during the Nanodegree Machine Learning DevOps Engineer.
