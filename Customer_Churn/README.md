# __Customer Churn Project__

This project it's for to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## __Files Description__

---

### __churn_library.py__

Library of functions to find customers who are likely to churn. 

### __churn_script_logging_and_tests.py__  

- Contain unit tests for the churn_library.py functions. 
- Log any errors and INFO messages. 

### __constants.py__  

Contain costant used by function in upper files 

## __Running Files__

---

### __churn_library.py__

    ipython churn_library.py

### __churn_script_logging_and_tests.py__

    pytest -v churn_script_logging_and_tests.py

## __Project Structure__

---

<pre>
.
├── churn_library.py
├── churn_script_logging_and_tests.py
├── constants.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── countplot_Churn.jpg
│   │   ├── countplot_Marital_Status.jpg
│   │   ├── heatmap_Correlation_Matrix.jpg
│   │   ├── histplot_Customer_Age.jpg
│   │   └── histplot_Total_Trans_Ct.jpg
│   └── results
│       ├── auc-roc_curve.jpg
│       ├── feature_importances_logistic regression.jpg
│       ├── feature_importances_logistic_regression.jpg
│       ├── feature_importances_random forest.jpg
│       ├── feature_importances_random_forest.jpg
│       ├── report_logistic_regression_test.jpg
│       ├── report_logistic_regression_train.jpg
│       ├── report_random_forest_test.jpg
│       └── report_random_forest_train.jpg
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md
└── requirements
    ├── requirements_py3.6.txt
    └── requirements_py3.8.txt
</pre>
