# constants doc string
"""
This is a constants script.

Author: Domenico Vesia
Date: 2021-05-26
"""

COLUMNS_TO_KEEP = [
    "Customer_Age",
    "Gender_Churn",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn"]

PLOT_COLUMNS_DICTIONARY = {"histplot": ["Customer_Age", "Total_Trans_Ct"],
                           "countplot": ["Churn", "Marital_Status"],
                           "heatmap": ["Correlation_Matrix"]}

OUTPUT_DICTIONARY = {
    "models_saved": [
        "logistic_model.pkl",
        "rfc_model.pkl"],
    "report_images_saved": [
        "report_logistic_regression_test.jpg",
        "report_logistic_regression_train.jpg",
        "report_random_forest_test.jpg",
        "report_random_forest_train.jpg"],
    "feature_importance_images_saved": [
        "feature_importances_logistic_regression.jpg",
        "feature_importances_random_forest.jpg"],
    "performance_curve_images_saved": ["auc-roc_curve.jpg"]}

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
