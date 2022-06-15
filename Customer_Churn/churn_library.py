# library doc string
"""
This is a library of functions to find customers who are likely to churn.

Author: Domenico Vesia
Date: 2021-05-26
"""

# import libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import constants


def create_dataframe_from_csv(path):
    """
    Returns pandas DataFrame from a csv file.


        Parameters:
            path (str): csv file location

        Returns:
            dataframe (pandas DataFrame): dataframe
    """
    dataframe = pd.read_csv(path)

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    dataframe = dataframe.drop('Attrition_Flag', axis=1)

    return dataframe


def perform_eda(dataframe):
    """
    Perform EDA on dataframe and save figures to images folder.


        Parameters:
            dataframe (pandas DataFrame)

        Returns:
            None
    """
    plot_columns_dictionary = constants.PLOT_COLUMNS_DICTIONARY

    for plot, columns_name in plot_columns_dictionary.items():
        if plot == "countplot":
            for column_name in columns_name:
                plt.figure(figsize=(20, 10))
                plt.title(f"{plot}_{column_name}", size=18)
                sns.countplot(data=dataframe, x=column_name)

                plt.savefig(f"images/eda/{plot}_{column_name}.jpg")
                plt.close()
        elif plot == "histplot":
            for column_name in columns_name:
                plt.figure(figsize=(20, 10))
                plt.title(f"{plot}_{column_name}", size=18)
                sns.histplot(data=dataframe, x=column_name, kde=True)

                plt.savefig(f"images/eda/{plot}_{column_name}.jpg")
                plt.close()
        elif plot == "heatmap":
            for column_name in columns_name:
                plt.figure(figsize=(20, 10))
                plt.title(f"{plot}_{column_name}", size=18)
                sns.heatmap(
                    dataframe.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)

                plt.savefig(f"images/eda/{plot}_{column_name}.jpg")
                plt.close()


def encoder_helper(dataframe):
    """
    Turn each categorical column into a new nunmerical column
    with churn propotion for each category.


        Parameters:
            dataframe (pandas DataFrame)
            categorical_columns_lst (list): list of categorical columns

        Returns:
            dataframe (pandas DataFrame): pandas DataFrame with refreshed columns
    """

    encoded_dataframe = dataframe.copy()
    categorical_columns_lst = encoded_dataframe.select_dtypes(
        ['object', 'bool']).columns

    for categorical_column in categorical_columns_lst:
        categorical_column_groups = encoded_dataframe.groupby(
            categorical_column).mean()['Churn']
        encoded_dataframe[categorical_column] = [
            categorical_column_groups.loc[val] for val in encoded_dataframe[categorical_column]]
        encoded_dataframe.rename(
            columns={
                categorical_column: categorical_column +
                '_Churn'},
            inplace=True)

    return encoded_dataframe


def preprocessing(encoded_dataframe):
    """
    Select the columns of interests,
    normalize its and splits the data into
    train and test set.


        Parameters:
            dataframe (pandas DataFrame): Pandas DataFrame

        Returns:
              x_train (pandas DataFrame): X training data
              x_test (pandas DataFrame): X testing data
              y_train (pandas Series): y training data
              y_test (pandas Series): y testing data
              x_data (pandas DataFrame): X data
     """
    keep_cols = constants.COLUMNS_TO_KEEP

    x_data = encoded_dataframe[keep_cols]
    y_data = encoded_dataframe['Churn']

    mm_scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, x_data


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    Produces classification report for training and testing results
    and stores report as image in images folder.


        Parameters:
            y_train (pandas Series): training response values
            y_test (pandas Series):  test response values
            y_train_preds_lr (pandas Series): training predictions from logistic regression
            y_train_preds_rf (pandas Series): training predictions from random forest
            y_test_preds_lr (pandas Series): test predictions from logistic regression
            y_test_preds_rf (pandas Series): test predictions from random forest

        Returns:
             None
    """
    classification_report_dictionary = {
        "random_forest": {
            "train": y_train_preds_rf,
            "test": y_test_preds_rf
        },

        "logistic_regression": {
            "train": y_train_preds_lr,
            "test": y_test_preds_lr
        }
    }

    for model, model_data in classification_report_dictionary.items():
        for stage in model_data:
            if stage == 'train':
                plt.figure(figsize=(10, 10))
                plt.text(
                    0.01, 1.0, str(f"{model} {stage}"), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.5, str(
                        classification_report(
                            y_train, model_data.get('train'))), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.axis('off')

                plt.savefig(f"./images/results/report_{model}_{stage}.jpg")
                plt.close()

            elif stage == 'test':
                plt.figure(figsize=(10, 10))
                plt.text(
                    0.01, 1.0, str(f"{model} {stage}"), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.text(
                    0.01, 0.5, str(
                        classification_report(
                            y_test, model_data.get('test'))), {
                        'fontsize': 10}, fontproperties='monospace')
                plt.axis('off')

                plt.savefig(f"./images/results/report_{model}_{stage}.jpg")
                plt.close()


def feature_importance_plot(
        feature_importance_dictionary,
        x_data,
        output_pth):
    """
    Creates the feature importances plot and
    stores it as image in images folder.


        Parameters:
            feature_importances_dictionary (dictionary): dictionary of model features importance
            x_data (pandas DataFrame): unlabeled dataframe
            output_pth (str): path to store the figure

        Returns:
            None

        Example:
            feature_importance_plot(
            {"logistic regression": lr_model.coef_[0]},
             X_data,
            /images/results)
    """

    for model, feature_importance in feature_importance_dictionary.items():
        importances = feature_importance
        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]
        plt.figure(figsize=(30, 20))
        plt.title(f"{model} feature importances")
        plt.ylabel("Importance")
        plt.bar(range(x_data.shape[1]), importances[indices])
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        plt.savefig(f"{output_pth}feature_importances_{model}.jpg")
        plt.close()


def plot_performances_curve(models_lst, x_test, y_test, output_pth):
    """
    Creates performances_curve plot and
    store it as image in images folder.


        Parameters:
            models_lst (list): list of models
            x_test (pandas DataFrame)
            y_test (pandas Serire)
            output_pth (str): path to store the figure

        Returns:
            None
    """

    plt.figure(figsize=(15, 8))
    axis = plt.gca()

    for model in models_lst:
        plot_roc_curve(model, x_test, y_test, ax=axis)

    plt.savefig(f"{output_pth}auc-roc_curve.jpg")
    plt.close()


def train_models(x_train, x_test, y_train, y_test, x_data):
    """
    Train and stores models results in images and models folders


        Parameters:
            x_train (pandas DataFrame): X training data
            x_test (pandas DataFrame): X testing data
            y_train (pandas Series): y training data
            y_test (pandas Series): y testing data
            x_data (pandas DataFrame): X unlabeled dataframe

        Returns:
            None
    """

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = constants.PARAM_GRID

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    models = [cv_rfc.best_estimator_, lrc]
    models_feature_importances_dictionary = {
        "random_forest": cv_rfc.best_estimator_.feature_importances_,
        "logistic_regression": lrc.coef_[0]}

    path = "./images/results/"

    feature_importance_plot(
        models_feature_importances_dictionary, x_data, path)
    plot_performances_curve(models, x_test, y_test, path)

if __name__ == "__main__":
    path = constants.PATH
    dataframe = create_dataframe_from_csv(path)
    perform_eda(dataframe)
    encoded_dataframe = encoder_helper(dataframe)
    x_train, x_test, y_train, y_test, x_data = preprocessing(encoded_dataframe)
    train_models(x_train, x_test, y_train, y_test, x_data)
    