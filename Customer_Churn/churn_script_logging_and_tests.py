"""
This is a testing and logging script.

Author: Domenico Vesia
Date: 2021-05-26
"""

import logging
import pytest
import joblib
import churn_library
import constants

FORMAT = "[%(asctime)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format=FORMAT,
    force=True)


@pytest.fixture(scope="module")
def return_path():
    """
    path fixture - returns csv file path
    """
    path = "./data/bank_data.csv"
    return path


@pytest.fixture(scope="module")
def return_dataframe(return_path):
    """
    dataframe fixture - returns pandas dataframe
    """
    dataframe = churn_library.create_dataframe_from_csv(return_path)
    return dataframe


@pytest.fixture(scope="module")
def return_categorical_columns(return_dataframe):
    """
    categorical columns fixture - returns categorical columns
    """
    categorical_columns = return_dataframe.select_dtypes(
        include=['object', 'bool']).columns
    return categorical_columns


@pytest.fixture(scope="module")
def return_encoded_dataframe(return_dataframe):
    """
    encoded dataframe fixtures - returns encoded dataframe
    """
    encoded_dataframe = churn_library.encoder_helper(return_dataframe)
    return encoded_dataframe


@pytest.fixture(scope="module")
def return_train_test_split_sequences(return_encoded_dataframe):
    """
    train_test_split fixtures - returns 4 series containing train and test features and label
    """
    x_train, x_test, y_train, y_test, x_data = churn_library.preprocessing(
        return_encoded_dataframe)
    return x_train, x_test, y_train, y_test, x_data


def test_create_dataframe_from_csv(return_path):
    """
    test create_dataframe_from_csv function
    """
    try:
        dataframe = churn_library.create_dataframe_from_csv(return_path)
        logging.info(return_path)
        logging.info("SUCCESS: Valid path")
    except FileNotFoundError as err:
        logging.info(return_path)
        logging.info("No such file or directory")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.info("DataFrame Shape: %s", dataframe.shape)
        logging.error("DataFrame seems to not have rows and columns")
        raise err
    try:
        assert "Churn" in dataframe.columns
        logging.info("DataFrame Shape: %s", dataframe.shape)
        logging.info("SUCCESS: DataFrame created")
    except AssertionError as err:
        logging.info(
            "Dataframe Columns\n%d",
            list(col for col in dataframe.columns))
        logging.error("Churn column not found")
        raise err


def test_perform_eda(return_dataframe):
    """
    test perform eda function
    """

    try:
        assert return_dataframe.shape[0] > 0
        assert return_dataframe.shape[1] > 0
    except AssertionError as err:
        logging.info("DataFrame Shape: %s", return_dataframe.shape)
        logging.error("DataFrame seems to not have rows and columns")
        raise err

    churn_library.perform_eda(return_dataframe)

    plot_columns_dictionary = constants.PLOT_COLUMNS_DICTIONARY

    pairs = [(plot, columns_name)
             for plot, columns_name in plot_columns_dictionary.items()
             for column_name in columns_name]

    try:
        for pair in pairs:
            for column_name in pair[1]:
                with open(f"images/eda/{pair[0]}_{column_name}.jpg", 'r'):
                    logging.info(
                        "SUCCESS: %s_%s.jpg created correctly",
                        pair[0],
                        column_name)
    except FileNotFoundError as err:
        logging.info(
            "%s_%s.jpg not found in images/eda/ directory",
            pair[0],
            column_name)
        raise err


def test_encoder_helper(return_dataframe):
    """
    test encoder helper
    """
    return_dataframe_cat_cols = return_dataframe.select_dtypes(
        ['object', 'bool']).columns
    encoded_dataframe = churn_library.encoder_helper(return_dataframe)

    try:
        for column in [
                column +
                '_Churn' for column in return_dataframe_cat_cols]:
            assert column in encoded_dataframe
        logging.info("DataFrame Encoded Correctly")
    except AssertionError as err:
        logging.error("%s not in encoded_dataframe", column)
        raise err


def test_preprocessing(return_encoded_dataframe):
    """
    test preprocessing function
    """
    x_train, x_test, y_train, y_test, x_data = churn_library.preprocessing(
        return_encoded_dataframe)
    stage_data = ["x_train", "x_test", "y_train", "y_test", "x_data"]

    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == len(x_data)
        logging.info("DataFrame Preprocessed Correctly")
        for data in stage_data:
            logging.info("%s %s", data, eval(data).shape)
    except ValueError as err:
        logging.info("%s %s", data, eval(data).shape)
        logging.error("Found input variables with inconsistent shape")
        raise err


def test_train_models(
        return_train_test_split_sequences):
    """
    test train_models function
    """
    churn_library.train_models(
        return_train_test_split_sequences[0],
        return_train_test_split_sequences[1],
        return_train_test_split_sequences[2],
        return_train_test_split_sequences[3],
        return_train_test_split_sequences[4])

    output_dictionary = constants.OUTPUT_DICTIONARY

    try:
        for model in output_dictionary["models_saved"]:
            joblib.load(f"models/{model}")
            logging.info("%s created", model)
    except FileNotFoundError as err:
        logging.info("%s not created", model)
        raise err

    try:
        for report_image in output_dictionary["report_images_saved"]:
            with open(f"images/results/{report_image}", 'r'):
                logging.info("%s created", report_image)
        logging.info("SUCCESS: classification_report_image correctly executed")
    except FileNotFoundError as err:
        logging.info("images/results/%s not created", report_image)
        raise err

    try:
        for fi_image in output_dictionary["feature_importance_images_saved"]:
            with open(f"images/results/{fi_image}", 'r'):
                logging.info("%s created", fi_image)
        logging.info("SUCCESS: feature_importance_plot correctly executed")
    except FileNotFoundError as err:
        logging.info("images/results/%s not created", fi_image)
        raise err

    try:
        for curve_image in output_dictionary["performance_curve_images_saved"]:
            with open(f"images/results/{curve_image}", 'r'):
                logging.info("%s created", curve_image)
        logging.info("SUCCESS: feature_importance_plot correctly executed")
    except FileNotFoundError as err:
        logging.info("images/results/%s not created", curve_image)
        raise err
