import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_target_dropdown(values_dict):
    return [
        {"label": value["new_value"], "value": index}
        for index, value in enumerate(values_dict)
    ]


def get_y_test_transformed(y_test):
    for value in y_test:
        if value is not int:
            y_test = LabelEncoder().fit_transform(y_test)
            break
    return y_test


def verify_and_round(val):
    if isinstance(val, pd.Series):
        val = val.values[0]
    if isinstance(val, float):
        val = round(val, 3)
    return val
