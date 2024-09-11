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
