from numbers import Number
from unicodedata import numeric
import numpy as np
from sklearn.calibration import LabelEncoder

def get_y_transformed(y):
        for value in y:
            if value is not int:
                y = LabelEncoder().fit_transform(y)
                break
        print(y)
        return y


def creating_qualitative_dict(variables):
    qualitative_variables_list = []
    if variables:
        column_names = []
        excluded_columns = []
        for variable in variables:
            qualitative_variable = {'column_name': variable[0]['props']['value']}
            if qualitative_variable['column_name'] not in excluded_columns:
                qualitative_variable['old_value'] = variable[1]['props']['value']
                try:
                    if not variable[2]['props']['value'].strip():
                        raise Exception
                    qualitative_variable['new_value'] = variable[2]['props']['value'].strip()

                    if qualitative_variable['column_name'] not in column_names:
                        column_names.append(qualitative_variable['column_name'])
                        qualitative_variables_list.insert(len(column_names) - 1, {
                            'column_name': qualitative_variable['column_name'],
                            'vriables': [{
                                'old_value': qualitative_variable['old_value'],
                                'new_value': qualitative_variable['new_value'],
                            }]
                        }
                                                          )
                    else:
                        qualitative_variables_list[len(column_names) - 1]['variables'].append(
                            {
                                'old_value': qualitative_variable['old_value'],
                                'new_value': qualitative_variable['new_value'],
                            }
                        )
                except Exception as e:
                    print(str(e))
                    if len(qualitative_variables_list) != 0:
                        excluded_columns.append(qualitative_variable['column_name'])
                        if qualitative_variables_list[len(column_names) - 1]['column_name'] == \
                                qualitative_variable['column_name']:
                            qualitative_variables_list.pop(len(column_names) - 1)

    return qualitative_variables_list


def get_modified_dataframe(df, target_description, qualitative_columns):
    new_df = df.copy()
    if target_description:
        target = target_description['column_name']
        for value in target_description['variables']:
            # noinspection PyTypeChecker
            new_df.replace({f'{target}': value['old_value']},
                    {f'{target}': value['new_value']},
                    inplace=True)

    for q_variables_modified in qualitative_columns:
        print(q_variables_modified)
        for q_variable_modified in q_variables_modified['variables']:
            new_df.replace({f'{q_variables_modified["column_name"]}': q_variable_modified['old_value']},
                       {f'{q_variables_modified["column_name"]}': q_variable_modified['new_value']},
                       inplace=True)
            
        print(new_df.head(3))
    return new_df


def update_y_pred(prediction: list, probability_predictions, cut_off, positive_class, old_class_names):
    total: int = len(probability_predictions)
    predictions = prediction
    for i in range(0, total):
        if cut_off <= probability_predictions[i][positive_class]:
            predictions[i] = old_class_names[positive_class]
        else:
            prob = 0
            for probability in probability_predictions[i]:
                if probability != probability_predictions[i][positive_class] and probability > prob:
                    prob = probability

            predictions[i] = old_class_names[int(np.where(probability_predictions[i] == prob)[0])]
    return predictions
