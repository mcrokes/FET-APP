import numpy as np


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


def get_modified_dataframe(model_c):
    df = model_c.get_dataframe()
    targets_modified = model_c.get_target_values_classification_dict()
    target = targets_modified['column_name']
    for value in targets_modified['variables']:
        # noinspection PyTypeChecker
        df.replace({f'{target}': value['old_value']},
                   {f'{target}': value['new_value']},
                   inplace=True)

    q_variables_modified_list = model_c.get_q_variables_values_list()

    for q_variables_modified in q_variables_modified_list:
        for q_variable_modified in q_variables_modified['variables']:
            df.replace({f'{q_variables_modified["column_name"]}': q_variable_modified['old_value']},
                       {f'{q_variables_modified["column_name"]}': q_variable_modified['new_value']},
                       inplace=True)
    return df


def update_y_pred(prediction: list, probability_predictions, cut_off, positive_class):
    total: int = len(probability_predictions)
    predictions = prediction
    for i in range(0, total):
        if cut_off <= probability_predictions[i][positive_class]:
            predictions[i] = positive_class
        else:
            prob = 0
            for probability in probability_predictions[i]:
                if probability != probability_predictions[i][positive_class] and probability > prob:
                    prob = probability

            predictions[i] = np.where(probability_predictions[i] == prob)[0]
    return predictions
