def get_target_dropdown(values_dict):
    return [{'label': value['new_value'], 'value': index} for index, value  in enumerate(values_dict)]
