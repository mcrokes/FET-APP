def get_target_dropdown(values_dict):
    return [{'label': value['new_value'], 'value': value['old_value']} for value in values_dict['variables']]
