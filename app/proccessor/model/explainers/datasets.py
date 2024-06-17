from dash import html, dash_table

from ..dataset_interaction_methods import get_modified_dataframe


def create_data_table(model_c):
    df = get_modified_dataframe(model_c)
    return (model_c.get_name(), html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white'
            },
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            },
        ),

    ]))
