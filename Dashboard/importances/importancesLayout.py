import multiprocessing
from pyclbr import Function
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from app.proccessor.models import ExplainedClassifierModel, ModelForProccess

importancesLayout = html.Div(
    [
        dcc.Loading([
            dbc.Row([
                dbc.Col([
                    html.Div(id='importance-output-upload')
                ],
                    xs=8, sm=8, md=8, lg=8, xl=8, xxl=8
                ),
                dbc.Col([
                    html.Div(["TEXTO CON EXPLICACION DEL GRÁFICO"],
                             id='importance-explanation')
                ],
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                )
            ], style={'padding-top': '20px'}
            ),
            dbc.Row([
                dbc.Col([
                    html.Div(id='permutation-importance-output-upload')
                ],
                    xs=8, sm=8, md=8, lg=8, xl=8, xxl=8
                ),
                dbc.Col([
                    html.Div(["TEXTO CON EXPLICACION DEL GRÁFICO"],
                             id='permutation-importance-explanation')
                ],
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                )
            ], style={'padding-top': '20px'}
            )
        ])

    ],
    style={'padding-left': '30px', 'padding-right': '30px', 'margin': 'auto'}
)

def importancesCallbacks(app, furl:Function):    
    @app.callback(
        Output("importance-output-upload", "children"),
        Output("importance-explanation", "children"),
        Output("permutation-importance-output-upload", "children"),
        Output("permutation-importance-explanation", "children"),
        Input("path", "href"),
    )
    def graph_explainers(cl):
        f = furl(cl)
        param1 = f.args["model_id"]
        try:
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == param1
            ).first()
            
            classifier_model: RandomForestClassifier = model_x.getElement("model")
            classifier_dataset: pd.DataFrame = model_x.data_set_data.getElement("dataset")
            df_feature_importance: pd.DataFrame = pd.DataFrame({'Predictor': classifier_model.feature_names_in_, 'Importancia': classifier_model.feature_importances_})
            importances_fig = px.bar(data_frame=df_feature_importance.sort_values('Importancia', ascending=False), x='Importancia',
                     y='Predictor',
                     title='Importance GINI')
            explanation = "EXPLANATION"
            permutation_importance_model = permutation_importance(
                estimator=classifier_model,
                X=classifier_dataset.drop(columns=model_x.target_row),
                y=classifier_dataset[model_x.target_row],
                n_repeats=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=multiprocessing.cpu_count() - 1,
                random_state=123
            )
            df_permutation_importance = pd.DataFrame(
                {k: permutation_importance_model[k] for k in ['importances_mean', 'importances_std']})
            df_permutation_importance['Predictor'] = classifier_model.feature_names_in_
            df_ordered_importance = df_permutation_importance.sort_values('importances_mean', ascending=True)
            permutation_fig = px.bar(data_frame=df_ordered_importance, error_x=df_ordered_importance['importances_std'],
                        x='importances_mean', y='Predictor',
                        title='IMPORTANCIAS POR PERMUTACION',
                        labels={'importances_mean': 'Importancia +- error: '})
            return (dcc.Graph(figure=importances_fig), explanation, dcc.Graph(figure=permutation_fig), explanation)
        except Exception as e:
            print(e)
            raise PreventUpdate