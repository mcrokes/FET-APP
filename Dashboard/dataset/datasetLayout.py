from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from app.proccessor.model.dataset_interaction_methods import get_modified_dataframe
from app.proccessor.models import ExplainedClassifierModel, ModelForProccess


def generateDataSetDistributions(df: pd.DataFrame):
    qualitative_graphs_array = []
    numeric_graphs_array = []
    index = 0
    for feature in df:
        values = list(set(df[feature]))
        counts = df[feature].value_counts()
        if len(values) < 5:
            qualitative_graphs_array.append({"predictor": feature, "graph_data": []})
            for value in values:
                qualitative_graphs_array[index]["graph_data"].append(
                    go.Bar(name=value, x=[value], y=[counts[value]])
                )
            index += 1
        else:
            x = []
            y = []
            for value, count in sorted(zip(values, counts), key=lambda x: x):
                x.append(value)
                y.append(count)
            numeric_graphs_array.append(
                {
                    "predictor": feature,
                    "graph_data": [
                        go.Scatter(
                            x=x,
                            y=y,
                            name="Line",
                            line=dict(color="royalblue", width=1, dash="dot"),
                        ),
                        go.Bar(name="Bar", x=x, y=y, width=0.5),
                    ],
                }
            )
    return qualitative_graphs_array, numeric_graphs_array


datasetLayout = html.Div(
    [
        dcc.Loading(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.H3(
                                            ["DATASET ", html.Span(id="dataset-title")],
                                            style={"text-align": "center"},
                                        ),
                                        html.Div(id="dataset-view"),
                                    ]
                                )
                            ],
                            style={"margin": "auto"},
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("DESCRIPCIÓN PREDICTORES"),
                                html.Div(id="features-description"),
                            ],
                            xs=8,
                            sm=8,
                            md=8,
                            lg=8,
                            xl=8,
                            xxl=8,
                        )
                    ],
                    style={"padding-top": "20px"},
                ),
                dbc.Row(
                    [
                        html.H3("METRICAS DEL DATA SET"),
                        html.H4("Variables Numericas"),
                        html.Div(id="numeric-plot"),
                        html.H4("Variables Objeto"),
                        html.Div(id="object-plot"), 
                        html.Div(id="correlation-plot"),
                    ],
                    style={"padding-top": "20px"},
                ),
            ]
        )
    ],
    className="section-content",
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def datasetCallbacks(app, furl: Function):
    @app.callback(
        Output("dataset-title", "children"),
        Output("dataset-view", "children"),
        Output("numeric-plot", "children"),
        Output("object-plot", "children"),
        Output("correlation-plot", "children"),
        Input("path", "href"),
    )
    def graph_explainers(cl):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == model_id
            ).first()
            
            
            original_df = model_x.data_set_data.getElement("dataset")
            df = model_x.data_set_data.getElement("dataset_modified")
            dtt = model_x.getElement("name")
            qualitative_graphs_array, numeric_graphs_array = (
                generateDataSetDistributions(df)
            )
            corr_matrix = original_df.drop(columns=model_x.getElement("target_row")).corr(method='pearson')
            return (
                dtt,
                html.Div(
                    [
                        dash_table.DataTable(
                            data=df.to_dict("records"),
                            columns=[{"name": i, "id": i} for i in df.columns],
                            page_size=10,
                        ),
                    ]
                ),
                [
                    html.Div(
                        id=f"contribution_graph_{data["predictor"]}",
                        children=dcc.Graph(
                            figure=go.Figure(
                                data=data["graph_data"],
                                layout=dict(title=data["predictor"]),
                            )
                        ),
                    )
                    for data in numeric_graphs_array
                ],
                [
                    html.Div(
                        id=f"contribution_graph_{data["predictor"]}",
                        children=dcc.Graph(
                            figure=go.Figure(
                                data=data["graph_data"],
                                layout=dict(title=data["predictor"]),
                            )
                        ),
                    )
                    for data in qualitative_graphs_array
                ],
                dcc.Graph(
                    figure= go.Figure(
                        data=go.Heatmap(
                            z=corr_matrix,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            text=round(corr_matrix, 2),
                            texttemplate="%{text}",
                        ),
                        layout=dict(title="Variables Correlation")
                    )
                )
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
