from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from app.proccessor.models import ExplainedClassifierModel


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
                html.Plaintext(
                    [
                        "Métricas del Conjunto de Datos: ",
                        html.Strong(id="dataset-title"),
                    ],
                    className="rules-title",
                    style={"font-size": "30px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Plaintext(
                                    "Conjunto de Datos Modificado",
                                    className="rules-title",
                                ),
                                html.I(
                                    id="dataset-info",
                                    className="fa fa-info-circle info-icon",
                                ),
                                dbc.Tooltip(
                                    [
                                        html.Plaintext(
                                            [
                                                "* Si desea filtrar las columnas ",
                                                html.Strong("Numéricas"),
                                                " deberá ingresar los símbolos (",
                                                html.Strong(">"),
                                                ") - (",
                                                html.Strong("<"),
                                                ") antes del número.",
                                            ]
                                        ),
                                    ],
                                    className="personalized-tooltip",
                                    target="dataset-info",
                                ),
                            ],
                            className="title-hint-container",
                        ),
                        html.Div(
                            id="modified-dataset-view", style={"overflow": "scroll"}
                        ),
                        html.Div(id="test-tert"),
                    ]
                ),
                html.Div(
                    [
                        html.Plaintext(
                            "Conjunto de Datos Original",
                            className="rules-title",
                        ),
                        html.Div(id="dataset-view", style={"overflow": "scroll"}),
                    ]
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
                        html.Plaintext(
                            "Variables Numéricas",
                            className="rules-title",
                        ),
                        dbc.Row(id="numeric-plot"),
                        html.Plaintext(
                            "Variables Objeto",
                            className="rules-title",
                        ),
                        dbc.Row(id="object-plot"),
                        html.Div(
                            [
                                html.Plaintext(
                                    "Correlación de Variables",
                                    className="rules-title",
                                ),
                                html.I(
                                    id="correlation-info",
                                    className="fa fa-info-circle info-icon",
                                ),
                                dbc.Tooltip(
                                    [
                                        html.Plaintext(
                                            [
                                                "Correlación: Medida de la relación entre dos variables, que varía de -1 (correlación negativa) a 1 (correlación positiva). ",
                                                html.Strong("Valores positivos"),
                                                " indican que las variables aumentan o disminuyen juntas, mientras que ",
                                                html.Strong("valores negativos"),
                                                " indican que una variable aumenta cuando la otra disminuye. Valores ",
                                                html.Strong("cercanos a 0"),
                                                " indican poca o ninguna correlación.",
                                            ]
                                        ),
                                    ],
                                    className="personalized-tooltip",
                                    target="correlation-info",
                                ),
                            ],
                            className="title-hint-container",
                        ),
                        html.Div(id="correlation-plot"),
                    ],
                ),
            ]
        )
    ],
    className="section-content",
    style={"margin": "auto"},
)


def datasetCallbacks(app, furl: Function):
    @app.callback(
        Output("dataset-title", "children"),
        Output("modified-dataset-view", "children"),
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
            classifier_model: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.explainer_model_id == model_id
            ).first()
            
            model_x = classifier_model.explainer_model

            original_df: pd.DataFrame = model_x.data_set_data.getElement("dataset")
            original_df_with_index = original_df.rename_axis("Índice").reset_index()
            df: pd.DataFrame = model_x.data_set_data.getElement("dataset_modified")
            df_with_index = df.rename_axis("Índice").reset_index()
            dtt = model_x.getElement("name")
            qualitative_graphs_array, numeric_graphs_array = (
                generateDataSetDistributions(df)
            )
            corr_matrix = original_df.drop(
                columns=model_x.getElement("target_row")
            ).corr(method="pearson")

            def setBottomLegend(fig):
                fig.update_layout(
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.3, xanchor="right", x=1
                    )
                )
                return fig

            return (
                dtt,
                html.Div(
                    [
                        dash_table.DataTable(
                            data=df_with_index.to_dict("records"),
                            columns=[
                                {"name": i, "id": i} for i in df_with_index.columns
                            ],
                            page_size=10,
                            filter_action="native",
                            filter_options={"placeholder_text": "Filtrar..."},
                            sort_action="native",
                            sort_mode="multi",
                            row_selectable="single",
                        )
                    ],
                    className="rules-table",
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            data=original_df_with_index.to_dict("records"),
                            columns=[
                                {"name": i, "id": i}
                                for i in original_df_with_index.columns
                            ],
                            page_size=10,
                            filter_action="native",
                            filter_options={"placeholder_text": "Filtrar..."},
                            sort_action="native",
                            sort_mode="multi",
                            row_selectable="single",
                        )
                    ],
                    className="rules-table",
                ),
                [
                    dbc.Col(
                        id=f"contribution_graph_{data["predictor"]}",
                        children=dcc.Graph(
                            figure=setBottomLegend(
                                go.Figure(
                                    data=data["graph_data"],
                                    layout=dict(title=data["predictor"]),
                                )
                            )
                        ),
                        xs=12,
                        sm=12,
                        md=6,
                        lg=6,
                        xl=6,
                        xxl=6,
                    )
                    for data in numeric_graphs_array
                ],
                [
                    dbc.Col(
                        id=f"contribution_graph_{data["predictor"]}",
                        children=dcc.Graph(
                            figure=setBottomLegend(
                                go.Figure(
                                    data=data["graph_data"],
                                    layout=dict(title=data["predictor"]),
                                )
                            )
                        ),
                        xs=12,
                        sm=12,
                        md=12,
                        lg=6,
                        xl=6,
                        xxl=6,
                    )
                    for data in qualitative_graphs_array
                ],
                dcc.Graph(
                    figure=setBottomLegend(
                        go.Figure(
                            data=go.Heatmap(
                                z=corr_matrix,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                text=round(corr_matrix, 2),
                                texttemplate="%{text}",
                            ),
                        )
                    )
                ),
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
