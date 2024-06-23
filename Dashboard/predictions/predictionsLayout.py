from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.models import ExplainedClassifierModel, ModelForProccess

from treeinterpreter import treeinterpreter as ti


def getTreeInterpreterParamethers(
    instance, model: RandomForestClassifier | DecisionTreeClassifier, class_names
):

    general_dict = {
        ("Instance", "Predictor"): [],
        ("Instance", "Value"): [],
    }
    contribution_graph_data = []

    prediction, bias, contributions = ti.predict(model, instance)
    predictions_graph_data = {
        "labels": class_names,
        "values": prediction[0],
    }

    for index, class_name in enumerate(class_names):
        contribution_graph_data.append({"class_name": class_name, "graph_data": []})
        general_dict[("Contribution", class_name)] = []
        bar_base = bias[0][index]
        media_array_x = ["Media Poblacional"]
        media_array_y = [bias[0][index]]
        sorted_contributions = sorted(
            zip(contributions[0], instance),
            key=lambda x: -max(x[0]),
        )
        for jIndex, (contribution, feature) in enumerate(sorted_contributions):
            media_array_x.append(feature)
            media_array_y.append(bar_base)
            if feature not in general_dict[("Instance", "Predictor")]:
                general_dict[("Instance", "Predictor")].append(feature)
                general_dict[("Instance", "Value")].append(
                    pd.Series(instance[feature]).values[0]
                )
            general_dict[("Contribution", class_name)].append(contribution[index])
            x = [feature]
            y = [feature]
            for sorted_contribution in sorted_contributions[jIndex:]:
                x.append(sorted_contribution[1])
                y.append(contribution[index])
            contribution_graph_data[index]["graph_data"].append(
                go.Bar(name=feature, x=x, y=y)
            )
        general_dict[("Contribution", class_name)].append(bias[0][index])
        contribution_graph_data[index]["graph_data"].insert(
            0, go.Bar(name="Media", x=media_array_x, y=media_array_y)
        )
        contribution_graph_data[index]["graph_data"].append(
            go.Bar(name="Prediction", x=["Predicción Final"], y=[prediction[0][index]])
        )

    general_dict[("Instance", "Predictor")].append("Media Poblacional")
    general_dict[("Instance", "Value")].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


predictionsLayout = html.Div(
    [
        dcc.Loading(
            [
                dbc.Row(id="instances-dropdown"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.H3(
                                            [
                                                "DATASET ",
                                                html.Span(id="predictions-title"),
                                            ],
                                            style={"text-align": "center"},
                                        ),
                                        html.Div(id="predictions-view"),
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
                                html.Div(id="predictions-description"),
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
                html.Div(id="contributions-output-upload"),
                html.Div(id="predictions-output-upload"),
            ]
        )
    ],
    className="section-content",
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)


def predictionsCallbacks(app, furl: Function):
    @app.callback(
        Output("instances-dropdown", "children"),
        Input("path", "href"),
    )
    def load_data(cl):
        f = furl(cl)
        model_id = f.args["model_id"]
        try:
            model_x: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                ExplainedClassifierModel.id == model_id
            ).first()
            ds = model_x.data_set_data.getElement("dataset")
            x_test = ds.drop(columns=model_x.getElement("target_row"))
            options = []
            for index, _ in x_test.iterrows():
                options.append({"label": index, "value": index})

            drop_down = dbc.Select(
                id="select",
                options=options,
            )

            return drop_down
        except Exception as e:
            print(e)
            raise PreventUpdate

    @app.callback(
        Output("predictions-title", "children"),
        Output("predictions-view", "children"),
        Output("contributions-output-upload", "children"),
        Output("predictions-output-upload", "children"),
        State("path", "href"),
        Input("select", "value"),
    )
    def graph_explainers(cl, n):
        if n:
            n = int(n)
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                model_x: ExplainedClassifierModel = (
                    ExplainedClassifierModel.query.filter(
                        ExplainedClassifierModel.id == model_id
                    ).first()
                )
                ds = model_x.data_set_data.getElement("dataset")
                x_test = ds.drop(columns=model_x.getElement("target_row"))

                instance: pd.DataFrame = (
                    x_test[n - 1 : n] if n >= 1 and n <= len(x_test) else x_test[-1:]
                )
                contribution_graph_data, general_dict, predictions_graph_data = (
                    getTreeInterpreterParamethers(
                        instance=instance,
                        class_names=["m", "v"],
                        model=model_x.getElement("model"),
                    )
                )
                df = pd.DataFrame(general_dict)
                dtt = "Title"
                pie_chart = go.Figure(
                    data=[
                        go.Pie(
                            labels=predictions_graph_data["labels"],
                            values=predictions_graph_data["values"],
                            hole=0.3,
                        )
                    ]
                )
                return (
                    dtt,
                    html.Div(
                        [
                            dash_table.DataTable(
                                data=[
                                    {
                                        **{f"{x1}_{x2}": y for (x1, x2), y in data},
                                    }
                                    for (n, data) in [
                                        *enumerate(
                                            [
                                                list(x.items())
                                                for x in df.T.to_dict().values()
                                            ]
                                        )
                                    ]
                                ],
                                columns=[
                                    {"name": [i, j], "id": f"{i}_{j}"}
                                    for i, j in df.columns
                                ],
                                page_size=10,
                                merge_duplicate_headers=True,
                            ),
                        ]
                    ),
                    [
                        html.Div(
                            id=f"contribution_graph_{data["class_name"]}",
                            children=dcc.Graph(
                                figure=go.Figure(
                                    data["graph_data"], layout=dict(barmode="stack")
                                )
                            ),
                        )
                        for data in contribution_graph_data
                    ],
                    dcc.Graph(figure=pie_chart),
                )
            except Exception as e:
                print(e)
        raise PreventUpdate
