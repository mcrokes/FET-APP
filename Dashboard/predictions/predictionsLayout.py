from pyclbr import Function
from re import M
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.model.values import get_target_dropdown
from app.proccessor.models import ExplainedClassifierModel

from treeinterpreter import treeinterpreter as ti


def getTreeInterpreterParamethers(
    instance,
    model: RandomForestClassifier | DecisionTreeClassifier,
    class_names,
    current_class,
):
# TODO add initial marker
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
        if index == current_class:
            contribution_graph_data.append({"class_name": class_name, "graph_data": []})
        general_dict[("Contribution", class_name)] = []
        bar_base = bias[0][index]
        media_array_x = ["Acumulado"]
        media_array_y = [bias[0][index]]
        sorted_contributions = sorted(
            zip(contributions[0], instance),
            key=lambda x: -max(x[0]),
        )

        for jIndex, (contribution, feature) in enumerate(sorted_contributions):
            if feature not in general_dict[("Instance", "Predictor")]:
                general_dict[("Instance", "Predictor")].append(feature)
                general_dict[("Instance", "Value")].append(
                    pd.Series(instance[feature]).values[0]
                )
            general_dict[("Contribution", class_name)].append(
                f"{round(contribution[index],3)} ({round(contribution[index]*100,1)}%)"
            )
            bar_base += contribution[index]
            if index == current_class:
                x = ["Acumulado"]
                y = [contribution[index]]

                contribution_graph_data[0]["graph_data"].append(
                    go.Bar(name=feature, x=x, y=y)
                )

        general_dict[("Contribution", class_name)].append(
            f"{round(bias[0][index],3)} ({round(bias[0][index]*100,1)}%)"
        )
        if index == current_class:
            contribution_graph_data[0]["graph_data"].insert(
                0, go.Bar(name="Media", x=media_array_x, y=media_array_y)
            )
            contribution_graph_data[0]["graph_data"].append(
                go.Bar(
                    name="Prediction", x=["Predicción Final"], y=[prediction[0][index]]
                )
            )

    general_dict[("Instance", "Predictor")].append("Media Poblacional")
    general_dict[("Instance", "Value")].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


def getIndividualPredictions(model, class_names, instance, cut_point, current_class):
    figures = []
    index = current_class
    class_name = class_names[index]
    # for index, class_name in enumerate(class_names):
    individual_predictions = [
        estimator.predict_proba(instance)[0] for estimator in model.estimators_
    ]

    sorted_predictions = np.array(
        sorted(individual_predictions, key=lambda x: -x[index])
    )
    predictions_for_actual_clase = sorted_predictions[:, index]
    x = list(range(len(predictions_for_actual_clase) + 1))[1:]
    y = np.round(predictions_for_actual_clase * 100, 2)
    data = [
        go.Bar(
            name="Árboles",
            y=y,
            x=x,
            marker_color=["blue" if val > cut_point else "red" for val in y],
        )
    ]
    data.append(
        go.Scatter(
            name="Punto de Corte",
            x=[0, x[-1]],
            y=[cut_point, cut_point],
            mode="lines",
            line=dict(dash="dash"),
        )
    )
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Predicción individual por árboles  para clase {class_name}",
        xaxis_title="Número de Árbol",
        yaxis_title="Certeza de Predicción %",
        bargap=0.1,
    )
    figures.append({"class_name": class_name, "graph_data": fig})

    return figures


class_selector = dcc.Dropdown(
    id="prediction-positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    clearable=False,
)

marks = {}
for n in range(11):
    if n == 0:
        marks["1"] = 1
    else:
        marks[f"{n*10}"] = f"{n*10}%"
slider = dcc.Slider(
    min=1, max=100, step=1, marks=marks, value=50, id="trees-cutoff-slider"
)

predictionsLayout = html.Div(
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
                                        html.Span(
                                            id="predictions-title",
                                            style={
                                                "color": "black",
                                                "font-size": "18px",
                                            },
                                        ),
                                    ],
                                    style={"text-align": "center"},
                                ),
                                html.Div(id="predictions-view"),
                            ]
                        )
                    ],
                    style={"margin": "auto"},
                    xs=12,
                    sm=12,
                    md=7,
                    lg=7,
                    xl=7,
                    xxl=7,
                ),
                dbc.Col(
                    html.Div(id="predictions-output-upload"),
                    style={"margin": "auto"},
                    xs=12,
                    sm=12,
                    md=5,
                    lg=5,
                    xl=5,
                    xxl=5,
                ),
            ],
            style={"padding-top": "20px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="predictions-class_selector-container",
                            children=[class_selector],
                            hidden=True,
                        ),
                    ],
                    xs=4,
                    sm=4,
                    md=4,
                    lg=4,
                    xl=4,
                    xxl=4,
                )
            ],
            style={"padding-top": "20px"},
        ),
        html.Div(id="contributions-output-upload"),
        html.Div(id="test"),
        html.Div(id="trees-output-upload"),
        html.Div(slider, id="trees-slider-container", hidden=True),
    ],
    className="section-content",
    style={"margin": "auto"},
)


def predictionsCallbacks(app, furl: Function):
    @app.callback(
        # Output("test", "children"),
        Output("test_graph", "figure"),
        State("test_graph", "figure"),
        Input("test_graph", "restyleData"),
        prevent_initial_call=True,
    )
    def update_point(figure, restyleData):
        if restyleData and (restyleData[1][-1] != len(figure["data"])-1 or restyleData[0]['visible'][restyleData[1][-1]]):
            def isDisabled(data):
                if data["name"].upper().find("PREDICTION") == 0:
                    return True
                elif data["name"].upper().find("TPR", 0) == 0:
                    figure["data"].remove(data)
                    return True
                elif data.get("visible") == "legendonly":                    
                    return True
                return False
            
            point = sum(data["y"][0] if not isDisabled(data) else 0 for data in figure["data"])
            trace = go.Scatter(
                    x=["Acumulado", "Predicción Final"],
                    y=[point, point],
                    mode="lines",
                    name=f"TPR {round(point * 100, 2)} %",
                    line=dict(dash="dash"),
                )
            figure["data"].append(trace)
            return figure
        else: 
            raise PreventUpdate

    @app.callback(
        Output("instances-dropdown", "children"),
        Output("trees-cutoff-slider", "value"),
        Output("prediction-positive-class-selector", "options"),
        Output("prediction-positive-class-selector", "value"),
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
            target_description = model_x.getElement("target_names_dict")
            options = []
            for index, _ in x_test.iterrows():
                options.append({"label": index, "value": index})

            drop_down = dbc.Select(
                id="select",
                options=options,
            )
            class_names = [var["new_value"] for var in target_description["variables"]]
            slider_initial_value = 100 / len(class_names) + 1

            return (
                drop_down,
                slider_initial_value,
                get_target_dropdown(target_description["variables"]),
                target_description["variables"][0]["old_value"],
            )
        except Exception as e:
            print(e)
            raise PreventUpdate

    @app.callback(
        Output("trees-output-upload", "children"),
        State("path", "href"),
        Input("select", "value"),
        Input("trees-cutoff-slider", "value"),
        Input("prediction-positive-class-selector", "value"),
    )
    def graph_trees_predictions(cl, n, cut_point, positive_class):
        print(positive_class)
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

                individual_predictions_graph = getIndividualPredictions(
                    current_class=positive_class,
                    class_names=[
                        var["new_value"]
                        for var in model_x.getElement("target_names_dict")["variables"]
                    ],
                    instance=instance,
                    model=model_x.getElement("model"),
                    cut_point=cut_point,
                )

                return [
                    html.Div(
                        id=f"contribution_graph_{data["class_name"]}",
                        children=dcc.Graph(
                            figure=go.Figure(
                                data["graph_data"], layout=dict(barmode="stack")
                            )
                        ),
                    )
                    for data in individual_predictions_graph
                ]

            except Exception as e:
                print(e)
        raise PreventUpdate

    @app.callback(
        Output("predictions-title", "children"),
        Output("predictions-view", "children"),
        Output("contributions-output-upload", "children"),
        Output("predictions-output-upload", "children"),
        Output("trees-slider-container", "hidden"),
        Output("predictions-class_selector-container", "hidden"),
        State("path", "href"),
        Input("select", "value"),
        Input("prediction-positive-class-selector", "value"),
    )
    def graph_explainers(cl, n, positive_class):
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
                        current_class=positive_class,
                        instance=instance,
                        class_names=[
                            var["new_value"]
                            for var in model_x.getElement("target_names_dict")[
                                "variables"
                            ]
                        ],
                        model=model_x.getElement("model"),
                    )
                )

                df = pd.DataFrame(general_dict)
                dtt = "Contribución de Cada Predictor por Clase"
                pie_chart = go.Figure(
                    data=[
                        go.Pie(
                            labels=predictions_graph_data["labels"],
                            values=predictions_graph_data["values"],
                            hole=0.3,
                        )
                    ]
                )

                def getFigure(fig):
                    return fig

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
                                style_header={
                                    "font-size": "16px",
                                    "font-weight": "bold",
                                    "text-align": "center",
                                },
                                style_data={
                                    "whiteSpace": "normal",
                                    "font-size": "14px",
                                    "text-align": "center",
                                },
                                fill_width=True,
                                style_table={"overflow": "scroll"},
                            ),
                        ]
                    ),
                    [
                        html.Div(
                            id=f"contribution_graph_{data["class_name"]}",
                            children=dcc.Graph(
                                id="test_graph",
                                figure=getFigure(
                                    go.Figure(
                                        data["graph_data"], layout=dict(barmode="stack")
                                    )
                                ),
                                config={"scrollZoom": True},
                            ),
                        )
                        for data in contribution_graph_data
                    ],
                    dcc.Graph(figure=pie_chart),
                    False,
                    False,
                )
            except Exception as e:
                print(e)
        raise PreventUpdate
