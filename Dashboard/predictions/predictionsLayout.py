import json
import pickle
from pyclbr import Function
from re import M
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Dashboard.metrics.metricsLayout import get_target_dropdown
from app.proccessor.models import ExplainedClassifierModel, ExplainedModel

from treeinterpreter import treeinterpreter as ti


def setBottomLegend(fig):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.5, xanchor="right", x=1)
    )
    return fig


def getTreeInterpreterParamethersRegressor(
        instance,
        instanceModified,
        model: RandomForestClassifier | DecisionTreeClassifier,
):
    general_dict = {
        ("Instance", "Predictor"): [],
        ("Instance", "Value"): [],
    }
    contribution_graph_data = []

    prediction, bias, contributions = ti.predict(model, instance)
    predictions_graph_data = {
        "labels": None,
        "values": prediction[0],
    }

    point = 0

    general_dict[("Contribution", "Contribution")] = []
    bar_base = bias[0]
    print('bias: ', bar_base)
    print('contributions: ', list(contributions[0]))
    print('prediction: ', list(prediction[0]))
    media_array_x = ["Acumulado"]
    media_array_y = [bias[0]]
    sorted_contributions = sorted(zip(list(contributions[0]), instance)
                                  )
    print('sorted_contributions: ', sorted_contributions)
    contribution_graph_data.append({"graph_data": []})
    point += bar_base

    for jIndex, (contribution, feature) in enumerate(sorted_contributions):
        if feature not in general_dict[("Instance", "Predictor")]:
            general_dict[("Instance", "Predictor")].append(feature)
            general_dict[("Instance", "Value")].append(instanceModified[feature])
        general_dict[("Contribution", "Contribution")].append(
            f"{round(contribution, 3)}"
        )
        bar_base += contribution
        x = ["Acumulado"]
        y = [contribution]
        print('general_dict: ', general_dict)

        contribution_graph_data[0]["graph_data"].append(
            go.Bar(
                name=f"{feature}({round(y[0], 3)}) {round(y[0] / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100, 2)}%",
                x=x,
                y=y,
            )
        )
        point += contribution

    general_dict[("Contribution", "Contribution")].append(
        f"{round(bias[0], 3)}"
    )

    contribution_graph_data[0]["graph_data"].insert(
        0,
        go.Bar(
            name=f"Media ({round(media_array_y[0], 3)}) {round(media_array_y[0] / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100, 3)}%",
            x=media_array_x,
            y=media_array_y,
        ),
    )
    contribution_graph_data[0]["graph_data"].append(
        go.Bar(
            name=f"Prediction ({round(prediction[0][0], 3)}) 100%",
            x=["Predicción Final"],
            y=[prediction[0][0]],
        )
    )
    contribution_graph_data[0]["graph_data"].append(
        go.Scatter(
            x=["Acumulado", "Predicción Final"],
            y=[point, point],
            mode="lines",
            name=f"Actual ({round(point, 3)}) {round((point / (prediction[0][0] if prediction[0][0] > 0 else 1) * 100), 2)} %",
            line=dict(dash="dash"),
            marker_color=["blue", "blue"],
        )
    )

    general_dict[("Instance", "Predictor")].append("Media Poblacional")
    general_dict[("Instance", "Value")].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


def getTreeInterpreterParamethersClassifier(
        instance,
        instanceModified,
        model: RandomForestClassifier | DecisionTreeClassifier,
        class_names,
        current_class,
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

    point = 0

    for index, class_name in enumerate(class_names):
        general_dict[("Contribution", class_name)] = []
        bar_base = bias[0][index]
        media_array_x = ["Acumulado"]
        media_array_y = [bias[0][index]]
        sorted_contributions = sorted(
            zip(contributions[0], instance),
            key=lambda x: -max(x[0]),
        )
        if index == current_class:
            contribution_graph_data.append({"class_name": class_name, "graph_data": []})
            point += bar_base

        for jIndex, (contribution, feature) in enumerate(sorted_contributions):
            if feature not in general_dict[("Instance", "Predictor")]:
                general_dict[("Instance", "Predictor")].append(feature)
                general_dict[("Instance", "Value")].append(instanceModified[feature])
            general_dict[("Contribution", class_name)].append(
                f"{round(contribution[index], 3)} ({round(contribution[index] * 100, 1)}%)"
            )
            bar_base += contribution[index]
            if index == current_class:
                x = ["Acumulado"]
                y = [contribution[index]]

                contribution_graph_data[0]["graph_data"].append(
                    go.Bar(
                        name=f"{feature}({round(y[0], 3)}) {round(y[0] / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100, 2)}%",
                        x=x,
                        y=y,
                    )
                )
                point += contribution[index]

        general_dict[("Contribution", class_name)].append(
            f"{round(bias[0][index], 3)} ({round(bias[0][index] * 100, 1)}%)"
        )
        if index == current_class:
            contribution_graph_data[0]["graph_data"].insert(
                0,
                go.Bar(
                    name=f"Media ({round(media_array_y[0], 3)}) {round(media_array_y[0] / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100, 3)}%",
                    x=media_array_x,
                    y=media_array_y,
                ),
            )
            contribution_graph_data[0]["graph_data"].append(
                go.Bar(
                    name=f"Prediction ({round(prediction[0][index], 3)}) 100%",
                    x=["Predicción Final"],
                    y=[prediction[0][index]],
                )
            )
            contribution_graph_data[0]["graph_data"].append(
                go.Scatter(
                    x=["Acumulado", "Predicción Final"],
                    y=[point, point],
                    mode="lines",
                    name=f"Actual ({round(point, 3)}) {round((point / (prediction[0][index] if prediction[0][index] > 0 else 1) * 100), 2)} %",
                    line=dict(dash="dash"),
                    marker_color=["blue", "blue"],
                )
            )

    general_dict[("Instance", "Predictor")].append("Media Poblacional")
    general_dict[("Instance", "Value")].append("-")
    return contribution_graph_data, general_dict, predictions_graph_data


def getIndividualPredictionsRegresssor(model: RandomForestRegressor, instance):
    figures = []
    individual_predictions = [
        estimator.predict(instance)[0] for estimator in model.estimators_
    ]

    x = list(range(len(individual_predictions) + 1))
    data = [
        go.Bar(
            name="Árboles",
            y=np.round(sorted(individual_predictions), 2),
            x=x[1:],
        ),
        go.Scatter(
            name="Predicción Real",
            x=[1, x[-1]],
            y=[model.predict(instance)[0], model.predict(instance)[0]],
            mode="lines",
            line=dict(dash="dash"),
        )
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Predicción individual por árboles",
        xaxis_title="Número de Árbol",
        yaxis_title="Certeza de Predicción %",
        bargap=0.1,
    )
    figures.append({"graph_data": fig})

    return figures


def getIndividualPredictionsClassifier(model, class_names, instance, cut_point, current_class):
    figures = []
    index = current_class
    class_name = class_names[index]
    individual_predictions = [
        estimator.predict_proba(instance)[0] for estimator in model.estimators_
    ]

    markers = []
    for val in np.array(individual_predictions)[:, index]:
        if val * 100 >= cut_point:
            markers.append("blue")
        else:
            markers.append("red")

    sorted_predictions = sorted(
        zip(np.array(individual_predictions)[:, index], markers),
        key=lambda x: (-x[0] if markers.count("blue") > markers.count("red") else x[0]),
    )
    predictions_for_actual_clase = np.array(sorted_predictions)[:, 0]
    prev_x = list(range(len(predictions_for_actual_clase) + 1))
    x = (
        prev_x[1:]
        if markers.count("blue") > markers.count("red")
        else prev_x[::-1][:-1]
    )
    y = np.round(predictions_for_actual_clase.astype(np.float64) * 100, 2)

    data = [
        go.Bar(
            name="Árboles",
            y=y,
            x=x,
            marker_color=np.array(sorted_predictions)[:, 1],
        )
    ]
    data.append(
        go.Scatter(
            name="Corte",
            x=[1, x[0]] if markers.count("blue") < markers.count("red") else [x[-1], 1],
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
        xaxis=dict(
            autorange=(
                "reversed" if markers.count("blue") > markers.count("red") else True
            )
        ),
    )
    figures.append({"class_name": class_name, "graph_data": fig})

    return figures


class_selector = dcc.Dropdown(
    id="prediction-positive-class-selector",
    placeholder="Seleccione como positiva la clase que desea analizar",
    clearable=False,
    style={"cursor": "pointer"},
)

marks = {}
for n in range(11):
    if n == 0:
        marks["1"] = 1
    else:
        marks[f"{n * 10}"] = f"{n * 10}%"
slider = dcc.Slider(
    min=1, max=100, step=1, marks=marks, value=50, id="trees-cutoff-slider"
)

id_sufix = ["contributions", "trees-graph"]
predictionsLayout = html.Div(
    [
        dcc.Store(id="current-class-data"),
        dbc.Row(
            [
                dbc.Col(
                    id="instances-dropdown",
                    xs=12,
                    sm=12,
                    md=5,
                    lg=5,
                    xl=5,
                    xxl=5,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.H3(
                                    [
                                        html.Plaintext(
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
                        html.Plaintext(
                            id="predictions-class_selector-title",
                            children=["Seleccione la Clase a Analizar"],
                            style={"color": "black"},
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=12,
                    lg=8,
                    xl=8,
                    xxl=8,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="predictions-class_selector-container",
                            children=[class_selector],
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=12,
                    lg=8,
                    xl=8,
                    xxl=8,
                ),
            ],
            style={"padding-top": "20px"},
        ),
        html.Div(
            [
                html.I(
                    id=f"{id_sufix[0]}-info",
                    className="fa fa-info-circle info-icon",
                ),
                dbc.Tooltip(
                    [
                        html.Plaintext(
                            [
                                "Se examina la ",
                                html.Strong("contribución individual"),
                                "de cada característica a la predicción, lo que permite identificar qué "
                                "características son más relevantes para el modelo.",
                            ]
                        ),
                    ],
                    className="personalized-tooltip",
                    target=f"{id_sufix[0]}-info",
                ),
            ],
            id=f"{id_sufix[0]}-container",
            style={"display": "flex", "justify-content": "end"},
        ),
        html.Div(id="contributions-output-upload"),
        html.Div(id="test"),
        html.Div(
            [
                html.I(
                    id=f"{id_sufix[1]}-info",
                    className="fa fa-info-circle info-icon",
                ),
                dbc.Tooltip(
                    [
                        html.Plaintext(
                            [
                                html.Strong("Cada árbol"),
                                " del modelo hace una predicción. La predicción final se basa en la ",
                                html.Strong("combinación de las predicciones"),
                                " de todos los árboles.",
                            ]
                        ),
                        html.Plaintext(
                            [
                                html.Strong("* El azul"),
                                " representa los arboles que predicen la clase seleccionada como positiva.",
                            ]
                        ),
                        html.Plaintext(
                            [
                                html.Strong("* El rojo"),
                                " representa los arboles que predicen la(s) clases restantes.",
                            ]
                        ),
                    ],
                    className="personalized-tooltip",
                    target=f"{id_sufix[1]}-info",
                ),
            ],
            id=f"{id_sufix[1]}-container",
            style={"display": "flex", "justify-content": "end"},
        ),
        html.Div(id="trees-output-upload"),
        html.Div(slider, id="trees-slider-container", hidden=True),
    ],
    className="section-content",
    style={"margin": "auto"},
)


def predictionsCallbacks(app, furl: Function, isRegressor: bool = False):
    @app.callback(
        Output("test_graph", "figure"),
        State("current-class-data", "data"),
        State("test_graph", "figure"),
        Input("test_graph", "restyleData"),
        prevent_initial_call=True,
    )
    def update_point(data, figure, restyleData):
        if restyleData and (
                restyleData[1][-1] != len(figure["data"]) - 1
                or restyleData[0]["visible"][restyleData[1][-1]]
        ):

            def isDisabled(inner_data):
                if inner_data["name"].upper().find("PREDICTION") == 0:
                    return True
                elif inner_data["name"].upper().find("ACTUAL", 0) == 0:
                    figure["data"].remove(inner_data)
                    return True
                elif inner_data.get("visible") == "legendonly":
                    return True
                return False

            point = sum(
                data["y"][0] if not isDisabled(data) else 0 for data in figure["data"]
            )
            data = json.loads(data)
            trace = go.Scatter(
                x=["Acumulado", "Predicción Final"],
                y=[point, point],
                mode="lines",
                name=f"Actual ({round(point, 3)}) {round(point / (data['prediction'] if data['prediction'] > 0 or isRegressor else 1) * 100, 2)}%",
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

            model_x: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == model_id
            ).first()

            ds = model_x.data_set_data.getElement("dataset")
            x_test = ds.drop(columns=model_x.getElement("target_row"))
            if isRegressor:
                class_names = None
                target_dropdown = [{'value': 0}]
            else:
                target_description = model_x.explainer_classifier.getElement("target_names_dict")
                class_names = [var["new_value"] for var in target_description["variables"]]
                target_dropdown = get_target_dropdown(target_description["variables"])
            options = []
            for index, _ in x_test.iterrows():
                options.append({"label": index, "value": index})

            drop_down = dcc.Dropdown(
                id="select",
                placeholder="Seleccione la Instancia a Analizar",
                options=options,
            )

            slider_initial_value = 100 / len(class_names) + 1 if not isRegressor else 50

            return (
                drop_down,
                slider_initial_value,
                target_dropdown,
                target_dropdown[0]["value"],
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
        if n:
            n = int(n)
            f = furl(cl)
            model_id = f.args["model_id"]
            try:
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                if isRegressor:
                    class_names = None
                else:
                    target_description = model_x.explainer_classifier.getElement("target_names_dict")
                    class_names = [var["new_value"] for var in target_description["variables"]]

                ds = model_x.data_set_data.getElement("dataset")
                x_test = ds.drop(columns=model_x.getElement("target_row"))
                instance: pd.DataFrame = (
                    x_test[n - 1: n] if 1 <= n <= len(x_test) else x_test[-1:]
                )

                if not isRegressor:
                    individual_predictions_graph = getIndividualPredictionsClassifier(
                        model=model_x.getElement("model"),
                        class_names=class_names,
                        instance=instance,
                        cut_point=cut_point,
                        current_class=positive_class
                    )
                else:
                    individual_predictions_graph = getIndividualPredictionsRegresssor(
                        model=model_x.getElement("model"),
                        instance=instance,
                    )

                return [
                    html.Div(
                        id=f"contribution_graph_{data['class_name'] if class_names is not None else 'regressor'}",
                        children=dcc.Graph(
                            figure=setBottomLegend(
                                go.Figure(
                                    data["graph_data"], layout=dict(barmode="stack")
                                )
                            )
                        ),
                    )
                    for data in individual_predictions_graph
                ]

            except Exception as e:
                print(e)
                raise PreventUpdate
        else:
            return []

    @app.callback(
        Output("predictions-title", "children"),
        Output("predictions-view", "children"),
        Output("contributions-output-upload", "children"),
        Output("predictions-output-upload", "children"),
        Output("trees-slider-container", "hidden"),
        Output("predictions-class_selector-container", "hidden"),
        Output("predictions-class_selector-title", "hidden"),
        Output(f"{id_sufix[0]}-container", "hidden"),
        Output(f"{id_sufix[1]}-container", "hidden"),
        Output("current-class-data", "data"),
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
                model_x: ExplainedModel = ExplainedModel.query.filter(
                    ExplainedModel.id == model_id
                ).first()

                if isRegressor:
                    class_names = None
                else:
                    target_description = model_x.explainer_classifier.getElement("target_names_dict")
                    class_names = [var["new_value"] for var in target_description["variables"]]

                ds = model_x.data_set_data.getElement("dataset")
                dsModified = model_x.data_set_data.getElement("dataset_modified")
                x_test = ds.drop(columns=model_x.getElement("target_row"))
                x_testModified = dsModified.drop(
                    columns=model_x.getElement("target_row")
                )

                instance: pd.DataFrame = (
                    x_test[n - 1: n] if 1 <= n <= len(x_test) else x_test[-1:]
                )
                instanceModified: pd.DataFrame = (
                    x_testModified[n - 1: n]
                    if 1 <= n <= len(x_test)
                    else x_testModified[-1:]
                )
                if isRegressor:
                    contribution_graph_data, general_dict, predictions_graph_data = (
                        getTreeInterpreterParamethersRegressor(
                            model=model_x.getElement("model"),
                            instance=instance,
                            instanceModified=instanceModified,
                        ))
                else:
                    contribution_graph_data, general_dict, predictions_graph_data = (
                        getTreeInterpreterParamethersClassifier(
                            current_class=positive_class,
                            instance=instance,
                            instanceModified=instanceModified,
                            class_names=class_names,
                            model=model_x.getElement("model"),
                        )
                    )

                print('contribution_graph_data: ', contribution_graph_data)
                print('general_dict: ', general_dict)
                print('predictions_graph_data: ', predictions_graph_data)

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
                    fig.update_layout(
                        title=f"Cotribucion individual por predictor{f' para {class_names[positive_class]}' if not isRegressor else ''}",
                        xaxis_title="Influeyentes",
                        yaxis_title="Certeza de Predicción de 0 a 1",
                    )
                    return setBottomLegend(fig)

                data_table = html.Div(
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
                                {"name": [i, j], "id": f"{i}_{j}"} if not isRegressor or i != 'Contribution' else {"name": [i], "id": f"{i}_{j}", "rowspan": 2}
                                for i, j in df.columns
                            ],
                            page_size=10,
                            merge_duplicate_headers=True,
                            style_header={
                                "font-size": "16px",
                                "font-weight": "bold",
                                "text-align": "center",
                                "color": "black",
                            },
                            style_header_conditional=[
                                {
                                    'if': {
                                        'header_index': 1,
                                        'column_id': 'Contribution_Contribution'
                                    },
                                    'backgroundColor': '#0010ff4d'
                                },
                            ],
                            style_data={
                                "whiteSpace": "normal",
                                "font-size": "14px",
                                "text-align": "center",
                                "color": "black",
                            },
                            fill_width=True,
                            style_table={"overflow": "scroll"},
                        ),
                    ]
                ),
                contributions_graph = [
                    html.Div(
                        id=f"contribution_graph_{data['class_name'] if class_names is not None else 'regressor'}",
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
                ]
                print('data_table: ', data_table)
                print('Error is on return')

                return (
                    dtt,
                    data_table,
                    contributions_graph,
                    dcc.Graph(figure=pie_chart) if not isRegressor else
                    html.Plaintext(
                        f"Predicción: {predictions_graph_data['values'][0]}",
                        style={
                            "color": "black",
                            "font-size": "18px",
                        },
                    ),
                    False if not isRegressor else True,
                    False if not isRegressor else True,
                    False if not isRegressor else True,
                    False,
                    False,
                    json.dumps(
                        {"prediction": predictions_graph_data["values"][positive_class] if not isRegressor else
                        predictions_graph_data['values'][0]}
                    ),
                )
            except Exception as e:
                raise e
                # print('error in graph_explainers: ', e)
                # raise PreventUpdate

        else:
            return [], [], [], [], True, True, True, True, True, None
