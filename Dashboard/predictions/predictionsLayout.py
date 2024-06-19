from pyclbr import Function
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.models import ModelForProccess

from treeinterpreter import treeinterpreter as ti
   

def getTreeInterpreterParamethers(n, data, model: RandomForestClassifier | DecisionTreeClassifier, class_names, positive_class: int):
    if n >= 1 and n <= len(data):
        instance: pd.DataFrame = data[n-1:n]
        df_dict = {
            ("Instance", "Predictor"): [],
            ("Instance", "Value"): [],
        }
        for feature in instance:
            df_dict[("Instance", "Predictor")].append(feature)
            df_dict[("Instance", "Value")].append(pd.Series(instance[feature]).values[0])
        
        
        prediction, bias, contributions = ti.predict(model, instance)        
        for index, class_name in enumerate(class_names):
            df_dict[("Contribution" ,class_name)] = [contribution[index] for contribution in contributions[0]]
            
        print(df_dict)
                
                
        interpretation = {
            "prediction": prediction[0][positive_class],
            "bias": bias[0][positive_class], # (trainset mean)
            "contributions": []
        }
        for c, feature in sorted(
            zip(contributions[0], model.feature_names_in_),
            key=lambda x: -x[0][positive_class],
        ):
            interpretation["contributions"].append({
                "predictor": feature,
                "contribution": c[positive_class]
            })
        return interpretation, df_dict
    else:
        return None


predictionsLayout = html.Div(
    [
        dcc.Location(id='path'),
        dcc.Loading(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        html.H3(
                                            ["DATASET ", html.Span(id="predictions-title")],
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
            ]
        )
    ],
    className="section-content",
    style={"padding-left": "30px", "padding-right": "30px", "margin": "auto"},
)

def predictionsCallbacks(app, furl:Function):    
    @app.callback(
        Output("predictions-title", "children"),
        Output("predictions-view", "children"),
        Input("path", "href"),
    )
    def graph_explainers(cl):
        f = furl(cl)
        param1 = f.args["model_id"]
        try:
            model_x: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == param1
            ).first()
            ds = model_x.getElement("dataset")
            interpretation, general_dict = getTreeInterpreterParamethers(
                data=ds.drop(columns=model_x.getElement("target_row")),
                class_names=["m", "v"],
                model=model_x.getElement("model"),
                n=20,
                positive_class = 0
            )
            df = pd.DataFrame(general_dict)
            dtt = "Title"
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
                                    *enumerate([list(x.items()) for x in df.T.to_dict().values()])
                                ]
                            ],
                            columns=[{"name": [i, j], "id": f"{i}_{j}"} for i, j in df.columns],
                            page_size=10,
                            merge_duplicate_headers=True,
                        ),
                    ]
                ),
            )
        except Exception as e:
            print(e)
            raise PreventUpdate
