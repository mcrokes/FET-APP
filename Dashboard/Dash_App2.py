# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 10:39:33 2018

@author: mcrokes
"""
from dash import Dash, html

from Dashboard.dataset.datasetLayout import datasetCallbacks, datasetLayout
from Dashboard.importances.importancesLayout import importancesCallbacks, importancesLayout
from Dashboard.metrics.metricsLayout import metricsCallbacks, metricsLayout
from Dashboard.surrogate.surrogateLayout import surrogateCallbacks, surrogateLayout

from .Dash_fun import apply_layout_with_auth, load_object, save_object
import dash_bootstrap_components as dbc
from furl import furl

url_base = "/dash/app2/"

holder = html.Plaintext("No se ha insertado ninugún modelo")

tab0_content = dbc.Card(
    dbc.CardBody([html.Div([datasetLayout], id="dataset-layout-output-upload")]),
    className="mt-3 section-card",
)

tab1_content = dbc.Card(
    dbc.CardBody([html.Div([importancesLayout], id="importance-layout-output-upload")]),
    className="mt-3 section-card",
)

tab2_content = dbc.Card(
    dbc.CardBody([html.Div([metricsLayout], id="graph-metrics-layout-output-upload")]),
    className="mt-3 section-card",
)

tab3_content = dbc.Card(
    dbc.CardBody([html.Div([surrogateLayout], id="surrogate-layout-output-upload")]),
    className="mt-3 section-card",
)

tab4_content = dbc.Card(
    dbc.CardBody([html.Div([holder], id="tryit-yourself-layout-output-upload")]),
    className="mt-3 section-card",
)

tabs = dbc.Tabs(
    [
        dbc.Tab(
            tab0_content,
            label="Dataset",
            className="classifier-tab",
        ),
        dbc.Tab(
            tab1_content,
            label="Importancias",
            className="classifier-tab",
        ),
        dbc.Tab(
            tab2_content,
            label="Métricas",
            className="classifier-tab",
        ),
        dbc.Tab(
            tab3_content,
            label="Subrogado",
            className="classifier-tab",
        ),
        dbc.Tab(
            tab4_content,
            label="Predicciones",
            className="classifier-tab",
        ),
    ],
    id="classifier-tabs",
)

layout = html.Div(
    [tabs],
    id="classifier-tabs-container",
    style={"width": "100%"},
)




def Add_Dash(server):
    app = Dash(
        server=server,
        url_base_pathname=url_base,
        external_stylesheets=[
            "/static/assets/CYBORG/bootstrap.min.css",
            "/static/assets/fontawesome-free/css/all.min.css",
            "/static/assets/styles.css",
        ],
    )
    apply_layout_with_auth(app, layout)
    datasetCallbacks(app, furl)
    importancesCallbacks(app, furl)
    metricsCallbacks(app, furl)
    surrogateCallbacks(app, furl)

    return app.server
