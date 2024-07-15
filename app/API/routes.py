from ctypes import Array
import threading
import time

from app.proccessor.models import ExplainedClassifierModel, ExplainedRegressorModel, ExplainedModel
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required, current_user
import pandas as pd

import joblib


@blueprint.route("/model/percent", methods=["GET", "POST"])
@login_required
def get_percent():
    db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.data.decode()
    ).first()
    return make_response([{
        "percent": db_model.percent_processed,
        "message": db_model.process_message
    }])

@blueprint.route("/classifier/list", methods=["GET", "POST"])
@login_required
def get_classifier_list():
    print("current_user: ", current_user.id)
    classifiers: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(ExplainedClassifierModel.user_id == current_user.id).all()
    response = []
    for classifier in classifiers:
        model: ExplainedModel = classifier.explainer_model
        response.append({"id": model.id, "name": model.name})
    return make_response(response)

@blueprint.route("/regressor/list", methods=["GET", "POST"])
@login_required
def get_regressor_list():
    regressors: ExplainedRegressorModel = ExplainedRegressorModel.query.filter(ExplainedRegressorModel.user_id == current_user.id).all()
    response = []
    for regressor in regressors:
        model: ExplainedModel = regressor.explainer_model
        response.append({"id": model.id, "name": model.name})
    return make_response(response)