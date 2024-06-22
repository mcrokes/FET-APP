from ctypes import Array
import threading
import time

from app.proccessor.forms import add_classifier
from app.proccessor.models import ExplainedClassifierModel
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required
import pandas as pd

import joblib

@blueprint.route("/classifier/percent", methods=["GET", "POST"])
@login_required
def get_percent():
    print(request.data)
    db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.data.decode()
    ).first()
    return make_response([db_model.percent_processed])

@blueprint.route("/classifier/list", methods=["GET", "POST"])
@login_required
def get_list():
    db_model: ExplainedClassifierModel = ExplainedClassifierModel.query.all()
    # db_model: ModelForProccess = ModelForProccess.query.all()
    response = []
    for model in db_model:
        response.append({"id": model.id, "name": model.name})
    return make_response(response)