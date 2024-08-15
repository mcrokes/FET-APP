from ctypes import Array
import threading
import time

from app.proccessor.models import ExplainedClassifierModel, ExplainedRegressorModel, ExplainedModel
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required, current_user

from ..base.models import User


@blueprint.route("/users/list", methods=["GET", "POST"])
@login_required
def get_Users():
    return {"data": [user.to_dict() for user in User.get_users_list()]}


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
    classifiers: Array[ExplainedClassifierModel] = ExplainedClassifierModel.query.filter(
        ExplainedClassifierModel.user_id == current_user.id).all()
    response = []
    for classifier in classifiers:
        response.append(classifier.to_dict())
    return {"data": response}


@blueprint.route("/regression/list", methods=["GET", "POST"])
@login_required
def get_regressor_list():
    regressors: Array[ExplainedRegressorModel] = ExplainedRegressorModel.query.filter(
        ExplainedRegressorModel.user_id == current_user.id).all()
    response = []
    for regressor in regressors:
        response.append(regressor.to_dict())
    return {"data": response}


@blueprint.route("/classifier/namelist", methods=["GET", "POST"])
@login_required
def get_classifier_namelist():
    classifiers: Array[ExplainedClassifierModel] = ExplainedClassifierModel.query.all()
    nameList = []
    idPath = {}
    for classifier in classifiers:
        nameList.append(classifier.name)
        idPath[f'{classifier.explainer_model.id}'] = classifier.name
    return {"data": nameList, "idPath": idPath }


@blueprint.route("/regressor/namelist", methods=["GET", "POST"])
@login_required
def get_regressor_namelist():
    regressors: Array[ExplainedRegressorModel] = ExplainedRegressorModel.query.all()
    nameList = []
    idPath = {}
    for regressor in regressors:
        nameList.append(regressor.name)
        idPath[f'{regressor.explainer_model.id}'] = regressor.name
    return {"data": nameList}
