import json
import os
from ctypes import Array

from app.proccessor.models import ExplainedClassifierModel, ExplainedRegressorModel, ExplainedModel
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required, current_user, AnonymousUserMixin

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


@blueprint.route("/classifier/delete", methods=["POST"])
@login_required
def delete_classifier():
    modelId = request.data.decode()
    classifier: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
        ExplainedClassifierModel.explainer_model_id == modelId
    ).first()
    try:
        classifier.delete_from_db()
        return {'status': 200, 'statusText': 'OK'}
    except:
        return {'status': 500, 'statusText': 'error on deletion'}


@blueprint.route("/regressor/delete", methods=["POST"])
@login_required
def delete_regressor():
    modelId = request.data.decode()
    regressor: ExplainedRegressorModel = ExplainedRegressorModel.query.filter(
        ExplainedRegressorModel.explainer_model_id == modelId).first()
    try:
        regressor.delete_from_db()
        return {'status': 200, 'statusText': 'OK'}
    except:
        return {'status': 500, 'statusText': 'error on deletion'}


@blueprint.route("/classifier/namelist", methods=["GET", "POST"])
@login_required
def get_classifier_namelist():
    classifiers: Array[ExplainedClassifierModel] = ExplainedClassifierModel.query.all()
    nameList = []
    idPath = {}
    for classifier in classifiers:
        nameList.append(classifier.name)
        idPath[f'{classifier.explainer_model_id}'] = classifier.name
    return {"data": nameList, "idPath": idPath}


@blueprint.route("/regressor/namelist", methods=["GET", "POST"])
@login_required
def get_regressor_namelist():
    regressors: Array[ExplainedRegressorModel] = ExplainedRegressorModel.query.all()
    nameList = []
    idPath = {}
    for regressor in regressors:
        nameList.append(regressor.name)
        idPath[f'{regressor.explainer_model_id}'] = regressor.name
    return {"data": nameList, "idPath": idPath}


def find_translations(current_language, keys):
    # Obtener la ruta actual de trabajo
    ruta_actual = os.getcwd()
    ruta_archivo = os.path.join(ruta_actual, f'app/base/static/languages/{current_language}.json')
    text = ''
    try:
        with open(ruta_archivo) as archivo:
            # Cargar el contenido del archivo en una variable
            datos = json.load(archivo)
            # print(datos)
            text = datos
            for key in keys:
                text = text[key]
    except Exception as e:
        print('Translation key error: ', e)
        return {"text": {}}

    return {"text": text}

@blueprint.route("/getTranslation", methods=["GET", "POST"])
def get_translation():
    requestData = json.loads(request.data.decode())
    print('requestData: ', requestData)
    keys = requestData["keys"]
    lang = requestData["lang"]
    isLogged = not isinstance(current_user, AnonymousUserMixin)

    default: dict = {'langSelection': lang if lang in ["es", "en", "ru"] else "es"}
    if isLogged:
        user: User = User.query.filter(User.id == current_user.id).first()
        current_language = user.langSelection if user.langSelection else default["langSelection"]
    else:
        current_language = default["langSelection"]

    # print('keys: ', keys)
    # Construir la ruta del archivo
    return find_translations(current_language, keys)




@blueprint.route("/changeLanguage", methods=["GET", "POST"])
@login_required
def change_language():
    lang = request.data.decode()
    user: User = User.query.filter(User.id == current_user.id).first()
    try:
        user.langSelection = lang
        user.db_commit()
        return {"status": 'ok'}
    except Exception as e:
        print('Error on setting language: ', e)
        user.langSelection = 'es'
        user.db_commit()
        return {"status": "reset"}