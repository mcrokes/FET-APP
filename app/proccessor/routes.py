import threading
from typing import Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.base.models import User
from app.proccessor.forms import add_model
from app.proccessor.model.dataset_interaction_methods import get_modified_dataframe
from app.proccessor.model.explainers.decision_tree_surrogate import ExplainSingleTree
from app.proccessor.models import (
    DataSetData,
    DataSetDataDistribution,
    ExplainedClassifierModel,
    ExplainedModel,
    ExplainedRegressorModel,
    ModelForProccess,
    SurrogateTreeData,
    Tree,
    TreeRule,
    TreeRuleCause,
)
from . import blueprint
from flask import current_app, render_template, request, redirect
from flask_login import current_user, login_required
import pandas as pd

import joblib


def thread_function(model_id, app, user_id, type: Literal["Classifier", "Regressor"], modelId):
    with app:
        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == model_id
        ).first()

        db_model.percent_processed = 10
        db_model.process_message = "Sincronizando datos..."
        db_model.db_commit()

        if type == "Classifier":
            db_model_model: RandomForestClassifier = db_model.getElement("model")
            db_model_classifier_target_description: dict = db_model.getElement(
                "target_description"
            )
            db_model_classifier_target_class_names: list = []
            for variable in db_model_classifier_target_description["variables"]:
                db_model_classifier_target_class_names.append(variable["new_value"])
        else:
            db_model_model: RandomForestRegressor = db_model.getElement("model")
            db_model_regressor_unit: dict = db_model.getElement("unit")

        db_model_dataset: pd.DataFrame = db_model.getElement("dataset")
        db_model_target_row: str = db_model.getElement("target_row")

        db_model_qualitative_columns = db_model.getElement(
            "qualitative_variables_saved"
        )
        db_model_qualitative_column_names: list = []

        for column in db_model_qualitative_columns:
            db_model_qualitative_column_names.append(column["column_name"])

        db_model.percent_processed = 20
        db_model.process_message = "Generando modelo base..."
        db_model.db_commit()

        #### BASE MODEL ####
        if modelId:
            model_data: ExplainedModel = ExplainedModel.query.filter(
                ExplainedModel.id == modelId
            ).first()
            model_data.setElements(
                **{
                    "model": db_model_model,
                    "indexesDict": "dict of indexes",
                    "indexColumnName": "column name for indexes",
                    "model_description": db_model.description,
                    "features_description": db_model.getElement("features_description"),
                    "target_row": db_model.target_row,
                    "q_variables_dict": db_model_qualitative_columns,
                    "test_size": 0.6,
                    "random_state": 123,
                }
            )
        else:
            model_data = ExplainedModel(
                **{
                    "model": db_model_model,
                    "indexesDict": "dict of indexes",
                    "indexColumnName": "column name for indexes",
                    "model_description": db_model.description,
                    "features_description": db_model.getElement("features_description"),
                    "target_row": db_model.target_row,
                    "q_variables_dict": db_model_qualitative_columns,
                    "test_size": 0.6,
                    "random_state": 123,
                }
            )

        db_model.percent_processed = 30
        db_model.process_message = "Cargando metricas del conjunto de datos..."
        db_model.db_commit()

        #### DATASET DATA ####
        if modelId:
            model_data.data_set_data.setElements(
                **{
                    "dataset": db_model_dataset,
                    "dataset_modified": get_modified_dataframe(
                        df=db_model_dataset,
                        target_description=(
                            db_model_classifier_target_description
                            if type == "Classifier"
                            else None
                        ),
                        qualitative_columns=db_model_qualitative_columns,
                    ),
                }
            )
        else:
            dataset_data = DataSetData(
                **{
                    "dataset": db_model_dataset,
                    "dataset_modified": get_modified_dataframe(
                        df=db_model_dataset,
                        target_description=(
                            db_model_classifier_target_description
                            if type == "Classifier"
                            else None
                        ),
                        qualitative_columns=db_model_qualitative_columns,
                    ),
                }
            )
        if modelId:
            model_data.data_set_data.data_set_data_distributions[0].setElements(
                **{
                    "isNumeric": False,
                    "rows_amount": 3,
                    "columns_amount": 3,
                    "distribution_dataset": "Dataset for the distribution type",
                    "isPrime": False,
                }
            )
            model_data.data_set_data.data_set_data_distributions[1].setElements(
                **{
                    "rows_amount": 3,
                    "columns_amount": 3,
                    "distribution_dataset": "Dataset for the distribution type",
                }
            )
            model_data.data_set_data.data_set_data_distributions[0].db_commit()
            model_data.data_set_data.data_set_data_distributions[1].db_commit()
        else:
            dataset_data_distribution_qualitative = DataSetDataDistribution(
                **{
                    "isNumeric": False,
                    "rows_amount": 3,
                    "columns_amount": 3,
                    "distribution_dataset": "Dataset for the distribution type",
                    "isPrime": False,
                }
            )
            dataset_data_distribution_numeric = DataSetDataDistribution(
                **{
                    "rows_amount": 3,
                    "columns_amount": 3,
                    "distribution_dataset": "Dataset for the distribution type",
                }
            )
            dataset_data_distribution_qualitative.data_set_data = dataset_data
            dataset_data_distribution_numeric.data_set_data = dataset_data

        db_model.percent_processed = 40
        db_model.process_message = "Creando modelo subrogado datos..."
        db_model.db_commit()

        tree_depth = 3
        surrogate_inexact_rules_amount = -1
        if modelId:
            model_data.surrogate_trees_data = []
        while surrogate_inexact_rules_amount != 0:

            db_model.process_message = (
                f"Creando modelo subrogado datos (de rofundidad {tree_depth})..."
            )
            db_model.db_commit()
            surrogate_tree = ExplainSingleTree.createSurrogateTree(
                trainedModel=db_model_model,
                x_train=db_model_dataset.drop(columns=db_model_target_row),
                max_depth=tree_depth,
            )
            tree_depth += 1
            rules = ExplainSingleTree.get_rules(
                tree_model=surrogate_tree.tree_,
                q_variables=db_model_qualitative_column_names,
                q_variables_values=db_model_qualitative_columns,
                features=surrogate_tree.feature_names_in_,
                class_names=(
                    db_model_classifier_target_class_names
                    if type == "Classifier"
                    else None
                ),
                model_type=type,
            )

            db_tree = Tree(
                **{
                    "depth": surrogate_tree.get_depth(),
                    "rules_amount": len(rules),
                }
            )

            surrogate_inexact_rules_amount = 0
            for rule in rules:
                db_rule = TreeRule(
                    **{
                        "target_value": rule["target_value"],
                        "probability": rule["probability"],
                        "samples_amount": rule["samples_amount"],
                    }
                )
                db_rule.tree = db_tree
                db_rule.add_to_db()

                for cause in rule["causes"]:
                    db_cause = TreeRuleCause(
                        **{
                            "predictor": cause["item"],
                            "relation_sign": cause["sign"],
                            "value": cause["value"],
                        }
                    )
                    db_cause.tree_rule = db_rule
                    db_cause.add_to_db()

                if rule["probability"] < 100:
                    surrogate_inexact_rules_amount += 1

            db_tree.inexact_rules_amount = surrogate_inexact_rules_amount
            db_tree.add_to_db()

            surrogate_tree_data = SurrogateTreeData(**{"tree_model": surrogate_tree})
            surrogate_tree_data.tree = db_tree
            surrogate_tree_data.explained_model = model_data
            surrogate_tree_data.add_to_db()

        db_model.percent_processed = 80
        db_model.process_message = "Guardando la base de datos del modelo..."
        db_model.db_commit()

        if modelId:
            model_data.data_set_data.db_commit()
            model_data.db_commit()
        else:
            print('Saving Models...')
            model_data.data_set_data = dataset_data
            dataset_data_distribution_numeric.add_to_db()
            dataset_data_distribution_qualitative.add_to_db()
            dataset_data.add_to_db()
            model_data.add_to_db()

        if type == "Classifier":
            if modelId:
                model_data.explainer_classifier.setElements(
                    **{
                        "target_names_dict": db_model_classifier_target_description,
                        "name": db_model.name,
                    }
                )
                model_data.explainer_classifier.db_commit()
            else:
                classifier_model_data = ExplainedClassifierModel(
                    **{
                        "target_names_dict": db_model_classifier_target_description,
                        "name": db_model.name,
                    }
                )
                classifier_model_data.explainer_model = model_data
                classifier_model_data.user = User.query.filter(User.id == user_id).first()
                classifier_model_data.add_to_db()
        else:
            if modelId:
                model_data.explainer_regressor.setElements(
                    **{
                        "unit": db_model_regressor_unit,
                        "name": db_model.name,
                    }
                )
                model_data.explainer_regressor.db_commit()
            else:
                regressor_model_data = ExplainedRegressorModel(
                    **{
                        "unit": db_model_regressor_unit,
                        "name": db_model.name,
                    }
                )
                regressor_model_data.explainer_model = model_data
                regressor_model_data.user = User.query.filter(User.id == user_id).first()
                regressor_model_data.add_to_db()

        db_model.percent_processed = 100
        db_model.process_message = "Completado !!!"
        db_model.db_commit()


@blueprint.route("/add_classifier", methods=["GET", "POST"])
@blueprint.route("/edit_classifier/<modelId>", methods=["GET", "POST"])
@login_required
def save_classifier(modelId: int = 0):
    if modelId:
        print('Entering..')
    form = add_model(request.form)
    if modelId:
        classifier: ExplainedModel = ExplainedModel.query.filter(
            ExplainedModel.id == modelId
        ).first()
        form.name.default = classifier.name
        form.description.default = classifier.model_description
    cancel = None
    try:
        cancel = request.form["cancel"]
    except:  # noqa: E722
        pass

    print(cancel)
    print(request.args)
    print(request.form)

    if cancel == "Second":
        print(cancel)
        db_models: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.user_id == current_user.id
        ).all()
        for db_model in db_models:
            db_model.delete_from_db()
        return render_template("add_model_classifier.html", form=form, status="Initial",
                               type='edit' if modelId else 'add')

    if "Initial" in request.form or cancel == "Add":
        print(cancel)
        try:
            if cancel == "Add":
                db_model: ModelForProccess = ModelForProccess.query.filter(
                    ModelForProccess.id == request.form["model_id"]
                ).first()
            else:
                name = request.form["name"]
                description = request.form["description"]
                if modelId:
                    classifier: ExplainedModel = ExplainedModel.query.filter(
                        ExplainedModel.id == modelId
                    ).first()
                    model = classifier.getElement('model')
                    training_df: pd.DataFrame = classifier.data_set_data.getElement('dataset')
                else:
                    model = joblib.load(request.files["model"])
                    training_df: pd.DataFrame
                    print('request.files["dataset"]: ', request.files["dataset"])
                    try:
                        training_df = pd.read_csv(request.files["dataset"])
                    except:  # noqa: E722
                        training_df = pd.read_excel(request.files["dataset"])


                db_model = ModelForProccess(
                    **{
                        "name": name,
                        "description": description,
                        "model": model,
                        "dataset": training_df,
                    }
                )
                db_model.user = User.query.filter(User.id == current_user.id).first()
                db_model.add_to_db()
            status = "Second"
            possible_targets = list(
                set(db_model.getElement("dataset").columns)
                - set(db_model.getElement("model").feature_names_in_)
            )
            return render_template(
                "add_model_classifier.html",
                form=possible_targets,
                status=status,
                model_id=db_model.id,
                type='edit' if modelId else 'add'
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template(
                "add_model_classifier.html", form=form, status=status, type='edit' if modelId else 'add'
            )
    elif "Second" in request.form or cancel == "Create":
        try:
            db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.form["model_id"]
            ).first()

            if cancel != "Create":
                print('Entering on Crete...')
                possible_not_needed_variables = list(
                    set(db_model.getElement("dataset").columns)
                    - set(db_model.getElement("model").feature_names_in_)
                )
                possible_not_needed_variables.remove(request.form["target"])
                print('possible_not_needed_variables:', possible_not_needed_variables)
                db_model.target_row = request.form["target"]
                db_model.setElements(
                    **{
                        'dataset': db_model.getElement("dataset").drop(
                            columns=possible_not_needed_variables)
                    }
                )
                db_model.db_commit()
            else:
                print(cancel)
                explainer: ExplainedClassifierModel = (
                    ExplainedClassifierModel.query.filter(
                        ExplainedClassifierModel.user_id == current_user.id
                    ).all()[-1]
                )
                print(explainer.name)
                if db_model.name == explainer.name:
                    explainer.delete_from_db()
                db_model.percent_processed = 0
                db_model.db_commit()

            qualitative_variables_form = []
            df = db_model.getElement("dataset")
            print('df: ', df)

            values_on_current = {}
            if modelId:
                classifier: ExplainedClassifierModel = ExplainedClassifierModel.query.filter(
                    ExplainedClassifierModel.explainer_model_id == modelId
                ).first()

                qualitative_target = classifier.getElement('target_names_dict')
                qualitative_variables_values = classifier.explainer_model.getElement('q_variables_dict')

                print(qualitative_target)
                values_on_current[f'{db_model.target_row}'] = []
                for variable in qualitative_target['variables']:
                    values_on_current[f'{db_model.target_row}'].append(variable['new_value'])

                print(qualitative_variables_values)
                for column in qualitative_variables_values:
                    values_on_current[f'{column["column_name"]}'] = []
                    for variable in column['variables']:
                        print('variable: ', variable)
                        values_on_current[f'{column["column_name"]}'].append(variable['new_value'])
            print('values_on_current: ', values_on_current)
            for column in df:
                if len(set(df[column])) < 5 or column == db_model.target_row:

                    qualitative_variables_form.append(
                        {
                            "name": column,
                            "variables": list(set(df[column])),
                            "values_on_current": values_on_current[column] if modelId else []
                        }
                    )

            status = "Add"
            return render_template(
                "add_model_classifier.html",
                form=qualitative_variables_form,
                variables=list(df.columns),
                status=status,
                model_id=db_model.id,
                type='edit' if modelId else 'add'
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template(
                "add_model_classifier.html", form=form, status=status, type='edit' if modelId else 'add'
            )
    elif "Add" in request.form:

        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == request.form["model_id"]
        ).first()
        df = db_model.getElement("dataset")
        qualitative_variables_saved = []
        target_description: object
        value_number: int = 0

        q_dict = {}
        features_description = {}
        for element in request.form:
            if element != "model_id":
                if element in df.columns:
                    features_description[element] = (
                        request.form[element]
                        if request.form[element] != ""
                        else "Sin descripción"
                    )
                elif "Q-Variable" in element or element == "Add":
                    value_number = 0
                    if q_dict != {}:
                        if q_dict["column_name"] == db_model.target_row:
                            target_description = q_dict
                        else:
                            qualitative_variables_saved.append(q_dict)
                    if element != "Add":
                        q_dict = {
                            "column_name": request.form[element],
                            "variables": [],
                        }
                else:
                    old_value = element.replace(f"{q_dict['column_name']}-", "")
                    new_value = (
                        request.form[element]
                        if request.form[element] != ""
                        else old_value
                    )
                    try:
                        old_value = int(old_value)
                    except:  # noqa: E722
                        pass

                    q_dict["variables"].append(
                        {
                            "old_value": old_value,
                            "new_value": new_value,
                        }
                    )
                    value_number += 1

        db_model.setElements(
            **{
                "qualitative_variables_saved": qualitative_variables_saved,
                "target_description": target_description,
                "features_description": features_description,
            }
        )
        db_model.db_commit()

        print("current_user_on_creation: ", current_user)
        user_id = current_user.id

        x = threading.Thread(
            target=thread_function,
            args=(db_model.id, current_app.app_context(), user_id, "Classifier", modelId),
        )
        x.start()
        return render_template(
            "add_model_classifier.html", model_id=db_model.id, status="Create", type='edit' if modelId else 'add'
        )
    elif "Create" in request.form:
        db_models: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.user_id == current_user.id
        ).all()
        for model in db_models:
            model.delete_from_db()
        return redirect('/processor/manage_classifiers')
    db_models: ModelForProccess = ModelForProccess.query.filter(
        ModelForProccess.user_id == current_user.id
    ).all()
    for model in db_models:
        model.delete_from_db()
    return render_template("add_model_classifier.html", form=form, status="Initial", type='edit' if modelId else 'add')


@blueprint.route("/add_regressor", methods=["GET", "POST"])
@blueprint.route("/edit_regressor/<modelId>", methods=["GET", "POST"])
@login_required
def save_regressor(modelId: int = 0):
    form = add_model(request.form)
    if modelId:
        regressor: ExplainedModel = ExplainedModel.query.filter(
            ExplainedModel.id == modelId
        ).first()
        form.name.default = regressor.name
        form.description.default = regressor.model_description
        unit = regressor.explainer_regressor.getElement('unit')
    cancel = None
    try:
        cancel = request.form["cancel"]
    except:  # noqa: E722
        pass

    print(cancel)
    print(request.form)

    if cancel == "Second":
        print(cancel)
        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == request.form["model_id"]
        ).first()
        db_model.delete_from_db()
        return render_template("add_model_regressor.html", form=form, status="Initial", type='edit' if modelId else 'add')

    if "Initial" in request.form or cancel == "Add":
        print(cancel)
        try:
            if cancel == "Add":
                db_model: ModelForProccess = ModelForProccess.query.filter(
                    ModelForProccess.id == request.form["model_id"]
                ).first()
            else:
                name = request.form["name"]
                description = request.form["description"]
                unit = request.form["unit"]
                if modelId:
                    classifier: ExplainedModel = ExplainedModel.query.filter(
                        ExplainedModel.id == modelId
                    ).first()
                    model = classifier.getElement('model')
                    training_df: pd.DataFrame = classifier.data_set_data.getElement('dataset')
                else:
                    model = joblib.load(request.files["model"])
                    training_df: pd.DataFrame
                    try:
                        training_df = pd.read_csv(request.files["dataset"])
                    except:  # noqa: E722
                        training_df = pd.read_excel(request.files["dataset"])

                db_model = ModelForProccess(
                    **{
                        "name": name,
                        "description": description,
                        "model": model,
                        "dataset": training_df,
                        "unit": unit,
                    }
                )
                db_model.user = User.query.filter(User.id == current_user.id).first()
                db_model.add_to_db()
            status = "Second"
            possible_targets = list(
                set(db_model.getElement("dataset").columns)
                - set(db_model.getElement("model").feature_names_in_)
            )
            return render_template(
                "add_model_regressor.html",
                form=possible_targets,
                status=status,
                model_id=db_model.id,
                type='edit' if modelId else 'add'
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template("add_model_regressor.html", form=form, status=status, type='edit' if modelId else 'add')
    elif "Second" in request.form or cancel == "Create":

        try:
            db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.form["model_id"]
            ).first()

            if cancel != "Create":
                possible_not_needed_variables = list(
                    set(db_model.getElement("dataset").columns)
                    - set(db_model.getElement("model").feature_names_in_)
                )
                possible_not_needed_variables.remove(request.form["target"])
                print("Possible Not Needeed: ", possible_not_needed_variables)
                db_model.target_row = request.form["target"]
                db_model.setElements(
                    **{
                        "dataset": db_model.getElement("dataset").drop(
                            columns=possible_not_needed_variables)
                    })
                db_model.db_commit()
            else:
                print(cancel)
                explainer: ExplainedRegressorModel = (
                    ExplainedRegressorModel.query.filter(
                        ExplainedRegressorModel.user_id == current_user.id
                    ).all()[-1]
                )
                print(explainer.name)
                if db_model.name == explainer.name:
                    explainer.delete_from_db()
                db_model.percent_processed = 0
                db_model.db_commit()

            qualitative_variables_form = []
            df = db_model.getElement("dataset")
            values_on_current = {}
            if modelId:
                regressor: ExplainedRegressorModel = ExplainedRegressorModel.query.filter(
                    ExplainedRegressorModel.explainer_model_id == modelId
                ).first()

                unit = regressor.getElement('unit')
                qualitative_variables_values = regressor.explainer_model.getElement('q_variables_dict')

                print(qualitative_variables_values)
                for column in qualitative_variables_values:
                    values_on_current[f'{column["column_name"]}'] = []
                    for variable in column['variables']:
                        print('variable: ', variable)
                        values_on_current[f'{column["column_name"]}'].append(variable['new_value'])
            print('values_on_current: ', values_on_current)
            for column in df:
                if len(set(df[column])) < 5 and column != db_model.target_row:
                    qualitative_variables_form.append(
                        {
                            "name": column,
                            "variables": list(set(df[column])),
                            "values_on_current": values_on_current[column] if modelId else []
                        }
                    )

            status = "Add"
            return render_template(
                "add_model_regressor.html",
                form=qualitative_variables_form,
                variables=list(df.columns),
                status=status,
                model_id=db_model.id,
                type='edit' if modelId else 'add'
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template("add_model_regressor.html", form=form, status=status, type='edit' if modelId else 'add')

    elif "Add" in request.form:

        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == request.form["model_id"]
        ).first()
        df = db_model.getElement("dataset")
        qualitative_variables_saved = []
        target_description: object
        value_number: int = 0

        q_dict = {}
        features_description = {}
        for element in request.form:
            if element != "model_id":
                if element in df.columns:
                    features_description[element] = (
                        request.form[element]
                        if request.form[element] != ""
                        else "Sin descripción"
                    )
                elif "Q-Variable" in element or element == "Add":
                    value_number = 0
                    if q_dict != {}:
                        if q_dict["column_name"] != db_model.target_row:
                            qualitative_variables_saved.append(q_dict)

                    if element != "Add":
                        q_dict = {
                            "column_name": request.form[element],
                            "variables": [],
                        }
                else:
                    old_value = element.replace(f"{q_dict['column_name']}-", "")
                    new_value = (
                        request.form[element]
                        if request.form[element] != ""
                        else old_value
                    )
                    try:
                        old_value = int(old_value)
                    except:  # noqa: E722
                        pass

                    q_dict["variables"].append(
                        {
                            "old_value": old_value,
                            "new_value": new_value,
                        }
                    )
                    value_number += 1

        db_model.setElements(
            **{
                "qualitative_variables_saved": qualitative_variables_saved,
                "features_description": features_description,
            }
        )
        db_model.db_commit()
        user_id = current_user.id
        x = threading.Thread(
            target=thread_function,
            args=(db_model.id, current_app.app_context(), user_id, "Regressor", modelId),
        )
        x.start()
        # Function to add, the models
        return render_template(
            "add_model_regressor.html", model_id=db_model.id, status="Create", type='edit' if modelId else 'add'
        )
    elif "Create" in request.form:
        db_models: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.user_id == current_user.id
        ).all()
        for model in db_models:
            model.delete_from_db()
        return redirect('/processor/manage_regressors')
    db_models: ModelForProccess = ModelForProccess.query.filter(
        ModelForProccess.user_id == current_user.id
    ).all()
    for model in db_models:
        model.delete_from_db()
    return render_template(
        "add_model_regressor.html",
        form=form,
        status="Initial",
        type='edit' if modelId else 'add',
        unit=unit if modelId else '')


@blueprint.route("/manage_classifiers", methods=["GET", "POST"])
@login_required
def manage_classifiers():
    return render_template("classifiers.html")


@blueprint.route("/manage_regressors", methods=["GET", "POST"])
@login_required
def manage_regressors():
    return render_template("regressors.html")
