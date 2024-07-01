import threading

from sklearn.ensemble import RandomForestClassifier
from app.proccessor.forms import add_classifier
from app.proccessor.model.dataset_interaction_methods import get_modified_dataframe
from app.proccessor.model.explainers.decision_tree_surrogate import ExplainSingleTree
from app.proccessor.models import (
    DataSetData,
    DataSetDataDistribution,
    ExplainedClassifierModel,
    ModelForProccess,
    SurrogateTreeClassifierData,
    Tree,
    TreeClassifierRule,
    TreeClassifierRuleCause,
)
from . import blueprint
from flask import current_app, render_template, request
from flask_login import login_required
import pandas as pd

import joblib


def thread_function(model_id, app):
    with app:
        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == model_id
        ).first()
        
        db_model.percent_processed = 10
        db_model.process_message = "Sincronizando datos..."
        db_model.db_commit()

        db_model_classifier_model: RandomForestClassifier = db_model.getElement("model")
        db_model_classifier_dataset: pd.DataFrame = db_model.getElement("dataset")
        db_model_classifier_target_row: str = db_model.getElement("target_row")
        db_model_classifier_target_description: dict = db_model.getElement(
            "target_description"
        )
        db_model_classifier_target_class_names: list = []
        db_model_classifier_qualitative_columns = db_model.getElement(
            "qualitative_variables_saved"
        )
        db_model_classifier_qualitative_column_names: list = []

        for variable in db_model_classifier_target_description["variables"]:
            db_model_classifier_target_class_names.append(variable["new_value"])

        for column in db_model_classifier_qualitative_columns:
            db_model_classifier_qualitative_column_names.append(column["column_name"])
            
        db_model.percent_processed = 20
        db_model.process_message = "Generando modelo base..."
        db_model.db_commit()

        #### BASE MODEL ####

        classifier_model_data = ExplainedClassifierModel(
            **{
                "name": db_model.name,
                "model": db_model_classifier_model,
                "indexesDict": "dict of indexes",
                "indexColumnName": "column name for indexes",
                "model_description": db_model.description,
                "features_description": db_model.getElement("features_description"),
                "target_row": db_model.target_row,
                "q_variables_dict": db_model_classifier_qualitative_columns,
                "test_size": 0.6,
                "random_state": 123,
                "target_names_dict": db_model_classifier_target_description,
            }
        )
        
        db_model.percent_processed = 30
        db_model.process_message = "Cargando metricas del conjunto de datos..."
        db_model.db_commit()

        #### DATASET DATA ####

        dataset_data = DataSetData(
            **{
                "dataset": db_model_classifier_dataset,
                "dataset_modified": get_modified_dataframe(
                    df=db_model_classifier_dataset,
                    target_description=db_model_classifier_target_description,
                    qualitative_columns=db_model_classifier_qualitative_columns,
                ),
            }
        )

        dataset_data_distribution_qualitative = DataSetDataDistribution(
            **{
                "isNumeric": False,
                "rows_amount": 3,
                "columns_amount": 3,
                "distribution_dataset": "Dataset for the distribution type",
                "isPrime": False,
            }
        )
        dataset_data_distribution_qualitative.data_set_data = dataset_data

        dataset_data_distribution_numeric = DataSetDataDistribution(
            **{
                "rows_amount": 3,
                "columns_amount": 3,
                "distribution_dataset": "Dataset for the distribution type",
            }
        )
        dataset_data_distribution_numeric.data_set_data = dataset_data
        
        db_model.percent_processed = 40
        db_model.process_message = "Creando modelo subrogado datos..."
        db_model.db_commit()

        tree_depth = 3
        surrogate_inexact_rules_amount = -1
        while surrogate_inexact_rules_amount != 0:
            
            db_model.process_message = (
                f"Creando modelo subrogado datos (de rofundidad {tree_depth})..."
            )
            db_model.db_commit()
            surrogate_tree = ExplainSingleTree.createSurrogateTree(
                model=db_model_classifier_model,
                x_train=db_model_classifier_dataset.drop(
                    columns=db_model_classifier_target_row
                ),
                max_depth=tree_depth,
            )
            tree_depth += 1
            rules = ExplainSingleTree.get_rules(
                model=surrogate_tree.tree_,
                q_variables=db_model_classifier_qualitative_column_names,
                q_variables_values=db_model_classifier_qualitative_columns,
                features=surrogate_tree.feature_names_in_,
                class_names=db_model_classifier_target_class_names,
                target=db_model_classifier_target_row,
            )

            db_tree = Tree(
                **{
                    "depth": surrogate_tree.get_depth(),
                    "rules_amount": len(rules),
                }
            )

            surrogate_inexact_rules_amount = 0
            for rule in rules:
                db_rule = TreeClassifierRule(
                    **{
                        "target_value": rule["target_value"],
                        "probability": rule["probability"],
                        "samples_amount": rule["samples_amount"],
                    }
                )
                db_rule.tree_classifier = db_tree
                db_rule.add_to_db()

                for cause in rule["causes"]:
                    db_cause = TreeClassifierRuleCause(
                        **{
                            "predictor": cause["item"],
                            "relation_sign": cause["sign"],
                            "value": cause["value"],
                        }
                    )
                    db_cause.tree_classifier_rule = db_rule
                    db_cause.add_to_db()

                if rule["probability"] < 100:
                    surrogate_inexact_rules_amount += 1

            db_tree.inexact_rules_amount = surrogate_inexact_rules_amount
            db_tree.add_to_db()

            surrogate_tree_data = SurrogateTreeClassifierData(
                **{"tree_model": surrogate_tree}
            )
            surrogate_tree_data.tree = db_tree
            surrogate_tree_data.explained_classifier_model = classifier_model_data
            surrogate_tree_data.add_to_db()
        
        db_model.percent_processed = 80
        db_model.process_message = "Guardando la base de datos del modelo..."
        db_model.db_commit()

        classifier_model_data.data_set_data = dataset_data
        
        dataset_data_distribution_numeric.add_to_db()
        dataset_data_distribution_qualitative.add_to_db()
        dataset_data.add_to_db()
        classifier_model_data.add_to_db()
        
        db_model.percent_processed = 100
        db_model.process_message = "Completado !!!"
        db_model.db_commit()


@blueprint.route("/classifier", methods=["GET", "POST"])
@login_required
def save_classifier():
    form = add_classifier(request.form)
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
        return render_template("add_models.html", form=form, status="Initial")

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
                    }
                )
                
                db_model.add_to_db()
            status = "Second"
            possible_targets = list(
                set(db_model.getElement("dataset").columns)
                - set(db_model.getElement("model").feature_names_in_)
            )
            return render_template(
                "add_models.html",
                form=possible_targets,
                status=status,
                model_id=db_model.id,
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template("add_models.html", form=form, status=status)
    elif "Second" in request.form or cancel == "Create":
       
        try:
            db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.form["model_id"]
            ).first()

            if cancel != "Create":
                db_model.target_row = request.form["target"]
                db_model.db_commit()
            else:
                print(cancel)
                explainer: ExplainedClassifierModel = ExplainedClassifierModel.query.all()[-1]
                print(explainer.name)
                if db_model.name == explainer.name:
                    explainer.delete_from_db()
                db_model.percent_processed = 0
                db_model.db_commit()
                
                
            qualitative_variables_form = []
            df = db_model.getElement("dataset")
            for column in df:
                if len(set(df[column])) < 5 or column == db_model.target_row:
                    qualitative_variables_form.append(
                        {
                            "name": column,
                            "variables": list(set(df[column])),
                        }
                    )

            status = "Add"
            return render_template(
                "add_models.html",
                form=qualitative_variables_form,
                variables=list(df.columns),
                status=status,
                model_id=db_model.id,
            )
        except Exception as e:
            print(e)
            status = "Wrong Data"
            return render_template("add_models.html", form=form, status=status)

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

        x = threading.Thread(
            target=thread_function, args=(db_model.id, current_app.app_context())
        )
        x.start()
        x.daemon
        # Function to add, the models
        return render_template("add_models.html", model_id=db_model.id, status="Create")
    elif "Create" in request.form:
        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == request.form["model_id"]
        ).first()
        db_model.delete_from_db()
    return render_template("add_models.html", form=form, status="Initial")
