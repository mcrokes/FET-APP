from ctypes import Array
import time
from tokenize import String

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from app.proccessor.forms import add_classifier
from app.proccessor.model.explainers.decision_tree_surrogate import ExplainSingleTree
from app.proccessor.model.explainers.importance import Importance
from app.proccessor.models import (
    DataSetData,
    DataSetDataDistribution,
    ExplainedClassifierModel,
    ImportancesData,
    InnerTreeClassifierData,
    ModelForProccess,
    PermutationImportance,
    PermutationImportancesData,
    Tree,
    TreeClassifierRule,
    TreeClassifierRuleCause,
)
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required
import pandas as pd

import joblib


def thread_function(model_id, app):
    with app:
        db_model: ModelForProccess = ModelForProccess.query.filter(
            ModelForProccess.id == model_id
        ).first()

        db_model_classifier_model: RandomForestClassifier = db_model.getElement("model")
        db_model_classifier_dataset: pd.DataFrame = db_model.getElement("dataset")
        db_model_classifier_description: str = db_model.getElement("description")
        db_model_classifier_name: str = db_model.getElement("name")
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
            
            
        #### BASE MODEL ####
        
        classifier_model_data = ExplainedClassifierModel(
            **{
                "name": db_model.name,
                "model": db_model_classifier_model,
                "indexesDict": "dict of indexes",
                "indexColumnName": "column name for indexes",
                "model_description": db_model.description,
                "features_description": "description for the features",
                "target_row": db_model.target_row,
                "q_variables_dict": "dictionary for qualitative variables",
                "test_size": 0.6,
                "random_state": 123,
                "target_names_dict": "dictionary for qualitative target names",
                # "dataset": db_model.to_dict()["dataset"],
            }
        )


        #### DATASET DATA ####

        dataset_data = DataSetData(
            **{
                "dataset": db_model_classifier_dataset,
                "dataset_modified": "db_model.to_dict()['dataset'] with q variables modification",
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

        #### IMPORTANCES DATA ####

        importances_data = ImportancesData(**{"explanation": "Importace Explanation"})

        permutation_importances_data = PermutationImportancesData(
            **{"explanation": "Permutation Importace Explanation"}
        )

        permutation_importances: pd.DataFrame = (
            Importance.create_permutation_importance(
                model=db_model_classifier_model,
                features=db_model_classifier_model.feature_names_in_,
                x_train=db_model_classifier_dataset.drop(
                    columns=db_model_classifier_target_row
                ),
                y_train=db_model_classifier_dataset[db_model_classifier_target_row],
            )
        )

        permutation_importance_list = []
        for _, row in permutation_importances.iterrows():
            permutation_importance = PermutationImportance(
                **{
                    "importance_mean": row["importances_mean"],
                    "importance_std": row["importances_std"],
                    "predictor": row["Predictor"],
                }
            )
            permutation_importance.permutation_importances_data = (
                permutation_importances_data
            )
            permutation_importance_list.append(permutation_importance)

        estimator: DecisionTreeClassifier = db_model_classifier_model.estimators_[0]
        
        estimator.get_depth()

        for index, tree in enumerate(db_model_classifier_model.estimators_):            
            rules = ExplainSingleTree.get_rules(
                model=tree.tree_,
                q_variables=db_model_classifier_qualitative_column_names,
                q_variables_values=db_model_classifier_qualitative_columns,
                features=db_model_classifier_model.feature_names_in_,
                class_names=db_model_classifier_target_class_names,
                target=db_model_classifier_target_row,
            )            
                    
            db_tree = Tree(
                **{
                    "depth": tree.get_depth(), 
                    "rules_amount": len(rules),
                }
            )
            
            inexact_rules_amount = 0
            for rule in rules:
                db_rule = TreeClassifierRule(
                    **{
                        "target_value": rule["target_value"],
                        "probability": rule["probability"],
                        "samples_amount": rule["samples_amount"]                            
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
                    inexact_rules_amount += 1
                
            
            db_tree.inexact_rules_amount = inexact_rules_amount
            db_tree.add_to_db()
            
            inner_tree_data = InnerTreeClassifierData(
                **{
                    "tree_number": index
                }
            )
            inner_tree_data.tree = db_tree
            inner_tree_data.explained_classifier_model = classifier_model_data
            inner_tree_data.add_to_db()


        

        classifier_model_data.data_set_data = dataset_data
        classifier_model_data.importances_data = importances_data
        classifier_model_data.permutation_importances_data = (
            permutation_importances_data
        )
        dataset_data_distribution_numeric.add_to_db()
        dataset_data_distribution_qualitative.add_to_db()
        dataset_data.add_to_db()
        for pi in permutation_importance_list:
            pi.add_to_db()
        importances_data.add_to_db()
        permutation_importances_data.add_to_db()
        classifier_model_data.add_to_db()

        print(
            classifier_model_data.data_set_data.data_set_data_distributions[0].isNumeric
        )
        # db_model.percent_processed = 10
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 26
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 36
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 58
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 89
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 99
        # db_model.db_commit()
        # time.sleep(5)
        # db_model.percent_processed = 100
        # db_model.db_commit()
        # time.sleep(5)


@blueprint.route("/classifier", methods=["GET", "POST"])
@login_required
def save_classifier():
    form = add_classifier(request.form)

    if "Initial" in request.form:
        try:
            name = request.form["name"]
            description = request.form["description"]
            model = joblib.load(request.files["model"])
            training_df = pd.read_csv(request.files["dataset"])
            db_model = ModelForProccess(
                **{
                    "name": name,
                    "description": description,
                    "model": model,
                    "dataset": training_df,
                }
            )
            status = "Second"
            db_model.add_to_db()
            possible_targets = list(
                set(db_model.getElement("dataset").columns)
                - set(model.feature_names_in_)
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
    elif "Second" in request.form:
        try:
            db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.form["model_id"]
            ).first()
            db_model.target_row = request.form["target"]
            db_model.db_commit()
            qualitative_variables_form = []
            df = db_model.getElement("dataset")
            for column in df:
                if len(set(df[column])) < 5:
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
        qualitative_variables_saved = []
        target_description: object

        q_dict = {}
        for element in request.form:
            if element != "model_id":
                if "Q-Variable" in element or element == "Add":
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
                    q_dict["variables"].append(
                        {
                            "old_value": element.replace(
                                f"{q_dict['column_name']}-", ""
                            ),
                            "new_value": request.form[element],
                        }
                    )

        print(target_description)

        test_size = 0.2
        random_state = 123

        db_model.setElements(
            **{
                "qualitative_variables_saved": qualitative_variables_saved,
                "target_description": target_description,
            }
        )
        db_model.db_commit()

        thread_function(db_model.id, current_app.app_context())
        # features_description = "features_description"
        # full_model = ClassificationTrainedModel(name=db_model.name, df=db_model.to_dict()["dataset"], predictors_description=features_description,
        #                                         target=db_model.target_row, test_size=test_size, random_state=random_state,
        #                                         model=db_model.to_dict()["model"], model_description=db_model.description,
        #                                         target_description=target_description,
        #                                         q_variables_values_list=qualitative_variables_saved)
        # x = threading.Thread(target=thread_function, args=(db_model.id, current_app.app_context()))
        # x.start()
        # Function to add, the model
        return render_template("add_models.html", model_id=db_model.id, status="Create")
    return render_template("add_models.html", form=form, status="Initial")
