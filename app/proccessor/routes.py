import threading
import time

from app.proccessor.forms import add_classifier
from app.proccessor.model.model import ClassificationTrainedModel
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import current_app, make_response, render_template, request
from flask_login import login_required
import pandas as pd

import joblib


def thread_function(model_id, app):
    print(1)
    with app:
        db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == model_id
            ).first()
        db_model.percent_processed = 10
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 26
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 36
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 58
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 89
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 99
        db_model.db_commit()
        time.sleep(5)
        db_model.percent_processed = 100
        db_model.db_commit()
        time.sleep(5)

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
                set(db_model.to_dict()["dataset"].columns)
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
            df = db_model.to_dict()["dataset"]
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
        target_description = {}
        
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
                            "old_value": element.replace(f'{q_dict['column_name']}-', ''),
                            "new_value": request.form[element],
                        }
                    )
        
        
        test_size = 0.2
        random_state = 123
        
        features_description = "features_description"        
        full_model = ClassificationTrainedModel(name=db_model.name, df=db_model.to_dict()["dataset"], predictors_description=features_description,
                                                target=db_model.target_row, test_size=test_size, random_state=random_state,
                                                model=db_model.to_dict()["model"], model_description=db_model.description,
                                                target_description=target_description,
                                                q_variables_values_list=qualitative_variables_saved)
        x = threading.Thread(target=thread_function, args=(db_model.id, current_app.app_context()))
        x.start()
        # Function to add, the model    
        return render_template("add_models.html", model_id=db_model.id, status="Create")
    return render_template("add_models.html", form=form, status="Initial")



@blueprint.route("/classifier/percent", methods=["GET", "POST"])
@login_required
def get_percent():
    db_model: ModelForProccess = ModelForProccess.query.filter(
                ModelForProccess.id == request.data.decode()
    ).first()
    return make_response([db_model.percent_processed])
