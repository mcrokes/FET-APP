import dbm
import json

from app.proccessor.forms import add_classifier
from app.proccessor.models import ModelForProccess
from . import blueprint
from flask import render_template, current_app, request, redirect
from flask_login import login_required, current_user
import pandas as pd
# from app import db

import joblib

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
            db_model = ModelForProccess(**{
                "name": name,
                "description": description,
                "model": model,
                "dataset": training_df
            })
            status = "Second"
            db_model.add_to_db()    
            possible_targets = list(set(db_model.to_dict()["dataset"].columns) - set(model.feature_names_in_))        
            return render_template("add_models.html", form=possible_targets, status=status, model_id=db_model.id)
        except Exception:
            status = "Wrong Data"
            return render_template("add_models.html", form=form, status=status) 
    elif "Second" in request.form:
        db_model:ModelForProccess = ModelForProccess.query.filter(ModelForProccess.id == request.form["model_id"]).first()
        db_model.target_row = request.form["target"]
        db_model.db_commit()
        status = "Add"
        return render_template("add_models.html", form=db_model.target_row, status=status, model_id=db_model.id)
    
    return render_template("add_models.html", form=form, status="Initial")
