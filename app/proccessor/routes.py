from app.proccessor.forms import add_classifier
from . import blueprint
from flask import render_template, current_app, request, redirect
from flask_login import login_required, current_user


@blueprint.route("/classifier")
@login_required
def save_classifier():
    form = add_classifier(request.form)
    if "Add" in request.form:
        if True:
            status = "Username is existing"
        elif True:
            status = "Email is existing"
        else:
            status = "Add user success !"
        return render_template("classifier_form.html", form=form, status=status)
    return render_template("classifier_form.html", form=form, status="")
