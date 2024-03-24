from . import blueprint
from flask import render_template, current_app, request, redirect
from flask_login import login_required, current_user
from .forms import (
    add_user_Form,
    delete_user_Form,
    change_password_Form,
    setting_password_Form,
)
from ..base.models import User


@blueprint.route("/api/users", methods=["GET"])
@login_required
def get_Users():
    return {"data": [user.to_dict() for user in User.get_users_list()]}


@blueprint.route("/manage_users", methods=["GET", "POST"])
@login_required
def manage_Users():
    admin_user = current_app.config["ADMIN"]["username"]
    if current_user.username == admin_user:
        # form = add_user_Form(request.form)
        users = User.get_users_list()
        print(users)
        # if "Add" in request.form:
        #     user = User.query.filter_by(username=request.form["username"]).first()
        #     email = User.query.filter_by(email=request.form["email"]).first()
        #     if user:
        #         status = "Username is existing"
        #     elif email:
        #         status = "Email is existing"
        #     else:
        #         User(**request.form).add_to_db()
        #         status = "Add user success !"
        #     return render_template("add_user.html", form=form, status=status)
        return render_template("manage_users.html")
    return redirect("/page_403")


@blueprint.route("/add_user", methods=["GET", "POST"])
@login_required
def add_User():
    admin_user = current_app.config["ADMIN"]["username"]
    if current_user.username == admin_user:
        form = add_user_Form(request.form)
        if "Add" in request.form:
            user = User.query.filter_by(username=request.form["username"]).first()
            email = User.query.filter_by(email=request.form["email"]).first()
            if user:
                status = "Username is existing"
            elif email:
                status = "Email is existing"
            else:
                User(**request.form).add_to_db()
                status = "Add user success !"
            return render_template("add_user.html", form=form, status=status)
        return render_template("add_user.html", form=form, status="")
    return redirect("/page_403")


@blueprint.route("/delete_user/<id>", methods=["GET", "POST"])
@login_required
def delete_user(id):
    admin_user = current_app.config["ADMIN"]["username"]
    if current_user.username == admin_user:
        user = User.query.filter_by(id=id).first()
        if user:
            form = delete_user_Form(user.username)
            if "Delete" in request.form:
                username = request.form["username"]
                user = User.query.filter_by(username=username).first()
                if user:
                    if username == admin_user:
                        status = "admin user can't be deleted !"
                    else:
                        user.delete_from_db()
                        status = "delete user success !"
                else:
                    status = "user doesn't exist !"
                return redirect("/setting/manage_users")
            return render_template("delete_user.html", form=form, status="", user=user)
        return redirect("/page_404")
    return redirect("/page_403")


@blueprint.route("/setting_password/<id>", methods=["GET", "POST"])
@login_required
def setting_password(id):
    print(id)
    admin_user = current_app.config["ADMIN"]["username"]
    if current_user.username == admin_user:
        form = setting_password_Form(request.form)
        if "Setting" in request.form:
            username = request.form["username"]
            user = User.query.filter_by(username=username).first()
            if user:
                if username == admin_user:
                    status = "please change admin password from server !"
                else:
                    user.password = user.hashpw(request.form["password"])
                    user.db_commit()
                    status = "Setting password success !"
            else:
                status = "user doesn't exist !"
            return render_template("setting_password.html", form=form, status=status)
        return render_template("setting_password.html", form=form, status="")
    return redirect("/page_403")


@blueprint.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    # admin_user = current_app.config["ADMIN"]["username"]
    # if current_user.username == admin_user:
    #     return "please change admin password from server"
    # else:
    form = change_password_Form(request.form)
    if "Change" in request.form:
        user = User.query.filter_by(username=current_user.username).first()
        if user.checkpw(request.form["origin_password"]):
            if request.form["new_password"] == request.form["new_password2"]:
                user.password = user.hashpw(request.form["new_password"])
                user.db_commit()
                status = "Change Password Success !"
            else:
                status = "Both New Password is Not Equal !"
        else:
            status = "Origin Password Error !"
        return render_template("change_password.html", form=form, status=status)
    return render_template("change_password.html", form=form, status="")
