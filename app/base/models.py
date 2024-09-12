from bcrypt import gensalt, hashpw, checkpw
from flask_login import UserMixin
from sqlalchemy import BINARY, Column, Integer, String, Boolean
from sqlalchemy.orm import relationship

from app import db, login_manager


class User(db.Model, UserMixin):
    __tablename__ = "User"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(BINARY)
    langSelection = Column(String)
    is_admin = Column(Boolean, default=False)

    model_for_proccess = relationship(
        "ModelForProccess", back_populates="user", cascade="all, delete-orphan"
    )

    classifier_models = relationship(
        "ExplainedClassifierModel", back_populates="user", cascade="all, delete-orphan"
    )

    regressor_models = relationship(
        "ExplainedRegressorModel", back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.username,
            "email": self.email,
        }

    def get_users_list():
        return User.query.filter(User.is_admin == False).all()  # noqa: E712

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack its value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, "__iter__") and not isinstance(value, str):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]
            if property == "password":
                value = hashpw(value.encode("utf8"), gensalt())
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)

    def add_to_db(self):
        db.session.add(self)
        self.db_commit()

    def delete_from_db(self):
        db.session.delete(self)
        self.db_commit()

    def db_commit(self):
        db.session.commit()

    def hashpw(self, password):
        return hashpw(password.encode("utf8"), gensalt())

    def checkpw(self, password):
        return checkpw(password.encode("utf8"), self.password)


@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()


@login_manager.request_loader
def request_loader(request):
    username = request.form.get("username")
    try:
        user = User.query.filter_by(username=username).first()
    except Exception as e:
        print(str(e))
        user = None
    return user if user else None
