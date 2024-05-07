from sqlalchemy import BINARY, Column, Integer, String, Boolean
import json

import pickle

from app import db
class ModelForProccess(db.Model):

    __tablename__ = "model_for_proccess"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model = Column(String)
    dataset = Column(String)
    description = Column(String)
    target_row = Column(String)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": pickle.loads(self.model),
            "dataset": pickle.loads(self.dataset),
            "target_row": self.target_row
        }

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, "__iter__") and not isinstance(value, object):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]
            if property == "model" or property == "dataset":
                value = pickle.dumps(value)
            
            
            setattr(self, property, value)

    def __repr__(self):
        return str(self.name)

    def add_to_db(self):
        db.session.add(self)
        self.db_commit()

    def delete_from_db(self):
        db.session.delete(self)
        self.db_commit()

    def db_commit(self):
        db.session.commit()