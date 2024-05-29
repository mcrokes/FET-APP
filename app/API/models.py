from sqlalchemy import BINARY, Column, Float, Integer, String, Boolean
import json

import pickle

from app import db
ENCODED_FIELDS = ["model", "indexesDict", "dataset", "features_description", "q_variables_dict"]
class ExplainedModel(db.Model):

    __abstract__ = True
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model = Column(String) # Encoded
    indexesDict = Column(String) # Encoded
    indexColumnName = Column(String)
    dataset = Column(String) # Encoded
    model_description = Column(String)
    features_description = Column(String) # Encoded
    target_row = Column(String)
    q_variables_dict = Column(String) # Encoded
    percent_processed = Column(Integer, default=0)
    test_size = Column(Float)
    random_state = Column(Integer)
    type = Column(String, default="Classifier")
    
    __mapper_args__ = {
        "polymorphic_on": type,
        "polymorphic_identity": "explained_model",
    }
    
    def getElement(self, name):
        element = getattr(self, name)
        return pickle.loads(element) if name in ENCODED_FIELDS else element

    # def to_dict(self):
    #     return {
    #         "id": self.id,
    #         "name": self.name,
    #         "description": self.description,
    #         "model": pickle.loads(self.model),
    #         "dataset": pickle.loads(self.dataset),
    #         "target_row": self.target_row
    #     }

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            # depending on whether value is an iterable or not, we must
            # unpack it's value (when **kwargs is request.form, some values
            # will be a 1-element list)
            if hasattr(value, "__iter__") and not isinstance(value, object):
                # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
                value = value[0]
            # if property == "model" or property == "dataset":
            if property in ENCODED_FIELDS:
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
        

class ExplainedClassifierModel(db.Model):
    
    __tablename__ = "explained_classifier_model"
    
    __mapper_args__ = {
        "polymorphic_identity": "explained_classifier_model",
    }
    
    target_names_dict = Column(String) # Encoded
    