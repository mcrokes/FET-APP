from sqlalchemy import BINARY, Column, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import relationship
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
    percent_processed = Column(Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": pickle.loads(self.model),
            "dataset": pickle.loads(self.dataset),
            "target_row": self.target_row,
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


ENCODED_FIELDS = [
    "model",
    "indexesDict",
    "dataset",
    "features_description",
    "q_variables_dict",
]


class ExplainedModel(db.Model):

    __abstract__ = True

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model = Column(String)  # Encoded
    indexesDict = Column(String)  # Encoded
    indexColumnName = Column(String)
    model_description = Column(String)
    features_description = Column(String)  # Encoded
    target_row = Column(String)
    q_variables_dict = Column(String)  # Encoded
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


class ExplainedClassifierModel(ExplainedModel):

    __tablename__ = "explained_classifier_model"

    __mapper_args__ = {
        "polymorphic_identity": "explained_classifier_model",
    }

    target_names_dict = Column(String)  # Encoded
    data_set_data = relationship(
        "DataSetData", uselist=False, back_populates="explained_classifier_model"
    )
    confusion_matrixes_data = relationship(
        "ConfusionMatrixesData", back_populates="explained_classifier_model"
    )
    importances_data = relationship(
        "ImportancesData", uselist=False, back_populates="explained_classifier_model"
    )
    permutation_importances_data = relationship(
        "PermutationImportancesData", uselist=False, back_populates="explained_classifier_model"
    )

class ImportancesData(db.Model):

    __tablename__ = "importances_data"

    explanation = Column(String)  # Encoded

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="importances_data"
    )

class PermutationImportancesData(db.Model):

    __tablename__ = "permutation_importances_data"

    explanation = Column(String)  # Encoded

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="permutation_importances_data"
    )
    
    permutation_importances = relationship(
        "PermutationImportance", back_populates="permutation_importances_data"
    )
    
class PermutationImportance(db.Model):

    __tablename__ = "permutation_importance"

    id = Column(Integer, primary_key=True)
    importance_mean = Column(Float)
    importance_std = Column(Float)
    predictor = Column(String)

    permutation_importances_data_id = Column(Integer, ForeignKey("permutation_importances_data.explained_classifier_model_id"))
    permutation_importances_data = relationship(
        "PermutationImportancesData", back_populates="permutation_importances"
    )

class DataSetData(db.Model):

    __tablename__ = "data_set_data"

    dataset = Column(String)  # Encoded
    dataset_modified = Column(String)  # Encoded

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="data_set_data"
    )
    
    data_set_data_distributions = relationship(
        "DataSetDataDistribution", back_populates="data_set_data"
    )
    
class DataSetDataDistribution(db.Model):

    __tablename__ = "data_set_data_distribution"

    id = Column(Integer, primary_key=True)
    isNumeric = Column(Boolean, default=True)
    rows_amount = Column(Integer)
    columns_amount = Column(Integer)
    isPrime = Column(Boolean, default=True)
    distribution_dataset = Column(String)  # Encoded

    data_set_data_id = Column(Integer, ForeignKey("data_set_data.explained_classifier_model_id"))
    data_set_data = relationship(
        "DataSetData", back_populates="data_set_data_distributions"
    )

class ConfusionMatrixesData(db.Model):

    __tablename__ = "confusion_matrixes_data"
    
    id = Column(Integer, primary_key=True)
    class_name = Column(String)

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id")
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", back_populates="confusion_matrixes_data"
    )
    
    confusion_matrixes = relationship(
        "ConfusionMatrix", back_populates="confusion_matrixes_data"
    )
    
class ConfusionMatrix(db.Model):

    __tablename__ = "confusion_matrix"
    
    id = Column(Integer, primary_key=True)
    matrix = Column(String)  # Encoded
    cut_off = Column(Integer)

    confusion_matrixes_data_id = Column(
        Integer, ForeignKey("confusion_matrixes_data.id")
    )
    confusion_matrixes_data = relationship(
        "ConfusionMatrixesData", back_populates="confusion_matrixes"
    )
    
    explanation = relationship(
        "ConfusionMatrixExplanation", uselist=False, back_populates="confusion_matrix"
    )
    
class ConfusionMatrixExplanation(db.Model):

    __tablename__ = "confusion_matrix_explanation"
    
    explanation_for_test = Column(String)  # Encoded

    confusion_matrix_id = Column(
        Integer, ForeignKey("confusion_matrix.id"), primary_key=True
    )
    confusion_matrix = relationship(
        "ConfusionMatrix", uselist=False, back_populates="explanation"
    )