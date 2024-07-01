from email import message
from typing import Literal
from cycler import K
from sqlalchemy import BINARY, Column, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declared_attr, AbstractConcreteBase
import json

import pickle

from app import db

ENCODED_FIELDS = [
    "model",
    "indexesDict",
    "dataset",
    "dataset_modified",
    "features_description",
    "q_variables_dict",
    "qualitative_variables_saved",
    "target_description",
    "target_names_dict",
    "value",
    "tree_model",
]


def _initDB_model(self, kwargs):
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


class dbInteractionMethods:
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

    def setElements(self, **kwargs):
        _initDB_model(self, kwargs)

    def getElement(self, name):
        element = getattr(self, name)
        return pickle.loads(element) if name in ENCODED_FIELDS else element


class ModelForProccess(db.Model, dbInteractionMethods):

    __tablename__ = "model_for_proccess"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model = Column(String)  # Encoded
    dataset = Column(String)  # Encoded
    description = Column(String)
    target_row = Column(String)
    percent_processed = Column(Integer, default=0)
    process_message = Column(String)
    qualitative_variables_saved = Column(String)  # Encoded
    target_description = Column(String)  # Encoded
    features_description = Column(String)  # Encoded
    should_stop = Column(Boolean, default=False)

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    def __repr__(self):
        return str(self.name)

    def getElement(
        self,
        name: Literal[
            "id",
            "name",
            "model",
            "dataset",
            "description",
            "target_row",
            "percent_processed",
            "process_message",
            "features_description",
            "q_variables_dict",
            "should_stop",
        ],
    ):
        return super().getElement(name)


class ExplainedModel(db.Model, dbInteractionMethods):

    __abstract__ = True

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model = Column(String)  # Encoded
    indexesDict = Column(String)  # Encoded
    indexColumnName = Column(String)
    model_description = Column(String)
    features_description = Column(String)  # Encoded
    target_row = Column(String)
    target_description = Column(String)  # Encoded
    q_variables_dict = Column(String)  # Encoded
    test_size = Column(Float)
    random_state = Column(Integer)
    type = Column(String, default="Classifier")

    __mapper_args__ = {
        "polymorphic_on": type,
        "polymorphic_identity": "explained_model",
    }

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)


class ExplainedClassifierModel(ExplainedModel):

    __tablename__ = "explained_classifier_model"

    __mapper_args__ = {
        "polymorphic_identity": "explained_classifier_model",
    }

    target_names_dict = Column(String)  # Encoded
    data_set_data = relationship(
        "DataSetData", uselist=False, back_populates="explained_classifier_model", cascade="all, delete-orphan"
    )
    surrogate_trees_data = relationship(
        "SurrogateTreeClassifierData", back_populates="explained_classifier_model", cascade="all, delete-orphan"
    )

    def getElement(
        self,
        name: Literal[
            "id",
            "name",
            "model",
            "indexesDict",
            "indexColumnName",
            "model_description",
            "features_description",
            "target_names_dict",
            "target_row",
            "q_variables_dict",
            "test_size",
            "random_state",
        ],
    ):
        return super().getElement(name)


class DataSetData(db.Model, dbInteractionMethods):

    __tablename__ = "data_set_data"

    dataset = Column(String)  # Encoded
    dataset_modified = Column(String)  # Encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True,
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="data_set_data"
    )

    data_set_data_distributions = relationship(
        "DataSetDataDistribution", back_populates="data_set_data", cascade="all, delete-orphan"
    )

    def getElement(
        self,
        name: Literal[
            "dataset",
            "dataset_modified",
        ],
    ):
        return super().getElement(name)


class DataSetDataDistribution(db.Model, dbInteractionMethods):

    __tablename__ = "data_set_data_distribution"

    id = Column(Integer, primary_key=True)
    isNumeric = Column(Boolean, default=True)
    rows_amount = Column(Integer)
    columns_amount = Column(Integer)
    isPrime = Column(Boolean, default=True)
    distribution_dataset = Column(String)  # Encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    data_set_data_id = Column(
        Integer, ForeignKey("data_set_data.explained_classifier_model_id")
    )
    data_set_data = relationship(
        "DataSetData", uselist=False, back_populates="data_set_data_distributions"
    )

    def getElement(
        self,
        name: Literal[
            "id",
            "isNumeric",
            "rows_amount",
            "columns_amount",
            "isPrime",
            "distribution_dataset",
        ],
    ):
        return super().getElement(name)

class Tree(db.Model, dbInteractionMethods):

    __tablename__ = "tree"

    id = Column(Integer, primary_key=True)
    depth = Column(Integer)
    rules_amount = Column(Integer)
    inexact_rules_amount = Column(Integer)

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    __mapper_args__ = {
        "polymorphic_identity": "tree",
    }

    surrogate_tree = relationship(
        "SurrogateTreeClassifierData", uselist=False, back_populates="tree", single_parent=True, cascade="all, delete-orphan"
    )

    rules = relationship("TreeClassifierRule", back_populates="tree_classifier", cascade="all, delete-orphan")

    def getElement(
        self,
        name: Literal[
            "id",
            "depth",
            "rules_amount",
            "inexact_rules_amount",
        ],
    ):
        return super().getElement(name)


class SurrogateTreeClassifierData(db.Model, dbInteractionMethods):

    __tablename__ = "surrogate_tree_classifier_data"
    __mapper_args__ = {
        "polymorphic_identity": "surrogate_tree_classifier_data",
    }

    tree_model = Column(String)  # Encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    tree_id = Column(Integer, ForeignKey("tree.id"), primary_key=True)
    tree = relationship("Tree", uselist=False, back_populates="surrogate_tree", single_parent=True, cascade="all, delete-orphan")

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id")
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel",
        uselist=False,
        back_populates="surrogate_trees_data"
    )

    def getElement(
        self,
        name: Literal["tree_model",],
    ):
        return super().getElement(name)


class TreeClassifierRule(db.Model, dbInteractionMethods):
    __tablename__ = "tree_classifier_rule"

    id = Column(Integer, primary_key=True)
    target_value = Column(String)
    probability = Column(Float)
    samples_amount = Column(Float)

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    tree_id = Column(
        Integer,
        ForeignKey("tree.id"),
    )
    tree_classifier = relationship("Tree", back_populates="rules")

    causes = relationship(
        "TreeClassifierRuleCause", back_populates="tree_classifier_rule", cascade="all, delete-orphan"
    )

    def getElement(
        self,
        name: Literal["id", "target_value", "probability", "samples_amount"],
    ):
        return super().getElement(name)


class TreeClassifierRuleCause(db.Model, dbInteractionMethods):
    __tablename__ = "tree_classifier_rule_cause"

    id = Column(Integer, primary_key=True)
    predictor = Column(String)
    relation_sign = Column(String)
    value = Column(String)  # encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    tree_classifier_rule_id = Column(
        Integer,
        ForeignKey("tree_classifier_rule.id"),
    )
    tree_classifier_rule = relationship("TreeClassifierRule", back_populates="causes")

    def getElement(
        self,
        name: Literal[
            "id",
            "predictor",
            "relation_sign",
            "value",
        ],
    ):
        return super().getElement(name)

