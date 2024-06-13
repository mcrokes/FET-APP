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
    "features_description",
    "q_variables_dict",
    "qualitative_variables_saved",
    "target_description",
]


def _initDB_model(self, kwargs):
    for property, value in kwargs.items():
        # depending on whether value is an iterable or not, we must
        # unpack it's value (when **kwargs is request.form, some values
        # will be a 1-element list)
        if hasattr(value, "__iter__") and not isinstance(value, object):
            # the ,= unpack of a singleton fails PEP8 (travis flake8 test)
            print(property, ": Added Via Has iter")
            value = value[0]
        # if property == "model" or property == "dataset":
        if property in ENCODED_FIELDS:
            print(property, ": Added Via pikle dumps")
            value = pickle.dumps(value)

        print(property, ": Added")
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
    qualitative_variables_saved = Column(String)  # Encoded
    target_description = Column(String)  # Encoded

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
        "DataSetData", uselist=False, back_populates="explained_classifier_model"
    )
    confusion_matrixes_data = relationship(
        "ConfusionMatrixesData", back_populates="explained_classifier_model"
    )
    roc_data = relationship("ROCData", back_populates="explained_classifier_model")
    importances_data = relationship(
        "ImportancesData", uselist=False, back_populates="explained_classifier_model"
    )
    permutation_importances_data = relationship(
        "PermutationImportancesData",
        uselist=False,
        back_populates="explained_classifier_model",
    )
    surrogate_trees_data = relationship(
        "SurrogateTreeClassifierData", back_populates="explained_classifier_model"
    )
    inner_trees_data = relationship(
        "InnerTreeClassifierData", back_populates="explained_classifier_model"
    )

    type,

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
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="data_set_data"
    )

    data_set_data_distributions = relationship(
        "DataSetDataDistribution", back_populates="data_set_data"
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


class ImportancesData(db.Model, dbInteractionMethods):

    __tablename__ = "importances_data"

    explanation = Column(String)  # Encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", uselist=False, back_populates="importances_data"
    )

    def getElement(
        self,
        name: Literal["explanation"],
    ):
        return super().getElement(name)


class PermutationImportancesData(db.Model, dbInteractionMethods):

    __tablename__ = "permutation_importances_data"

    explanation = Column(String)  # Encoded

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id"), primary_key=True
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel",
        uselist=False,
        back_populates="permutation_importances_data",
    )

    permutation_importances = relationship(
        "PermutationImportance", back_populates="permutation_importances_data"
    )

    def getElement(
        self,
        name: Literal["explanation"],
    ):
        return super().getElement(name)


class PermutationImportance(db.Model, dbInteractionMethods):

    __tablename__ = "permutation_importance"

    id = Column(Integer, primary_key=True)
    importance_mean = Column(Float)
    importance_std = Column(Float)
    predictor = Column(String)

    def __init__(self, **kwargs):
        _initDB_model(self, kwargs)

    permutation_importances_data_id = Column(
        Integer,
        ForeignKey("permutation_importances_data.explained_classifier_model_id"),
    )
    permutation_importances_data = relationship(
        "PermutationImportancesData", back_populates="permutation_importances"
    )

    def getElement(
        self,
        name: Literal[
            "id",
            "importance_mean",
            "importance_std",
            "predictor",
        ],
    ):
        return super().getElement(name)


class Tree(db.Model):

    __tablename__ = "tree"

    id = Column(Integer, primary_key=True)
    depth = Column(Integer)
    rules_amount = Column(Integer)
    inexact_rules_amount = Column(Integer)

    __mapper_args__ = {
        "polymorphic_identity": "tree",
    }

    surrogate_tree = relationship(
        "SurrogateTreeClassifierData", uselist=False, back_populates="tree"
    )

    inner_tree = relationship(
        "InnerTreeClassifierData", uselist=False, back_populates="tree"
    )

    rules = relationship("TreeClassifierRule", back_populates="tree_classifier")


class SurrogateTreeClassifierData(db.Model):

    __tablename__ = "surrogate_tree_classifier_data"
    __mapper_args__ = {
        "polymorphic_identity": "surrogate_tree_classifier_data",
    }

    tree_model = Column(String)  # Encoded

    tree_id = Column(Integer, ForeignKey("tree.id"), primary_key=True)
    tree = relationship("Tree", uselist=False, back_populates="surrogate_tree")

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id")
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel",
        uselist=False,
        back_populates="surrogate_trees_data",
    )


class InnerTreeClassifierData(db.Model):

    __tablename__ = "inner_tree_classifier_data"

    __mapper_args__ = {
        "polymorphic_identity": "inner_tree_classifier_data",
    }

    tree_number = Column(Integer)

    tree_id = Column(Integer, ForeignKey("tree.id"), primary_key=True)
    tree = relationship("Tree", uselist=False, back_populates="inner_tree")

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id")
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel",
        uselist=False,
        back_populates="inner_trees_data",
    )


class TreeClassifierRule(db.Model):
    __tablename__ = "tree_classifier_rule"

    id = Column(Integer, primary_key=True)
    target_value = Column(String)
    probability = Column(Float)
    samples_amount = Column(Integer)

    tree_id = Column(
        Integer,
        ForeignKey("tree.id"),
    )
    tree_classifier = relationship("Tree", back_populates="rules")

    causes = relationship(
        "TreeClassifierRuleCause", back_populates="tree_classifier_rule"
    )


class TreeClassifierRuleCause(db.Model):
    __tablename__ = "tree_classifier_rule_cause"

    id = Column(Integer, primary_key=True)
    predictor = Column(String)
    relation_sign = Column(String)
    value = Column(String)

    tree_classifier_rule_id = Column(
        Integer,
        ForeignKey("tree_classifier_rule.id"),
    )
    tree_classifier_rule = relationship("TreeClassifierRule", back_populates="causes")


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


class ROCData(db.Model):

    __tablename__ = "roc_data"

    id = Column(Integer, primary_key=True)
    class_name = Column(String)
    fpr = Column(String)  # Encoded
    tpr = Column(String)  # Encoded
    auc_score = Column(String)  # Encoded

    explained_classifier_model_id = Column(
        Integer, ForeignKey("explained_classifier_model.id")
    )
    explained_classifier_model = relationship(
        "ExplainedClassifierModel", back_populates="roc_data"
    )
