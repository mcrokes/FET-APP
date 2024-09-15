import pandas as pd
import joblib as jb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .rules_uploader import graph_rules
from .utils import graph_card
from .explainers.confusion_matrix import ConfusionMatrix
from .explainers.datasets import create_data_table
from .explainers.decision_tree_surrogate import SurrogateTree
from .explainers.importance import Importance
from .explainers.roc_curve import ROC
from .values import get_target_dropdown


# Esta clase funciona como asimilador de cada modelo entrenado con sus datos descriptivos, así como el dataset con que
# fue entrenado
# ====================================================================
class TrainedModel:

    # Inicializador de clase
    def __init__(self,
                 name,
                 df: pd.DataFrame,
                 predictors_description,
                 target: str,
                 test_size,
                 random_state,
                 model,
                 model_description: str,
                 q_variables_values_list):
        self.__name = name
        self.__df = df
        self.__trained_model: RandomForestClassifier | RandomForestRegressor = model
        self.__model_description = model_description
        self.__predictors_description = predictors_description
        self.__X_train, self.__X_test, self.__y_train, self.__y_test, self.__target_values = self.__sort_dataset(
            target, test_size, random_state)
        self.__target = target
        self.__q_variables_values_list = q_variables_values_list
        self.__q_variables_values_names = self.get_q_variables_values_names()

    def get_dataframe(self):
        return self.__df

    def get_feature_names(self):
        return list(self.__df.drop(columns=self.__target).columns)

    def get_name(self):
        return self.__name

    def get_target_values(self):
        return self.__target_values

    def get_trained_model_pure(self):
        return self.__trained_model

    def get_model_description(self):
        return self.__model_description

    def get_predictors_description(self):
        return self.__predictors_description

    def get_x_train(self):
        return self.__X_train

    def get_x_test(self):
        return self.__X_test

    def get_y_train(self):
        return self.__y_train

    def get_y_test(self):
        return self.__y_test

    def get_q_variables_values_list(self):
        return self.__q_variables_values_list

    def get_q_variables_values_names(self):
        try:
            return self.__q_variables_values_names
        except Exception as e:
            str(e)
            variables = []
            for variable in self.__q_variables_values_list:
                print(variable)
                variables.append(variable['column_name'])
            return variables

    # Aqui se divide el dataset en las variables de entrenamiento y testeo
    def __sort_dataset(self,
                       target,
                       test_size,
                       random_state):
        data = self.__df.drop(columns=target)
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            self.__df[target], test_size=test_size,
                                                            random_state=random_state)
        x_train.index = [i for i in range(len(x_train))]
        print()
        x_train.index.name = "Identifier"
        x_test.index = [i for i in range(len(x_test))]
        x_test.index.name = "Identifier"
        return x_train, x_test, y_train, y_test, self.__df[target]

    def predict(self, x_test):
        return self.__trained_model.predict(x_test)

    def predict_proba(self, x_test):
        return self.__trained_model.predict_proba(x_test)

    # con esta funcion se reentrena el modelo
    def retrain_model(self,
                      n_estimators,
                      criterion,
                      max_depth,
                      max_features,
                      oob_score,
                      random_state):
        # RetrainValues = [n_estimators, criterion, max_depth, max_features, oob_score, random_state]
        # self.trainingDepperndenciesModelList.append(RetrainValues)
        # self.reentrenamientos += 1
        pass

    # SERIALIZACIÓN DEL MODELO ENTRENADO
    def save_model(self, url: str):
        # Salva del modelo
        jb.dump(self.__trained_model, url)

        # Carga del modelo
        # modeloCargado = joblib.load('modeloCargado.joblib')

        return True


# Esta clase funciona como asimilador de cada modelo entrenado especificamente de Regresión
# ====================================================================
class RegressionTrainedModel(TrainedModel):

    # Inicializador de clase
    def __init__(self,
                 name,
                 measurement_unity,
                 df,
                 predictors_description,
                 target,
                 test_size,
                 random_state,
                 model,
                 model_description,
                 q_variables_values_list):
        super().__init__(name, df, predictors_description, target, test_size, random_state, model, model_description,
                         q_variables_values_list)
        self.__trained_model = model
        self.__measurement_unity = measurement_unity

    # con esta funcion se reentrena el modelo
    def retrain_model(self,
                      n_estimators,
                      criterion,
                      max_depth,
                      max_features,
                      oob_score,
                      random_state):
        super().retrain_model(n_estimators, criterion, max_depth, max_features, oob_score, random_state)

        modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            oob_score=oob_score,
            random_state=random_state
        )
        modelo.fit(self.__X_train, self.__y_train)
        self.__trained_model = modelo

    def get_measurement_unity(self):
        return self.__measurement_unity


# Esta clase funciona como asimilador de cada modelo entrenado especificamente de Clasificación
# ====================================================================
class ClassificationTrainedModel(TrainedModel):

    # Class initializer
    def __init__(self,
                 name: str,
                 df,
                 predictors_description,
                 target,
                 test_size,
                 random_state,
                 model,
                 model_description,
                 q_variables_values_list,
                 target_description):
        super().__init__(name, df, predictors_description, target, test_size, random_state, model, model_description,
                         q_variables_values_list)

        self.__target_classifier = target
        self.__target_values = self.get_target_values()

        self.__target_class_names = []
        self.__possible_target_values_classification_dict = self.__update_target_values_classification_dict(
            target_description)
        self.__model_visualization = self.create_model_visualization()

    # CREATE TARGET DESCRIPTOR
    def get_model_visualization(self):
        return self.__model_visualization

    def get_target_class_names(self):
        return self.__target_class_names

    def get_target_classifier(self):
        return self.__target_classifier

    # Retrain the model
    def retrain_model(self,
                      n_estimators,
                      criterion,
                      max_depth,
                      max_features,
                      oob_score,
                      random_state):
        super().retrain_model(n_estimators, criterion, max_depth, max_features, oob_score, random_state)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            oob_score=oob_score,
            random_state=random_state
        )
        model.fit(self.__X_train, self.__y_train)
        self.__trained_model = model

    def get_y_train_transformed(self):
        for value in self.get_y_train():
            if value is not int:
                encoded_value = LabelEncoder().fit_transform(self.get_y_train())
                return encoded_value
        return self.get_y_train()

    def get_y_test_transformed(self):
        for value in self.get_y_test():
            if value is not int:
                self.__y_test = LabelEncoder().fit_transform(self.get_y_test())
                break
        return self.get_y_test()

    def __update_target_values_classification_dict(self, target_description):
        values = set(self.__target_values.array)
        self.__target_class_names = []
        if target_description:
            for name in target_description['variables']:
                self.__target_class_names.append(name['new_value'])
            return target_description
        else:
            possible_target_values_classification_dict = {
                'column_name': self.__target_classifier,
                'variables': []
            }
            for value in values:
                self.__target_class_names.append(value)
                possible_target_values_classification_dict['variables'].append(
                    {
                        'old_value': value,
                        'new_value': value,
                    }
                )

        return possible_target_values_classification_dict

    def get_target_values_classification_dict(self):
        if not self.__possible_target_values_classification_dict:
            return self.__update_target_values_classification_dict([])
        return self.__possible_target_values_classification_dict

    def create_model_visualization(self):
        print("Creating data_table...")
        dtt, dt = create_data_table(self)
        print("Creating importance...")
        gi = graph_card(
            Importance.create_importance(
                features=self.get_feature_names(),
                importance=self.get_trained_model_pure().feature_importances_
            )
        )
        print("Creating permutation_importance...")
        pi = graph_card(
            Importance.create_permutation_importance(
                model=self.get_trained_model_pure(),
                x_train=self.get_x_train(), y_train=self.get_y_train_transformed(),
                features=self.get_feature_names()
            )
        )
        print("Creating SurrogateTree...")
        surrogate_model = SurrogateTree(self, 5)
        print("Creating SurrogateTree rules...")
        rg = graph_rules(surrogate_model.get_rules())
        print("Creating SurrogateTree tree...")
        tg = surrogate_model.graph_tree()
        print("Creating matrix...")
        ms = graph_card(ConfusionMatrix.initialize_matrix(None, None, self))
        print("Creating ROC...")
        rs = graph_card(
            ROC.create_curve(
                self.predict_proba(self.get_x_test()), self.get_y_test_transformed(),
                get_target_dropdown(self.__possible_target_values_classification_dict)
            )
        )
        print(" ---------CREATED EXPLAINER-----------")

        return dtt, dt, gi, pi, rg, tg, ms, rs, get_target_dropdown(self.__possible_target_values_classification_dict)
