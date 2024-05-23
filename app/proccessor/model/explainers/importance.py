import multiprocessing

import plotly.express as px
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder


class Importance:
    @staticmethod
    def create_importance(features, importance):
        df_feature_importance = pd.DataFrame({'Predictor': features, 'Importancia': importance})
        fig = px.bar(data_frame=df_feature_importance.sort_values('Importancia', ascending=False), x='Importancia',
                     y='Predictor',
                     title='Importance GINI')
        return fig

    @staticmethod
    def create_permutation_importance(model, x_train, y_train, features):
        permutation_importance_model = permutation_importance(
            estimator=model,
            X=x_train,
            y=y_train,
            n_repeats=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=multiprocessing.cpu_count() - 1,
            random_state=123
        )
        df_permutation_importance = pd.DataFrame(
            {k: permutation_importance_model[k] for k in ['importances_mean', 'importances_std']})
        df_permutation_importance['Predictor'] = features
        df_ordered_importance = df_permutation_importance.sort_values('importances_mean', ascending=True)
        fig = px.bar(data_frame=df_ordered_importance, error_x=df_ordered_importance['importances_std'],
                     x='importances_mean', y='Predictor',
                     title='IMPORTANCIAS POR PERMUTACION',
                     labels={'importances_mean': 'Incremento del error tras la permutacion'})

        return fig
