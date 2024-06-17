from ..dataset_interaction_methods import update_y_pred
from sklearn import metrics
import plotly.express as px


class ConfusionMatrix:
    @staticmethod
    def __create_matrix(y_test, y_pred, class_names):
        # Generate the confusion matrix
        cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

        fig = px.imshow(img=cm,
                        title="MATRIZ DE CONFUSION",
                        labels=dict(x="PREDICCIONES", y="VALORES REALES", color="CANTIDAD"),
                        x=class_names,
                        y=class_names,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        )
        return fig

    @staticmethod
    def initialize_matrix(dropdown_value, slider_value, data):
        y_pred = data.predict(data.get_x_test())

        if dropdown_value is not None:
            positive_class = int(dropdown_value)
            probability_predictions = data.predict_proba(data.get_x_test())

            try:
                y_pred = update_y_pred(prediction=y_pred, probability_predictions=probability_predictions,
                                       cut_off=slider_value, positive_class=positive_class)

            except Exception as e:
                print(e)

        return ConfusionMatrix.__create_matrix(y_test=data.get_y_test_transformed(), y_pred=y_pred,
                                               class_names=data.get_target_class_names())
