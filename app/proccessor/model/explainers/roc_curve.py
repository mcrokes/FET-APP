import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc


class ROC:
    @staticmethod
    def create_curve(y_scores, y_true, options):
        # One hot encode the labels in order to plot them
        # y_onehot = pd.get_dummies(y_true)
        # print(y_onehot)

        data = []
        trace1 = go.Scatter(x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash'),
                            showlegend=False

                            )

        data.append(trace1)
        cont = 0
        for i in range(y_scores.shape[1]):
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=i)
            auc_score = auc(fpr, tpr)

            name = f"{options[cont]['label']} (AUC={auc_score:.2f})"
            trace2 = go.Scatter(x=fpr, y=tpr,
                                name=name,
                                mode='lines')
            data.append(trace2)
            cont += 1

        layout = go.Layout(
            title='ROC-AUC curva',
            yaxis=dict(title='Tasa de Positivos'),
            xaxis=dict(title='Tasa de Falsos Positivos')
        )

        fig = go.Figure(data=data, layout=layout)

        return fig
