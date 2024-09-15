import base64
import matplotlib

from dtreeviz import model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from sklearn.tree._tree import Tree, TREE_UNDEFINED

from app.processor.model.dataset_interaction_methods import get_y_transformed

matplotlib.use('SVG')


class ExplainSingleTree:

    @staticmethod
    def get_rules(tree_model: Tree, q_variables, q_variables_values, features, class_names, model_type):
        tree_: Tree = tree_model
        feature_name = [
            features[i] if i != TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def get_simplified_route(route):
            simplified_route = list(route)
            watched_classes_plus = []
            watched_classes_less = []
            done = False
            iteration = 0
            while not done:
                analyzed_rule = {}
                analyzed_item = ""
                analyzed_sign = ""
                analyzed_value = []
                analyzed_type = ""
                for i in range(len(route)):
                    if (iteration + i) < len(route):
                        actual = route[i + iteration]
                        if actual["item"] not in q_variables and (analyzed_type == "number" or not analyzed_type):
                            if not analyzed_item:
                                analyzed_rule = actual
                                analyzed_type = "number"
                                analyzed_item = actual["item"]
                                analyzed_value = actual["value"]
                                if analyzed_item not in watched_classes_plus and actual["sign"] == " > ":
                                    watched_classes_plus.append(analyzed_item)
                                    analyzed_sign = " > "
                                elif analyzed_item not in watched_classes_less and actual["sign"] == " <= ":
                                    watched_classes_less.append(analyzed_item)
                                    analyzed_sign = " <= "
                            elif actual["item"] == analyzed_item and actual["sign"] == analyzed_sign:
                                try:
                                    if analyzed_sign == " > ":
                                        if actual["value"] > analyzed_value:
                                            simplified_route.remove(analyzed_rule)
                                            analyzed_value = actual["value"]
                                            analyzed_rule = actual
                                        elif actual["value"] <= analyzed_value:
                                            simplified_route.remove(actual)
                                    elif analyzed_sign == " <= ":
                                        if actual["value"] <= analyzed_value:
                                            simplified_route.remove(analyzed_rule)
                                            analyzed_value = actual["value"]
                                            analyzed_rule = actual
                                        elif actual["value"] > analyzed_value:
                                            simplified_route.remove(actual)
                                except Exception as e:
                                    str(e)
                                    break
                        elif analyzed_type == "object" or not analyzed_type:
                            if not analyzed_item:
                                analyzed_rule = actual
                                analyzed_type = "object"
                                analyzed_sign = "es :"
                                analyzed_item = actual["item"]
                                analyzed_value = actual["value"]
                            elif actual["item"] == analyzed_item:
                                try:
                                    if all(val in analyzed_value for val in actual["value"]):
                                        simplified_route.remove(analyzed_rule)
                                        analyzed_value = actual["value"]
                                    elif all(analyzed_val in actual["value"] for analyzed_val in analyzed_value):
                                        simplified_route.remove(actual)
                                except Exception as e:
                                    str(e)
                                    break
                    else:
                        break
                iteration += 1
                if iteration >= len(route):
                    done = True
            return simplified_route

        def get_rule(name, threshold, bigger):
            tree_rule = [{
                "item": name,
                "sign": "",
                "value": []
            }]
            if name in q_variables:
                new_args = []
                tree_rule[0]["sign"] = "es: "
                for variables in q_variables_values:
                    if variables["column_name"] == name:
                        for values in variables["variables"]:
                            if (not bigger and float(values["old_value"]) <= threshold) or (
                                    bigger and float(values["old_value"]) > threshold):
                                new_args.append(values["new_value"])
                tree_rule[0]["value"] = new_args
            else:
                if not bigger:
                    tree_rule[0]["sign"] = " <= "
                else:
                    tree_rule[0]["sign"] = " > "
                tree_rule[0]["value"] = [f"{np.round(threshold, 3)}"]

            return tree_rule

        def recurse(node, route, routes):

            if tree_.feature[node] != TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_path_rules, right_path_rules = list(route), list(route)
                left_path_rules += get_rule(name, threshold, bigger=False)
                recurse(tree_.children_left[node], left_path_rules, routes)
                right_path_rules += get_rule(name, threshold, bigger=True)
                recurse(tree_.children_right[node], right_path_rules, routes)
            else:
                simplified_route = get_simplified_route(route)
                simplified_route += [(tree_.value[node], tree_.n_node_samples[node])]
                routes += [simplified_route]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = {"causes": path[:-1]}
            classes = path[-1][0][0]
            position = np.argmax(classes)
            rule["target_value"] = class_names[position] if model_type == "Classifier" else round(classes[position], 4)
            rule["probability"] = np.round(100.0 * classes[position] / np.sum(classes), 2)
            rule["samples_amount"] = path[-1][1]
            rules.append(rule)

        return rules

    @staticmethod
    def graph_tree(tree, x_train, y_train, feature_names, class_names):

        viz = model(
            model=tree,
            X_train=x_train,
            y_train=get_y_transformed(y_train),
            feature_names=feature_names,
            class_names=class_names
        )
        try:
            svg = open(viz.view().save_svg(), "rb").read()
            print("HERE")
            encoded = base64.b64encode(svg)
            svg_encoded = "data:image/svg+xml;base64,{}".format(encoded.decode())
            return svg_encoded, viz
        except Exception as e:
            print(e)
            return ""

    @staticmethod
    def createSurrogateTree(x_train, trainedModel: RandomForestClassifier | RandomForestRegressor, max_depth: int):
        if isinstance(trainedModel, RandomForestClassifier):
            surrogate: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=max_depth, random_state=123)
            surrogate.fit(X=x_train, y=get_y_transformed(trainedModel.predict(x_train)))
        else:
            surrogate: DecisionTreeRegressor = DecisionTreeRegressor(max_depth=max_depth, random_state=123)
            surrogate.fit(X=x_train, y=trainedModel.predict(x_train))
        return surrogate


class SurrogateTree:

    def __init__(self, x_train, model: RandomForestClassifier, df, class_names, target, max_depth: int):
        self.__surrogate_t = DecisionTreeClassifier(max_depth=max_depth)
        self.__x_train = x_train
        self.__y_train = model.predict(x_train)
        self.__surrogate_t.fit(self.__x_train, self.__y_train)
