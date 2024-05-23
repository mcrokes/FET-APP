import base64

from dtreeviz.trees import model
from sklearn import tree
from sklearn.tree import _tree
import numpy as np


class SurrogateTree:
    def __init__(self, explainer_classifier, max_depth: int):
        self.__surrogate_t = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.__x_train = explainer_classifier.get_x_train()
        self.__y_train = explainer_classifier.predict(explainer_classifier.get_x_train())
        self.__surrogate_t.fit(self.__x_train, self.__y_train)
        # self.__importance = self.__surrogate_t.feature_importances_
        # self.__indices = np.argsort(self.__importance)
        self.__q_variables = explainer_classifier.get_q_variables_values_names()
        self.__q_variables_values = explainer_classifier.get_q_variables_values_list()
        self.__target = explainer_classifier.get_target_classifier()
        self.__class_names = explainer_classifier.get_target_class_names()
        self.__feature_names = explainer_classifier.get_feature_names()
        self.__df = explainer_classifier.get_dataframe()

    # def get_text_representation(self):
    #     return tree.export_text(self.__cls_t, feature_names=self.features)

    def get_rules(self):
        tree_ = self.__surrogate_t.tree_
        feature_name = [
            self.__feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []
        cont = 1

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
                        if actual["item"] not in self.__q_variables and (
                                analyzed_type == "number" or not analyzed_type):
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
            rule = [{
                "item": name,
                "sign": "",
                "value": []
            }]
            if name in self.__q_variables:
                new_args = []
                first = True
                # condition = "es : "
                rule[0]["sign"] = "es: "
                for variables in self.__q_variables_values:
                    if variables["column_name"] == name:
                        for values in variables["variables"]:
                            print(values["old_value"])
                            print(threshold)
                            if (not bigger and float(values["old_value"]) <= threshold) or (
                                    bigger and float(values["old_value"]) > threshold):
                                new_args.append(values["new_value"])
                rule[0]["value"] = new_args
            else:
                if not bigger:
                    rule[0]["sign"] = " <= "
                else:
                    rule[0]["sign"] = " > "
                rule[0]["value"] = [f"{np.round(threshold, 3)}"]

            return rule

        def get_text_route(route):
            text_route = []
            for val in route:
                if val["item"] not in self.__q_variables:
                    text_route += [f"({val['item']} {val['sign']} {val['value'][0]})"]
                else:
                    values = ""
                    first = True

                    for i in range(len(val['value'])):
                        if first:
                            values += f"{val['value'][i]}"
                            first = False
                        else:
                            values += f" o {val['value'][i]}"
                    text_route += [f"{val['item']} {val['sign']} {values}"]
            return text_route

        def recurse(node, route, routes):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_path_rules, right_path_rules = list(route), list(route)
                left_path_rules += get_rule(name, threshold, bigger=False)
                recurse(tree_.children_left[node], left_path_rules, routes)
                right_path_rules += get_rule(name, threshold, bigger=True)
                recurse(tree_.children_right[node], right_path_rules, routes)
            else:
                simplified_route = get_simplified_route(route)
                text_route = get_text_route(simplified_route)
                text_route += [(tree_.value[node], tree_.n_node_samples[node])]
                routes += [text_route]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "Si "

            for p in path[:-1]:
                if rule != "Si ":
                    rule += " y "
                rule += str(p)
            rule += " entonces "
            if self.__class_names is None:
                rule += "respuesta: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                position = np.argmax(classes)
                rule += f"{self.__target}: {self.__class_names[position]} (con probabilidad: " \
                        f"{np.round(100.0 * classes[position] / np.sum(classes), 2)}%)"
            rule += f" | basado en {path[-1][1]:,} muestras"
            rules += [f"Regla {cont}: {rule}"]
            cont += 1

        return rules

    def graph_tree(self):

        viz = model(
            self.__surrogate_t,
            X_train=self.__x_train,
            y_train=self.__y_train,
            feature_names=self.__feature_names,
            class_names=self.__class_names
        )
        try:
            svg = open(viz.view().save_svg(), "rb").read()
            encoded = base64.b64encode(svg)
            svg_encoded = "data:image/svg+xml;base64,{}".format(encoded.decode())
            return svg_encoded
        except Exception as e:
            print(e)
            return ""
