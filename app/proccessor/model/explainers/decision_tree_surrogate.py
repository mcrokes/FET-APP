import base64
import matplotlib

from dtreeviz.trees import model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np

matplotlib.use('SVG')

class ExplainSingleTree:
    
    @staticmethod
    def get_rules(model, q_variables, q_variables_values, features, class_names, target):
        tree_ = model
        feature_name = [
            features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
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
            while (not done):
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
            rule = [{
                "item": name,
                "sign": "",
                "value": []
            }]
            if name in q_variables:
                new_args = []
                # condition = "es : "
                rule[0]["sign"] = "es: "
                for variables in q_variables_values:
                    if variables["column_name"] == name:
                        for values in variables["variables"]:
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
                simplified_route += [(tree_.value[node], tree_.n_node_samples[node])]
                routes += [simplified_route]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            # rule = "Si "
            rule = {}
            
            rule["causes"] = path[:-1]
            classes = path[-1][0][0]
            position = np.argmax(classes)
            rule["target_value"] = class_names[position]
            rule["probability"] = np.round(100.0 * classes[position] / np.sum(classes), 2)
            rule["samples_amount"] = path[-1][1]
            rules.append(rule)

        return rules
    
    @staticmethod
    def graph_tree(tree, x_train, y_train, feature_names, class_names):
        
        viz = model(
            model=tree,
            X_train=x_train,
            y_train=y_train,
            feature_names=feature_names,
            class_names=class_names
        )
        try:
            svg = open(viz.view().save_svg(), "rb").read()
            encoded = base64.b64encode(svg)
            svg_encoded = "data:image/svg+xml;base64,{}".format(encoded.decode())
            return svg_encoded
        except Exception as e:
            print(e)
            return ""
    
    @staticmethod
    def createSurrogateTree(x_train, model: RandomForestClassifier, max_depth: int):
        surrogate: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=max_depth)
        surrogate.fit(X=x_train, y=model.predict(x_train))
        return surrogate
     
class SurrogateTree:
    
    
    
    def __init__(self, x_train, model: RandomForestClassifier, df, class_names, target, max_depth: int):
        self.__surrogate_t = DecisionTreeClassifier(max_depth=max_depth)
        self.__x_train = x_train
        self.__y_train = model.predict(x_train)
        self.__surrogate_t.fit(self.__x_train, self.__y_train)

    
