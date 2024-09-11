from sklearn.preprocessing import LabelEncoder

from app.API.routes import find_translations


def setText(translations, key, keyPath):
    return translations.get(key) if translations.get(key) else f"{keyPath}.{key} "


def getTranslations(lang, section, t_type):
    return find_translations(lang, ['dashboard', section, t_type])['text']


def findTranslationsParent(translations, key):
    return translations.get(key) if translations.get(key) else {}


def get_target_dropdown(values_dict):
    return [
        {"label": value["new_value"], "value": index}
        for index, value in enumerate(values_dict)
    ]


def get_y_test_transformed(y_test):
    for value in y_test:
        if value is not int:
            y_test = LabelEncoder().fit_transform(y_test)
            break
    return y_test
