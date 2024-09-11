from app.API.routes import find_translations


def setText(translations, key, keyPath):
    return translations.get(key) if translations.get(key) else f"{keyPath}.{key} "


def getTranslations(lang, section, t_type):
    return find_translations(lang, ['dashboard', section, t_type])['text']


def findTranslationsParent(translations, key):
    return translations.get(key) if translations.get(key) else {}