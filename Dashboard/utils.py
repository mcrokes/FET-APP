def setText(translations, key, keyPath):
    return translations.get(key) if translations.get(key) else f"{keyPath}.{key} "
