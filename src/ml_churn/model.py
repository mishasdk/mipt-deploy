from lightgbm import LGBMClassifier


def build_lgbm(random_state: int = 42, **params) -> LGBMClassifier:
    return LGBMClassifier(random_state=random_state, verbose=-1, **params)
