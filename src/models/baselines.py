from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.config import SEED


def get_baselines() -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent", random_state=SEED)),
        ]),
        "logistic_regression": Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced")),
        ]),
    }
