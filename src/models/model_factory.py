from typing import Any, Dict

from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor


def build_model(model_family: str, params: Dict[str, Any]) -> Any:
    builders = {
        "lightgbm": lambda p: LGBMRegressor(**p),
        "random_forest": lambda p: RandomForestRegressor(**p),
        "extra_trees": lambda p: ExtraTreesRegressor(**p),
        "ridge": lambda p: Ridge(**p),
        "lasso": lambda p: Lasso(**p),
        "elasticnet": lambda p: ElasticNet(**p),
        "gradient_boosting": lambda p: GradientBoostingRegressor(**p),
        "hist_gradient_boosting": lambda p: HistGradientBoostingRegressor(**p),
        "adaboost": lambda p: AdaBoostRegressor(**p),
        "decision_tree": lambda p: DecisionTreeRegressor(**p),
    }
    if model_family not in builders:
        raise ValueError(
            f"Unknown model family '{model_family}'. Available: {sorted(builders.keys())}"
        )
    return builders[model_family](params)
