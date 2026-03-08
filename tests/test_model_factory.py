"""Tests for src.models.model_factory."""

import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor

from src.models.model_factory import build_model


class TestBuildModel:
    def test_lightgbm(self):
        model = build_model("lightgbm", {"n_estimators": 10})
        assert isinstance(model, LGBMRegressor)

    def test_random_forest(self):
        model = build_model("random_forest", {"n_estimators": 10})
        assert isinstance(model, RandomForestRegressor)

    def test_extra_trees(self):
        model = build_model("extra_trees", {"n_estimators": 10})
        assert isinstance(model, ExtraTreesRegressor)

    def test_ridge(self):
        model = build_model("ridge", {"alpha": 1.0})
        assert isinstance(model, Ridge)

    def test_lasso(self):
        model = build_model("lasso", {"alpha": 1.0})
        assert isinstance(model, Lasso)

    def test_elasticnet(self):
        model = build_model("elasticnet", {"alpha": 1.0})
        assert isinstance(model, ElasticNet)

    def test_decision_tree(self):
        model = build_model("decision_tree", {"max_depth": 5})
        assert isinstance(model, DecisionTreeRegressor)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown model family"):
            build_model("nonexistent_model", {})

    def test_params_passed_through(self):
        model = build_model("ridge", {"alpha": 2.5})
        assert model.alpha == 2.5
