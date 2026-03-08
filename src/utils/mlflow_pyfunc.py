import joblib
import mlflow.pyfunc


class SklearnJoblibPyfuncModel(mlflow.pyfunc.PythonModel):
    """Load a compressed joblib sklearn model as an MLflow pyfunc."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = joblib.load(context.artifacts["model_file"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input):
        return self.model.predict(model_input)
