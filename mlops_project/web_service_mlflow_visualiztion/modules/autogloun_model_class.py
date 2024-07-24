import mlflow
from autogluon.tabular import TabularPredictor


class AutogluonModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

    def predict(self, context, model_input):
        return self.predictor.predict(model_input)
