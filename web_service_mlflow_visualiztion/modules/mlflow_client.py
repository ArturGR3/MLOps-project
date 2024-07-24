import mlflow
import time
import os
import subprocess
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from mlflow.pyfunc import PythonModel

# from modules.kaggle_client import KaggleClient

# from metrics.py
from sklearn.metrics import mean_squared_log_error
import numpy as np


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


class MLflowAutoGluon:
    def __init__(
        self,
        tracking_server,
        backend_store,
        artifact_location,
        experiment_name,
        competition_name,
        target_name,
    ):
        self.tracking_server = tracking_server
        self.backend_store = backend_store
        self.artifact_location = artifact_location
        self.experiment_name = experiment_name
        self.competition_name = competition_name
        self.target_name = target_name
        self.setup_mlflow()

    class AutogluonModel(PythonModel):
        def load_context(self, context):
            self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

        def predict(self, context, model_input):
            return self.predictor.predict(model_input)

    @staticmethod
    def check_port_in_use(port):
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()

    @staticmethod
    def kill_process_on_port(port=5000):
        in_use, pids = MLflowAutoGluon.check_port_in_use(port)
        if in_use:
            for pid in pids.split("\n"):
                subprocess.run(["kill", "-9", pid])
            print(f"Killed processes on port {port}: {pids}")
        else:
            print(f"No process to kill on port {port}")

    def start_mlflow_server(self, port=5000):
        self.kill_process_on_port(port)
        with open(os.devnull, "wb") as devnull:
            subprocess.Popen(
                ["mlflow", "server", "--backend-store-uri", f"sqlite:///{self.backend_store}"],
                stdout=devnull,
                stderr=devnull,
                stdin=devnull,
                close_fds=True,
            )
        time.sleep(5)
        in_use, _ = self.check_port_in_use(port)
        if in_use:
            print(f"MLflow server is running on port {port}")
        else:
            print(f"Failed to start MLflow server on port {port}")

    def create_mlflow_experiment(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
            experiment = mlflow.get_experiment(experiment_id)
            print(f"Experiment {self.experiment_name} created")
        else:
            print(f"Experiment {self.experiment_name} already exists")

        print(f"Experiment Name: {experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Creation timestamp: {experiment.creation_time}")

    def setup_mlflow(self):
        if self.tracking_server == "no":
            print("Local setup with no tracking server")
            mlflow.set_tracking_uri(f"file://{self.backend_store}")
        elif self.tracking_server == "local":
            print("Local setup with Local tracking server")
            self.start_mlflow_server()
            mlflow.set_tracking_uri(f"sqlite:///{self.backend_store}")
        elif self.tracking_server == "remote":
            print("Remote setup with remote tracking server")
            self.kill_process_on_port()
            mlflow.set_tracking_uri(f"http://{self.backend_store}:5000")
        self.create_mlflow_experiment()
        mlflow.set_experiment(self.experiment_name)
        print(f"Current tracking URI: {mlflow.get_tracking_uri()}")

    def train_and_log_model(
        self,
        presets,
        target,
        train_transformed,
        test_transformed,
        run_time,
        for_deployment=True,
        for_kaggle_submission=False,  # Changed to False to disable Kaggle submission
    ):
        rmsle = make_scorer(
            "rmsle", root_mean_squared_log_error, greater_is_better=False, needs_proba=False
        )

        for preset in presets:
            with mlflow.start_run(run_name=f"{preset}") as parent_run:
                predictor = TabularPredictor(
                    label=target, path=f"AutoGluon_mlflow_{preset}", eval_metric=rmsle
                ).fit(
                    train_data=train_transformed,
                    time_limit=run_time * 60,
                    presets=preset,
                    excluded_model_types=["KNN", "NN"],
                )

                test_pred = predictor.predict(test_transformed)
                leaderboard = predictor.leaderboard(silent=True)
                best_model = leaderboard.loc[leaderboard["score_val"].idxmax()]
                mlflow.log_metrics(
                    {
                        "score_val": round(-best_model["score_val"], 4),
                        "inference_time": round(best_model["pred_time_val"], 4),
                        "fit_time": round(best_model["fit_time"], 4),
                    }
                )

                for index, row in leaderboard.iterrows():
                    with mlflow.start_run(run_name=row["model"], nested=True) as child_run:
                        mlflow.set_tags({"model_name": row["model"]})
                        mlflow.log_metrics(
                            {
                                "score_val": round(-row["score_val"], 4),
                                "inference_time": round(row["pred_time_val"], 4),
                                "fit_time": round(row["fit_time"], 4),
                            }
                        )

                predictor.refit_full()

                # if for_kaggle_submission:
                #     kaggle_client = KaggleClient(self.competition_name, target)
                #     submission_path = f"data/{self.competition_name}/submission_files"
                #     os.makedirs(submission_path, exist_ok=True)

                #     submission = pd.read_csv(
                #         f"data/{self.competition_name}/raw/sample_submission.csv"
                #     )
                #     submission[target] = predictor.predict(test_transformed)
                #     submission_file = f"{submission_path}/sub_{run_time}_{preset}.csv"
                #     submission.to_csv(submission_file, index=False)

                #     kaggle_score = kaggle_client.submit(
                #         submission_file=submission_file,
                #         model_name=preset,
                #         message=f"AutoGluon {preset} {run_time} min",
                #     )
                #     mlflow.log_metric("kaggle_score", kaggle_score)

                save_path = f"AutoGluon_mlflow_{preset}" + (
                    "_deployment" if for_deployment else ""
                )
                predictor_clone = (
                    predictor.clone_for_deployment(
                        path=save_path, return_clone=True, dirs_exist_ok=True
                    )
                    if for_deployment
                    else predictor.save(save_path)
                )

                # if for_kaggle_submission:
                #     submission.to_csv(f"{save_path}/sub_{run_time}_{preset}.csv", index=False)

                model = self.AutogluonModel()
                mlflow.pyfunc.log_model(
                    artifact_path=save_path,
                    python_model=model,
                    artifacts={"predictor_path": save_path},
                )

            mlflow.end_run()
