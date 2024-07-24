import os
from dotenv import load_dotenv
import mlflow
import boto3
from google.cloud import storage as gcs


class MLflowHandler:
    def __init__(self, experiment_name, tracking_uri=None):
        load_dotenv()  # Load environment variables from .env file

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        self.client = mlflow.tracking.MlflowClient()

    def log_params(self, params):
        with mlflow.start_run() as run:
            for key, value in params.items():
                mlflow.log_param(key, value)

    def log_metrics(self, metrics):
        with mlflow.start_run() as run:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

    def log_artifacts(self, local_path, artifact_path=None):
        with mlflow.start_run() as run:
            mlflow.log_artifacts(local_path, artifact_path)

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def upload_to_s3(self, local_file, bucket_name, s3_file):
        s3 = boto3.client("s3")
        s3.upload_file(local_file, bucket_name, s3_file)

    def upload_to_gcs(self, local_file, bucket_name, gcs_file):
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_file)
        blob.upload_from_filename(local_file)

    def download_from_s3(self, s3_file, bucket_name, local_file):
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, s3_file, local_file)

    def download_from_gcs(self, gcs_file, bucket_name, local_file):
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_file)
        blob.download_to_filename(local_file)


# Example usage
if __name__ == "__main__":
    experiment_name = "example_experiment"
    handler = MLflowHandler(experiment_name)

    # Set tracking URI if needed
    # handler.set_tracking_uri("http://your_mlflow_tracking_server")

    # Log parameters and metrics
    params = {"param1": 5, "param2": 10}
    metrics = {"metric1": 0.5, "metric2": 0.8}
    handler.log_params(params)
    handler.log_metrics(metrics)

    # Log artifacts
    handler.log_artifacts("/path/to/local/artifacts")

    # Upload and download files from S3
    handler.upload_to_s3("local_file.txt", "your_bucket_name", "s3_file.txt")
    handler.download_from_s3("s3_file.txt", "your_bucket_name", "downloaded_local_file.txt")

    # Upload and download files from GCS
    handler.upload_to_gcs("local_file.txt", "your_bucket_name", "gcs_file.txt")
    handler.download_from_gcs("gcs_file.txt", "your_bucket_name", "downloaded_local_file.txt")
