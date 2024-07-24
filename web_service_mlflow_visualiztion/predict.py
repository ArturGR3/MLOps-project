import pandas as pd
import os
import time 
from autogluon.tabular import TabularPredictor, TabularDataset
from flask import Flask, request, jsonify
import boto3
from download_folder_s3 import download_s3_folder, list_files_in_s3_folder
from dotenv import load_dotenv, find_dotenv
from prometheus_client import start_http_server, Summary, Counter, Histogram
import threading

load_dotenv(find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=True))

# Specify your bucket name and model file path
AWS_BUCKER_NAME = os.getenv("AWS_BUCKET_NAME")
experiment_id = os.getenv("EXPERIMENT_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize the S3 client
s3 = boto3.client("s3")

# Download the folder from s3
s3_model_folder = f"{experiment_id}/artifacts/AutoGluon_mlflow_best_quality_deployment/artifacts/AutoGluon_mlflow_best_quality_deployment/"
local_model_path = "model_ag_deployment"
# Ensure the local folder path exists
os.makedirs(local_model_path, exist_ok=True)

# download the model from s3
download_s3_folder(
    bucket_name=AWS_BUCKER_NAME, s3_folder=s3_model_folder, local_dir=local_model_path
)

# Load the model from the local file
try:
    predictor = TabularPredictor.load("model_ag_deployment")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Prometheus metrics
REQUEST_TIME = Histogram('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNTER = Counter('request_count', 'Number of requests received')

# Create Flask application
app = Flask("prediction")

# Define predict endpoint
@app.route("/predict", methods=["POST"])
@REQUEST_TIME.time()
def predict_endpoint():
    REQUEST_COUNTER.inc()
    start_time = time.time()
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the data to a TabularDataset for AutoGluon model
        data_df = TabularDataset([data])

        # Make the prediction using the loaded model
        pred = predictor.predict(data_df)[0]
        result = {"cost": round(float(pred), 2)}

        # Return the prediction result as JSON
        return jsonify(result)
    except Exception as e:
        # Return error message if an exception occurs
        return jsonify({"error": str(e)}), 400
    finally:
        # Calculate the inference time
        duration = time.time() - start_time
        REQUEST_TIME.observe(duration)

# Function to start the Prometheus metrics server
def start_metrics_server():
    start_http_server(8000)

# Start the metrics server in a separate thread
threading.Thread(target=start_metrics_server, daemon=True).start()

if __name__ == "__main__":
    start_http_server(8000)  # Start the Prometheus metrics server
    app.run(host="0.0.0.0", port=9696, debug=True)
