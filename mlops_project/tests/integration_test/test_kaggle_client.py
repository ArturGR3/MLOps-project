import os
from dotenv import load_dotenv, find_dotenv
import sys

load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
project_path = os.getenv("PROJECT_PATH")
print(f"project path {project_path}")
if project_path not in sys.path:
    sys.path.append(project_path)
from modules.kaggle_client import KaggleClient


def test_kaggle_integration():
    competition_name = "playground-series-s3e11"
    target = 'cost'
    model_name = "Autogluon_test"
    message = "Integration Test Submission"
    client = KaggleClient(competition_name, target)
    data_path = os.path.join(project_path, f"data/{competition_name}/raw")

    # Test download_data method
    
    client.download_data()
    assert os.path.exists(f"{data_path}/sample_submission.csv")

    # Test submit method
    submission_file = f"{data_path}/sample_submission.csv"
    score = client.submit(submission_file, model_name, message)
    assert score is not None
