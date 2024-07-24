import os
import zipfile
import numpy as np
import time
import pandas as pd
from dotenv import load_dotenv, find_dotenv


# Load environment variables from .env file before import KaggleAPI, so it does not fail
load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")


from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient


class KaggleClient:
    """
    A class for interacting with the Kaggle API and performing operations related to Kaggle competitions.

    Args:
        competition_name (str): The name of the Kaggle competition.
        target_column (str): The name of the target column in the competition dataset.

    Methods:
        download_data(): Downloads the competition data files from Kaggle.
        submit(submission_file, model_name, message): Submits a competition submission file to Kaggle and retrieves the submission score.

    """

    def __init__(self, competition_name, target_column):
        self.api = KaggleApi(ApiClient())
        self.api.authenticate()
        self.competition_name = competition_name
        self.target_column = target_column

    def download_data(self):
        """
        Downloads the competition data files from Kaggle and extracts them to the specified data path.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, f"data/{self.competition_name}/raw")
        os.makedirs(data_path, exist_ok=True)
        self.api.competition_download_files(self.competition_name, path=data_path)
        zip_file = os.path.join(data_path, f"{self.competition_name}.zip")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_file)
        print(f"Data downloaded {data_path}")

    def submit(self, submission_file, model_name, message):
        """
        Submits a competition submission file to Kaggle and retrieves the submission score.

        Args:
            submission_file (str): The path to the submission file.
            model_name (str): The name of the model used for the submission.
            message (str): The description/message for the submission.

        Returns:
            float: The submission score, rounded to 4 decimal places, or None if the score retrieval fails.
        """
        self.api.competition_submit(submission_file, message, self.competition_name)
        print(
            f"Submission {submission_file} for {self.competition_name} using {model_name}: '{message}'"
        )
        polling_interval = 5
        max_wait_time = 60
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            submissions = self.api.competitions_submissions_list(self.competition_name)
            for sub in submissions:
                if sub["description"] == message:
                    public_score = sub.get("publicScore", "")
                    if public_score:
                        print(f"Submission score: {public_score}")
                        return round(np.float32(public_score), 4)
            time.sleep(polling_interval)
        print("Failed to retrieve submission score within the maximum wait time.")
        return None
