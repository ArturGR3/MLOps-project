import os
from dotenv import load_dotenv, find_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from google.cloud import storage as gcs


class CloudStorageHandler:
    def __init__(self):
        load_dotenv(
            find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True)
        )

    ## AWS S3 connection
    def access_s3_bucket(self):
        try:
            # Load AWS credentials and bucket name from environment variables
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            bucket_name = os.getenv("AWS_BUCKET_NAME")

            # Validate credentials presence
            if not access_key or not secret_key or not bucket_name:
                raise ValueError(
                    "AWS credentials or bucket name not found in environment variables"
                )

            # Initialize S3 client
            s3 = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )

            # Check if the bucket exists by listing buckets
            buckets = s3.list_buckets()["Buckets"]
            bucket_names = [bucket["Name"] for bucket in buckets]

            if bucket_name not in bucket_names:
                raise ValueError(f"AWS bucket '{bucket_name}' not found")

            return bucket_name, s3

        except (NoCredentialsError, PartialCredentialsError) as e:
            print(f"AWS credentials error: {str(e)}")
        except ValueError as e:
            print(f"AWS configuration error: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

    def upload_to_s3(self, local_file, s3_file):
        try:
            bucket_name, s3 = self.access_s3_bucket()
            s3.upload_file(local_file, bucket_name, s3_file)
        except Exception as e:
            print(f"Failed to upload to S3: {str(e)}")

    def download_from_s3(self, s3_file, local_file):
        try:
            bucket_name, s3 = self.access_s3_bucket()
            s3.download_file(bucket_name, s3_file, local_file)
        except Exception as e:
            print(f"Failed to download from S3: {str(e)}")

    ## GCS connection
    def access_gcs_bucket(self):
        try:
            # Load GCP credentials and bucket name from environment variables
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            bucket_name = os.getenv("GCP_BUCKET_NAME")

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            key_path = os.path.join(project_root, f"{credentials_path}")

            # Validate credentials presence
            if not credentials_path or not bucket_name:
                raise ValueError(
                    "GCP credentials or bucket name not found in environment variables"
                )

            # Initialize GCS client
            client = gcs.Client.from_service_account_json(key_path)
            bucket = client.bucket(bucket_name)

            # Check if the bucket exists
            if not bucket.exists():
                raise ValueError(f"GCS bucket '{bucket_name}' not found")

            return bucket_name, client

        except ValueError as e:
            print(f"GCS configuration error: {str(e)}")
            raise  # Re-raise the exception to propagate it further
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise  # Re-raise the exception to propagate it further

    def upload_to_gcs(self, local_file, gcs_file):
        try:
            bucket_name, client = self.access_gcs_bucket()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_file)
            blob.upload_from_filename(local_file)
        except Exception as e:
            print(f"Failed to upload to GCS: {str(e)}")
            raise

    def download_from_gcs(self, gcs_file, local_file):
        try:
            bucket_name, client = self.access_gcs_bucket()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(gcs_file)
            blob.download_to_filename(local_file)
        except Exception as e:
            print(f"Failed to download from GCS: {str(e)}")
            raise
