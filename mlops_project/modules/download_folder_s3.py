import boto3
from botocore.exceptions import ClientError
import os


# list the files in the s3 folder
def list_files_in_s3_folder(bucket_name, s3_folder):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get("Contents", []):
            print(obj["Key"])


def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client("s3")
    # Print the lists all objects within the specified folder

    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    try:
        # List objects within the S3 folder
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
            for obj in page.get("Contents", []):
                print(f"Downloading {obj['Key']}")
                # Get the relative path of the file
                relative_path = os.path.relpath(obj["Key"], s3_folder)
                # Construct the full local path
                local_file_path = os.path.join(local_dir, relative_path)
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                # Download the file
                s3.download_file(bucket_name, obj["Key"], local_file_path)
                print(f"Downloaded {obj['Key']} to {local_file_path}")

        print(
            f"Successfully downloaded folder '{s3_folder}' from bucket '{bucket_name}' to '{local_dir}'"
        )
    except ClientError as e:
        print(f"Error downloading folder: {e}")
