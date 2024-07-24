import os
import pytest
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
project_path = os.getenv("PROJECT_PATH")
print(f"project path {project_path}")
if project_path not in sys.path:
    sys.path.append(project_path)
from modules.cloud_storage_handler import CloudStorageHandler # Adjust the import to match the new class name

integration_data_path = "tests/integration_test/data"

@pytest.fixture(scope="module")
def handler():
    # Setup
    handler = CloudStorageHandler()
    yield handler  # provide the fixture value
    # Teardown if needed


def test_upload_download_s3(handler):
    local_file = f"{project_path}/{integration_data_path}/local_file.txt"
    s3_file = "s3_file.txt"

    # Test upload to S3
    handler.upload_to_s3(local_file, s3_file)

    # Test download from S3
    downloaded_file = f"{project_path}/{integration_data_path}/downloaded_local_file.txt"
    handler.download_from_s3(s3_file, downloaded_file)

    # Assert that the downloaded file exists and has the same content as the original
    assert os.path.exists(downloaded_file)
    with open(local_file, "r") as f1, open(downloaded_file, "r") as f2:
        assert f1.read() == f2.read()


# def test_upload_download_gcs(handler):
#     local_file = f"{project_path}/{integration_data_path}/local_file.txt"
#     gcs_file = "gcs_file.txt"

#     # Test upload to GCS
#     handler.upload_to_gcs(local_file, gcs_file)

#     # Test download from GCS
#     downloaded_file = f"{project_path}/{integration_data_path}/downloaded_local_file.txt"
#     handler.download_from_gcs(gcs_file, downloaded_file)

#     # Assert that the downloaded file exists and has the same content as the original
#     assert os.path.exists(downloaded_file)
#     with open(local_file, "r") as f1, open(downloaded_file, "r") as f2:
#         assert f1.read() == f2.read()

if __name__ == "__main__":
    
    pytest.main([__file__])