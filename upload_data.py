import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

# ---- Config (pulled from environment, not hardcoded) ----
bucket = os.getenv("S3_BUCKET", "housing-regression-data")
region = os.getenv("AWS_REGION", "auto")
endpoint_url = os.getenv("S3_ENDPOINT_URL")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# This script lives in the project root, so PROJECT_ROOT is just here
PROJECT_ROOT = Path(__file__).resolve().parent
local_data_dir = PROJECT_ROOT / "data" / "processed"
local_model_dir = PROJECT_ROOT / "models"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
)

# ---- Helper function ----
def upload_file(local_path: Path, s3_key: str):
    if not local_path.exists():
        print(f"File not found: {local_path}")
        return
    print(f"Uploading {local_path} -> s3://{bucket}/{s3_key}")
    s3.upload_file(str(local_path), bucket, s3_key)

# ---- Upload required datasets ----
upload_file(local_data_dir / "feature_engineered_holdout.csv", "processed/feature_engineered_holdout.csv")
upload_file(local_data_dir / "cleaning_holdout.csv", "processed/cleaning_holdout.csv")
upload_file(local_data_dir / "feature_engineered_train.csv", "processed/feature_engineered_train.csv")

# ---- Upload model ----
upload_file(local_model_dir / "xgb_best_model.pkl", "models/xgb_best_model.pkl")

print("Done.")