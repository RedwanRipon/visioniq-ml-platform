import os
import shutil
import sys

from src.utils.config import PROJECT_ROOT

LOCAL_ROOT = PROJECT_ROOT / "data" / "storage"


def _s3():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),   # MinIO address, or None for AWS
                        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))


def upload_file(local_path, key):
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        _s3().upload_file(str(local_path), bucket, key)
        return f"s3://{bucket}/{key}"
    dest = LOCAL_ROOT / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest)
    return str(dest)


def download_file(key, local_path):
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        _s3().download_file(bucket, key, str(local_path))
    else:
        shutil.copy2(LOCAL_ROOT / key, local_path)
    return local_path


def list_objects(prefix=""):
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        resp = _s3().list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    return [p.relative_to(LOCAL_ROOT).as_posix() for p in LOCAL_ROOT.rglob("*") if p.is_file()]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "upload":
        print(upload_file(sys.argv[2], sys.argv[3]))
    elif cmd == "list":
        print("\n".join(list_objects()))