"""
MLOps Router Lambda
────────────────────────────────────────────────────────────────────
Triggered by: S3 ObjectCreated events on the ml-models bucket

Responsibilities:
1. Parse S3 event to get the uploaded object key
2. Extract service name and model version from the path
3. Fetch GitHub PAT from Secrets Manager
4. Trigger the correct GitHub Actions workflow via workflow_dispatch
5. Pass model_version, s3_path, and service_name as workflow inputs

Expected S3 key format:
  text-service/v12/model.pt   → triggers text-service-deploy.yml
  image-service/v7/model.pt   → triggers image-service-deploy.yml
"""

import json
import os
import re
import urllib.request
import urllib.error
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables (set by Terraform)
GITHUB_OWNER = os.environ["GITHUB_OWNER"]
GITHUB_REPO  = os.environ["GITHUB_REPO"]
SECRET_ARN   = os.environ["GITHUB_SECRET_ARN"]

# Map service prefix → GitHub Actions workflow filename
WORKFLOW_MAP = {
    "text-service":  "text-service-deploy.yml",
    "image-service": "image-service-deploy.yml",
}

secrets_client = boto3.client("secretsmanager")


def get_github_token() -> str:
    """Fetch GitHub PAT from Secrets Manager at runtime (never hardcoded)."""
    response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
    secret = json.loads(response["SecretString"])
    return secret["token"]


def parse_s3_key(object_key: str) -> tuple[str, str]:
    """
    Parse the S3 object key and return (service_name, model_version).

    Example:
        "text-service/v12/model.pt" → ("text-service", "v12")
        "image-service/v7/model.pt" → ("image-service", "v7")

    Raises ValueError if the key doesn't match the expected pattern.
    """
    # Pattern: <service>/<version>/<filename>
    pattern = r"^(text-service|image-service)/(v\d+)/(.+)$"
    match = re.match(pattern, object_key)
    if not match:
        raise ValueError(
            f"Unexpected S3 key format: '{object_key}'. "
            f"Expected: <service>/<version>/model.pt"
        )
    service_name   = match.group(1)
    model_version  = match.group(2)
    return service_name, model_version


def trigger_github_workflow(
    token: str,
    workflow_file: str,
    service_name: str,
    model_version: str,
    s3_path: str,
) -> None:
    """
    Trigger a GitHub Actions workflow via workflow_dispatch API.

    API reference:
    https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event
    """
    url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/actions/workflows/{workflow_file}/dispatches"
    )

    payload = json.dumps({
        "ref": "main",                       # branch to run the workflow on
        "inputs": {
            "service_name":    service_name,
            "model_version":   model_version,
            "s3_path":         s3_path,
        }
    }).encode("utf-8")

    headers = {
        "Accept":               "application/vnd.github+json",
        "Authorization":        f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type":         "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            logger.info(
                "GitHub API response: HTTP %s for workflow '%s'",
                status, workflow_file
            )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error(
            "GitHub API error: HTTP %s — %s", e.code, body
        )
        raise


def lambda_handler(event: dict, context) -> dict:
    """
    Main Lambda entry point.

    Processes one S3 record at a time (S3 batches up to 1 record per
    invocation for ObjectCreated events, but we handle multiple to be safe).
    """
    logger.info("Received event: %s", json.dumps(event))

    processed = []
    errors     = []

    for record in event.get("Records", []):
        try:
            bucket_name = record["s3"]["bucket"]["name"]
            object_key  = record["s3"]["object"]["key"]

            logger.info("Processing upload: s3://%s/%s", bucket_name, object_key)

            # ── Parse key ─────────────────────────────────────────
            service_name, model_version = parse_s3_key(object_key)

            # ── Resolve workflow file ─────────────────────────────
            workflow_file = WORKFLOW_MAP.get(service_name)
            if not workflow_file:
                raise ValueError(
                    f"No workflow mapped for service '{service_name}'. "
                    f"Supported services: {list(WORKFLOW_MAP.keys())}"
                )

            s3_path = f"s3://{bucket_name}/{object_key}"

            logger.info(
                "Routing: service=%s version=%s → workflow=%s",
                service_name, model_version, workflow_file
            )

            # ── Fetch GitHub token ────────────────────────────────
            token = get_github_token()

            # ── Trigger workflow ──────────────────────────────────
            trigger_github_workflow(
                token         = token,
                workflow_file = workflow_file,
                service_name  = service_name,
                model_version = model_version,
                s3_path       = s3_path,
            )

            processed.append({
                "s3_path":      s3_path,
                "service":      service_name,
                "version":      model_version,
                "workflow":     workflow_file,
            })

        except Exception as exc:
            logger.error("Failed to process record: %s", exc, exc_info=True)
            errors.append({"record": record, "error": str(exc)})

    result = {
        "statusCode": 200 if not errors else 207,
        "processed":  processed,
        "errors":     errors,
    }
    logger.info("Result: %s", json.dumps(result))
    return result
