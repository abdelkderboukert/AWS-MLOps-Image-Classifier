#!/bin/bash

# Default values
REGION="us-east-1"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u)
      AWS_ACCOUNT_ID="$2"
      shift 2
      ;;
    -r)
      REGION="$2"
      shift 2
      ;;
    -n)
      REPO_NAME="$2"
      shift 2
      ;;
    -t)
      IMAGE_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate inputs
if [ -z "$AWS_ACCOUNT_ID" ] || [ -z "$REPO_NAME" ] || [ -z "$IMAGE_TAG" ]; then
    echo "Usage: ./push_to_aws.sh -u <account_id> -n <repo_name> -t <tag> [-r <region>]"
    exit 1
fi

# Construct the full ECR URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "--- Starting AWS ECR Push Process ---"

# 1. Login to ECR
echo "Step 1: Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# 2. Tag the local image
echo "Step 2: Tagging image as $ECR_URI/$REPO_NAME:$IMAGE_TAG"
docker tag $REPO_NAME:$IMAGE_TAG $ECR_URI/$REPO_NAME:$IMAGE_TAG
 
# 3. Push to AWS
echo "Step 3: Pushing to AWS..."
docker push $ECR_URI/$REPO_NAME:$IMAGE_TAG

echo "--- Done! Image is now in AWS ECR ---"