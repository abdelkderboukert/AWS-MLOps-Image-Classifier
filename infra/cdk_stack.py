from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_s3 as s3,
    RemovalPolicy
)
from constructs import Construct

class PokemonMlopsStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # ECR Repo for our Docker images
        self.repo = ecr.Repository(
            self, "PokemonRepo",
            repository_name="pokemon-classifier",
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True
        )

        # S3 Bucket for raw data and saved models
        self.bucket = s3.Bucket(
            self, "PokemonDataBucket",
            bucket_name=f"pokemon-mlops-data-{self.account}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )