"""
Generate a labelled dataset by integrating human-generated labels on AWS with
local unlabelled dataset.
"""

import json
import pathlib
import sys
from typing import Any

import boto3
import pandas as pd
import typer
import yaml
from loguru import logger
from pydantic import BaseModel, computed_field, model_validator


class IntegrateHumanLabelsConfig(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_bucket_name: str
    aws_img_path: str
    aws_labels_path: str

    dataset_id: str
    local_base_path: pathlib.Path

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        if "local_base_path" not in data:
            data["local_base_path"] = pathlib.Path.cwd() / "runs/dataset"
        return data

    @computed_field
    @property
    def run_path(self) -> pathlib.Path:
        return (self.local_base_path / self.dataset_id).resolve()

    @computed_field
    @property
    def local_labels_path(self) -> pathlib.Path:
        return (self.run_path / "labels.json").resolve()

    @computed_field
    @property
    def unlabelled_dataset_path(self) -> pathlib.Path:
        return (self.run_path / "unlabelled_data.csv").resolve()

    @computed_field
    @property
    def labelled_dataset_path(self) -> pathlib.Path:
        return (self.run_path / "dataset.csv").resolve()


def integrate_human_labels(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = IntegrateHumanLabelsConfig(**config_dict)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )

    s3.download_file(
        config.aws_bucket_name, config.aws_labels_path, config.local_labels_path
    )

    labels_dict = {}

    with open(config.local_labels_path, "r") as labels_file:
        for line in labels_file:
            label_json = json.loads(line)
            img_filename = label_json["source-ref"].replace(
                f"s3://{config.aws_bucket_name}/{config.aws_img_path}/", ""
            )
            labels_dict[img_filename] = label_json["Target"]

    df = pd.read_csv(config.unlabelled_dataset_path)

    df["human_label"] = [
        labels_dict[image_file_name] for image_file_name in df["image_file_name"]
    ]

    df.to_csv(config.labelled_dataset_path, index=False)


if __name__ == "__main__":
    typer.run(integrate_human_labels)
