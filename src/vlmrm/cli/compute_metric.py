"""
Compute an aggregated metric for a set of runs given a vector of labels for each run.
"""

import copy
import pathlib
import sys

import pandas as pd
import typer
import yaml
from loguru import logger
from pydantic import BaseModel, computed_field


class ComputeMetricConfig(BaseModel):
    targets_csv_name: str
    dataset_paths: list[str]
    run_names: list[str]
    labels: list[str]
    run_column_name: str

    @computed_field
    @property
    def dataset_base_path(self) -> pathlib.Path:
        return pathlib.Path("./runs/dataset").resolve()


def integrate_human_labels(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = ComputeMetricConfig(**config_dict)

    float_labels = [float(label) for label in config.labels]
    print(float_labels)

    zero_dict = {label: 0.0 for label in config.labels}

    columns = [config.run_column_name, "Accuracy"] + config.labels
    results = pd.DataFrame(columns=columns)

    for dataset_path, run_name in zip(config.dataset_paths, config.run_names):
        targets_path = (
            config.dataset_base_path / dataset_path / config.targets_csv_name
        ).resolve()
        df = pd.read_csv(targets_path, header=None)
        n_samples = len(df)
        value_counts = df[0].value_counts()
        label_dist_dict = {
            f"{index:.2f}": value / n_samples  # value
            for index, value in zip(value_counts.index, value_counts.values)
        }
        row = copy.deepcopy(zero_dict)
        row.update(label_dist_dict)
        row[config.run_column_name] = run_name
        row["Accuracy"] = df[0].mean()
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    print(results)


if __name__ == "__main__":
    typer.run(integrate_human_labels)
