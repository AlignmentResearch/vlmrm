"""Sample images from a given folder."""

import math
import os
import pathlib
import random
import sys
from typing import Any

import typer
import yaml
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field, model_validator

from vlmrm.util.util import get_run_hash


class SampleImagesConfig(BaseModel):
    dataset_id: str
    base_path: pathlib.Path
    sample_size: int
    grid_width: int
    seed: int

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        if "base_path" not in data:
            data["base_path"] = pathlib.Path.cwd() / "runs/dataset"
        return data

    @computed_field
    @property
    def dataset_path(self) -> pathlib.Path:
        return (self.base_path / self.dataset_id / "img").resolve()


def sample_images(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = SampleImagesConfig(**config_dict)

    random.seed(config.seed)

    with os.scandir(config.dataset_path) as entries:
        images = [entry.name for entry in entries if entry.is_file()]

    selected_images = random.sample(images, config.sample_size)

    images = [
        Image.open((config.dataset_path / image_filename).resolve())
        for image_filename in selected_images
    ]

    img_width = images[0].width
    img_height = images[0].height

    total_width = img_width * config.grid_width
    total_height = img_height * math.ceil(len(images) / config.grid_width)

    composite_image = Image.new("RGB", (total_width, total_height))

    for img_idx, img in enumerate(images):
        composite_image.paste(
            img,
            (
                img_width * (img_idx % config.grid_width),
                img_height * (img_idx // config.grid_width),
            ),
        )

    composite_image.show()

    composite_image.save(
        f"{str(config.dataset_path)[:-4]}/img_sample_{get_run_hash()}.png"
    )

    for image in images:
        image.close()


if __name__ == "__main__":
    typer.run(sample_images)
