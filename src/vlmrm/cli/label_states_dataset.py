"""
Automatically label an unlabelled dataset using a trained ML model that
takes (Gymnasium) RL states as input.
"""

import sys

import typer
from loguru import logger
from pydantic import BaseModel


class LabelStatesDatasetConfig(BaseModel):
    pass


def label_states_dataset(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    # config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    # config = LabelStatesDatasetConfig(**config_dict)

    # # Step 1: Load the unlabelled dataset
    # unlabelled_dataset_path = "unlabelled_dataset.csv"  # TODO Replace path
    # unlabelled_data = pd.read_csv(unlabelled_dataset_path)

    # # Step 2: Load the pre-trained Random Forest model
    # model_path = "random_forest_model.pkl"  # Replace path to pre-trained model file
    # clf = joblib.load(model_path)

    # # Step 3: Use the model to predict labels for the unlabelled dataset
    # predicted_labels = clf.predict(unlabelled_data)

    # # Step 4: Add the predicted labels as a new column to the dataset
    # unlabelled_data['predicted_label'] = predicted_labels

    # # Step 5: Save the labelled dataset to a new CSV file
    # labelled_dataset_path = "labelled_dataset.csv"  # Replace path
    # unlabelled_data.to_csv(labelled_dataset_path, index=False)

    # print("Labelled dataset saved to", labelled_dataset_path)
    # Replace "unlabelled_dataset.csv" with path unlabelled dataset CSV file
    # and "random_forest_model.pkl" with the path to your pre-trained Random Forest
    # model file. This script will add a new column called 'predicted_label' to the
    # dataset with the predicted labels and save the labelled dataset to a new CSV
    # file specified by "labelled_dataset.csv".


if __name__ == "__main__":
    typer.run(label_states_dataset)
