"""Train a ML model to classify a (Gymnasium) RL state"""

import pathlib
import sys

import typer
from loguru import logger
from pydantic import BaseModel

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import cross_val_score, train_test_split


class TrainDatasetLabellerConfig(BaseModel):
    dataset_id: str
    base_path: pathlib.Path
    seed: int


def train_datset_labeller(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    # config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    # config = TrainDatasetLabellerConfig(**config_dict)

    # # Step 1: Read the CSV dataset
    # dataset_path = "your_dataset.csv"  # Replace with the path to your CSV file
    # data = pd.read_csv(dataset_path)

    # # Assuming your target column is named "target" and features are all other columns
    # X = data.drop(columns=["target"])
    # y = data["target"]

    # # Step 2: Randomly split the dataset into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # # Step 3: Train and evaluate a random forest classifier using 10-fold
    # # cross-validation
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # # Perform 10-fold cross-validation
    # cv_scores = cross_val_score(clf, X_train, y_train, cv=10)

    # # Step 4: Print the cross-validation results
    # print("Cross-validation scores:", cv_scores)
    # print("Mean Accuracy:", cv_scores.mean())

    # # Fit the model on the entire training set
    # clf.fit(X_train, y_train)

    # # Step 5: Evaluate the model on the test set
    # y_pred = clf.predict(X_test)

    # print("\nTest Set Evaluation:")
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    # # Step 6: Save the trained model to a file
    # model_filename = "random_forest_model.pkl"
    # joblib.dump(clf, model_filename)
    # print("\nModel saved to", model_filename)


if __name__ == "__main__":
    typer.run(train_datset_labeller)
