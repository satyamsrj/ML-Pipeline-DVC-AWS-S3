import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger("Model_Training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading data from %s: %s", file_path, e)
        raise

def train_model(x_train: np.array, y_train: np.array, params: dict) -> RandomForestClassifier:
    """Train a Random Forest Classifier on the training data."""
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in x_train and y_train must be the same.")
        logger.debug("Initializing RandomForest with parameters %s", params)

        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=params.get("random_state", 42)
        )
        model.fit(x_train, y_train)
        logger.debug("Model training completed successfully")
        return model
    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
        raise
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model to %s: %s", file_path, e)
        raise

def main():
    """Main function to load data, train the model, and save the trained model."""
    try:
        # Load preprocessed data
        train_data = load_data("data/processed/train_tfidf.csv")
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Define hyperparameters
        params = {"n_estimators": 100, "random_state": 42}

        # Train the model
        model = train_model(x_train, y_train, params)

        # Save the trained model
        save_model(model, "models/random_forest_model.pkl")
    except Exception as e:
        logger.error("Error in main function: %s", e)
        raise

if __name__ == "__main__":
    main()
