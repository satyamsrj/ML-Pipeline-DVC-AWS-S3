import os
import numpy as np
import pandas as pd
import pickle
import logging
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('Model_Evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully from %s", model_path)
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load test data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded successfully from %s", data_path)
        return df
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_path, e)
        raise

def evaluate_model(model, x_test: np.array, y_test: np.array) -> dict:
    """Evaluate the performance of a trained model."""
    try:
        y_pred = model.predict(x_test)

        # Handle models that may not support predict_proba
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc
        }

        logger.debug("Model evaluation metrics: %s", metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error("Error occurred while evaluating model: %s", e)
        raise

def save_metrics(metrics_dict: dict, metrics_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as file:
            json.dump(metrics_dict, file, indent=5)
        logger.debug('Metrics saved to %s', metrics_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        model = load_model('./models/random_forest_model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics_dict = evaluate_model(model, x_test, y_test)
        save_metrics(metrics_dict, 'report/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation %s', e)
        print(f"error: {e}")

if __name__ == '__main__':
    main()
