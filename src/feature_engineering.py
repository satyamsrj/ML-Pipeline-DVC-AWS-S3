import pandas as pd
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('Feature_Engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """Load preprocessed data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)  # Fill missing values with empty strings
        logger.debug("Preprocessed data loaded successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading preprocessed data from %s: %s", file_path, e)
        raise

def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int, ) -> tuple:
    """Apply TF-IDF vectorization to the text data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        x_train = train_data['tweet'].values
        y_train = train_data['label'].values
        x_test = test_data['tweet'].values
        y_test = test_data['label'].values

        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)
        logger.debug("TF-IDF vectorization applied successfully with max_features=%d", max_features)

        train_df = pd.DataFrame(x_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        test_df['label'] = y_test

        logger.debug("TF-IDF vectorization completed successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error("Error during TF-IDF vectorization: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved successfully to %s", file_path)
    except IOError as e:
        logger.error("Failed to save data to CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

def main(): 
    try:
        train_data = load_preprocessed_data("data/interim/train_processed.csv")
        test_data = load_preprocessed_data("data/interim/test_processed.csv")

        train_df, test_df = apply_tfidf_vectorization(train_data, test_data, max_features=50)

        save_data(train_df, "data/processed/train_tfidf.csv")
        save_data(test_df, "data/processed/test_tfidf.csv")
        logger.debug("Feature engineering completed successfully.")
    except Exception as e:
        logger.error("Error in main function: %s", e)
        raise

if __name__ == "__main__":
    main()