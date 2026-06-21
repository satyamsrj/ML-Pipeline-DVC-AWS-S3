import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import joblib

# Ensure required NLTK resources are available
for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Data_Preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # DEBUG for console
file_handler = logging.FileHandler(os.path.join(log_dir, "data_preprocessing.log"))
file_handler.setLevel(logging.DEBUG)     # DEBUG for file

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize reusable objects
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def transform_tweet(tweet: str) -> str:
    """Transform tweet: lowercase, tokenize, remove punctuation/stopwords, stem words."""
    tokens = nltk.word_tokenize(tweet.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

def preprocess_df(df: pd.DataFrame, tweet_column="tweet", target_column="label") -> pd.DataFrame:
    """Preprocess DataFrame: transform tweets, encode labels, remove duplicates."""
    try:
        df[tweet_column] = df[tweet_column].apply(transform_tweet)
        logger.debug("Tweet transformation completed")

        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.debug("Label encoding completed")

        df.drop_duplicates(inplace=True)
        logger.debug("Duplicates removed")

        return df, le
    except KeyError as e:
        logger.error("Missing expected columns: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def main(tweet_column="tweet", target_column="label"):
    """Load, preprocess, and save train/test datasets."""
    try:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logger.debug("Data loaded successfully")

        train_data, train_le = preprocess_df(train_data, tweet_column, target_column)
        test_data, test_le = preprocess_df(test_data, tweet_column, target_column)

        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        joblib.dump(train_le, os.path.join(data_path, "label_encoder.pkl"))

        logger.debug("Preprocessed data and encoder saved to %s", data_path)
    except FileNotFoundError as e:
        logger.error("CSV file not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("CSV file is empty: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error in main: %s", e)
        raise

if __name__ == "__main__":
    main()
