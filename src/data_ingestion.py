import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import yaml
# Ensure logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('Data_Ingestion')
logger.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """load Parameters from the yaml file"""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrived from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML erroe %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected erroe :%s',e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing the data by handling missing values"""
    try:
       
        df.rename(columns={'v1': 'label', 'v2': 'tweet'}, inplace=True)
        logger.debug('Data preprocessing completed successfully')
        return df
    except KeyError as e:
        logger.error('Missing expected columns in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while preprocessing the data: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the preprocessed data to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug('Data saved successfully to %s', raw_data_path)
    except IOError as e:
        logger.error('Failed to save data to CSV files: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_url = 'https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv'
        data_path = 'data'
        df = load_data(data_url)
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path)
    except Exception as e:
        logger.error('Unexpected error occurred in main function: %s', e)
        raise


if __name__ == "__main__":
    main()
