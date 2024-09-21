import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from _logger import get_logger

logger = get_logger()

# def load_data(file_path):
#     logging.info(f"Loading data from {file_path}...")
#     df = pd.read_csv(file_path, parse_dates=['Date'])
#     logging.info(f"Data loaded with shape {df.shape}")
#     return df

def load_data(file_path, parse_dates=None):
    logging.info(f"Loading data from {file_path}...")
    
    # Only parse dates if 'parse_dates' is provided
    if parse_dates:
        df = pd.read_csv(file_path, parse_dates=parse_dates)
    else:
        df = pd.read_csv(file_path)
    
    logging.info(f"Data loaded with shape {df.shape}")
    return df


def check_missing_values(df, df_name):
    logging.info(f"Checking for missing values in {df_name}...")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        logging.info(f"Missing values in {df_name}: {missing_values}")
    else:
        logging.info(f"No missing values found in {df_name}.")


def handle_missing_store_data(store_df):
    logging.info("Handling missing values in Store data...")
    
    # Use a dictionary to fill multiple columns at once
    store_df.fillna({
        'CompetitionDistance': store_df['CompetitionDistance'].median(),
        'CompetitionOpenSinceMonth': store_df['CompetitionOpenSinceMonth'].mode()[0],
        'CompetitionOpenSinceYear': store_df['CompetitionOpenSinceYear'].mode()[0],
        'Promo2SinceWeek': 0,
        'Promo2SinceYear': 0,
        'PromoInterval': 'None'
    }, inplace=True)
    
    logging.info("Missing values handled in Store data.")
    return store_df


def detect_outliers_iqr(df, column):
    logging.info(f"Detecting outliers in {column} using IQR...")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    
    if len(outliers) > 0:
        logging.info(f"Outliers detected in {column}: {outliers.shape[0]} rows")
    else:
        logging.info(f"No outliers detected in {column}")
    return outliers

# Outlier detection using Z-score
def detect_outliers_zscore(df, column, threshold=3):
    logging.info(f"Detecting outliers in {column} using Z-score...")
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    
    if len(outliers) > 0:
        logging.info(f"Outliers detected in {column}: {outliers.shape[0]} rows")
    else:
        logging.info(f"No outliers detected in {column}")
    return outliers


# Feature Engineering: Extract month, day, year, and day of the week
# def add_date_features(df):
#     logging.info("Adding date-related features...")
#     df['Year'] = df.index.year
#     df['Month'] = df.index.month
#     df['Day'] = df.index.day
#     df['DayOfWeek'] = df.index.dayofweek
#     logging.info("Date features added.")
#     return df


def add_date_features(df):
    logger.info("Adding date-related features...")

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("The DataFrame index is not a DatetimeIndex. Make sure the 'Date' column is set as the index.")
        return df

    # Add new features based on the 'Date' index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['WeekOfYear'] = df.index.isocalendar().week
    df['DayOfWeek'] = df.index.dayofweek
    
    logger.info("Date-related features added successfully.")
    return df

# Feature Engineering: Add a flag for weekends
def add_weekend_flag(df):
    logging.info("Adding weekend flag...")
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    logging.info("Weekend flag added.")
    return df





def clean_store_data(df):
    logger.info("Cleaning store data...")

    # Optionally log the initial state of missing values
    logger.info(f"Initial NA values in CompetitionDistance: {df['CompetitionDistance'].isna().sum()}")

    # Fill missing 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with 0 (assumes no competition before)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    
    # Fill missing 'Promo2SinceWeek' and 'Promo2SinceYear' with 0 (indicating no active promo)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)

    # Handle 'PromoInterval': Convert to binary indicator whether the store has promos throughout the year
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    df['HasPromo'] = df['PromoInterval'].apply(lambda x: 0 if x == 'None' else 1)

    logger.info("Store data cleaned successfully.")
    return df

