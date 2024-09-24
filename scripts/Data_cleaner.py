import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

class data_cleaner:
    def drop_duplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """
        data.drop_duplicates(inplace=True)
        return data

    def percent_missing(self, data: pd.DataFrame) -> float:
        """
        calculate the percentage of missing values from dataframe
        """
        totalCells = np.prod(data.shape)
        missingCount = data.isnull().sum()
        totalMising = missingCount.sum()

        return round(totalMising / totalCells * 100, 2)

    def get_numerical_columns(self, data: pd.DataFrame) -> list:
        """
        get numerical columns
        """
        return data.select_dtypes(include=['number']).columns.to_list()

    def get_categorical_columns(self, data: pd.DataFrame) -> list:
        """
        get categorical columns
        """
        return data.select_dtypes(include=['object', 'datetime64[ns]']).columns.to_list()

    def percent_missing_column(self, data: pd.DataFrame, col: str) -> float:
        """
        calculate the percentage of missing values for the specified column
        """
        try:
            col_len = len(data[col])
        except KeyError:
            print(f"{col} not found")
            return None
        missing_count = data[col].isnull().sum()
        return round(missing_count / col_len * 100, 2)

    def fill_missing_values_categorical(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        fill missing values with specified method for categorical columns
        """
        categorical_columns = self.get_categorical_columns(data)
        if method == "ffill":
            for col in categorical_columns:
                data[col].fillna(method='ffill', inplace=True)
        elif method == "bfill":
            for col in categorical_columns:
                data[col].fillna(method='bfill', inplace=True)
        elif method == "mode":
            for col in categorical_columns:
                data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            print("Method unknown")
        return data

    def fill_missing_values_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values for specific numerical columns
        """
        numerical_columns = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']

        # For 'Open', fill NaN values with 0 (assuming store was closed)
        if 'Open' in numerical_columns:
            data['Open'].fillna(0, inplace=True)

        # For 'Sales' and 'Customers', fill NaN values with 0 when 'Open' is 0
        if 'Sales' in numerical_columns:
            data.loc[data['Open'] == 0, 'Sales'] = data.loc[data['Open'] == 0, 'Sales'].fillna(0)
        if 'Customers' in numerical_columns:
            data.loc[data['Open'] == 0, 'Customers'] = data.loc[data['Open'] == 0, 'Customers'].fillna(0)

        # Fill remaining missing values in 'Sales' and 'Customers' with median
        if 'Sales' in numerical_columns:
            data['Sales'].fillna(data['Sales'].median(), inplace=True)
        if 'Customers' in numerical_columns:
            data['Customers'].fillna(data['Customers'].median(), inplace=True)

        return data

    def fill_missing_values_specific(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values for specific categorical columns such as 'StateHoliday'
        """
        categorical_columns = ['StateHoliday']

        # Fill missing values for 'StateHoliday' with '0'
        if 'StateHoliday' in categorical_columns:
            data['StateHoliday'].fillna('0', inplace=True)

        return data

    def normalizer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        normalize numerical columns
        """
        norm = Normalizer()
        numerical_cols = self.get_numerical_columns(data)
        normalized_data = pd.DataFrame(norm.fit_transform(data[numerical_cols]), columns=numerical_cols)
        return normalized_data

    def Nan_to_zero(self, data: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        substitute NaN values with 0 for specific columns
        """
        data[cols] = data[cols].fillna(0)
        return data

    def Nan_to_none(self, data: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        replace NaN values with 'none' for specific columns
        """
        data[cols] = data[cols].fillna('none')
        return data
