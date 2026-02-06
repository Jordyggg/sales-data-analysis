"""
Sales Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SalesDataPreprocessor:
    """
    Preprocessing pipeline for sales data
    """
    
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        
    def load_data(self, filepath, file_type='csv'):
        """
        Load sales data from various file formats
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        file_type : str
            Type of file ('csv', 'excel', 'json')
        
        Returns:
        --------
        pd.DataFrame
        """
        try:
            if file_type == 'csv':
                self.data = pd.read_csv(filepath)
            elif file_type == 'excel':
                self.data = pd.read_excel(filepath)
            elif file_type == 'json':
                self.data = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            print(f"✅ Data loaded successfully: {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def check_data_quality(self):
        """
        Perform data quality checks
        """
        if self.data is None:
            print("❌ No data loaded")
            return
        
        print("\n" + "="*60)
        print("📊 DATA QUALITY REPORT")
        print("="*60)
        
        # Basic info
        print(f"\n📏 Shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        
        # Missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        if missing.sum() > 0:
            print("\n⚠️ Missing Values:")
            for col, count in missing[missing > 0].items():
                print(f"  • {col}: {count} ({missing_pct[col]:.2f}%)")
        else:
            print("\n✅ No missing values")
        
        # Duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\n🔄 Duplicate rows: {duplicates}")
        
        # Data types
        print("\n📝 Data Types:")
        for col, dtype in self.data.dtypes.items():
            print(f"  • {col}: {dtype}")
        
        # Basic statistics
        print("\n📊 Numeric Columns Summary:")
        print(self.data.describe())
    
    def clean_missing_values(self, strategy='auto'):
        """
        Handle missing values
        
        Parameters:
        -----------
        strategy : str
            'auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        """
        if self.data is None:
            print("❌ No data loaded")
            return
        
        self.cleaned_data = self.data.copy()
        
        if strategy == 'drop':
            self.cleaned_data = self.cleaned_data.dropna()
            print(f"✅ Dropped rows with missing values")
            
        elif strategy == 'auto':
            # Numeric columns: fill with median
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    self.cleaned_data[col].fillna(self.cleaned_data[col].median(), inplace=True)
            
            # Categorical columns: fill with mode
            cat_cols = self.cleaned_data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if self.cleaned_data[col].isnull().sum() > 0:
                    self.cleaned_data[col].fillna(self.cleaned_data[col].mode()[0], inplace=True)
            
            print(f"✅ Missing values handled automatically")
        
        print(f"📏 New shape: {self.cleaned_data.shape}")
        return self.cleaned_data
    
    def remove_duplicates(self):
        """
        Remove duplicate rows
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        before = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        after = len(self.cleaned_data)
        
        removed = before - after
        print(f"✅ Removed {removed} duplicate rows")
        
        return self.cleaned_data
    
    def convert_date_columns(self, date_columns):
        """
        Convert columns to datetime format
        
        Parameters:
        -----------
        date_columns : list
            List of column names to convert
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        for col in date_columns:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = pd.to_datetime(self.cleaned_data[col], errors='coerce')
                print(f"✅ Converted {col} to datetime")
        
        return self.cleaned_data
    
    def create_time_features(self, date_column='Date'):
        """
        Extract time-based features from date column
        
        Parameters:
        -----------
        date_column : str
            Name of the date column
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        if date_column not in self.cleaned_data.columns:
            print(f"❌ Column {date_column} not found")
            return
        
        # Ensure datetime
        self.cleaned_data[date_column] = pd.to_datetime(self.cleaned_data[date_column])
        
        # Extract features
        self.cleaned_data['Year'] = self.cleaned_data[date_column].dt.year
        self.cleaned_data['Month'] = self.cleaned_data[date_column].dt.month
        self.cleaned_data['Month_Name'] = self.cleaned_data[date_column].dt.strftime('%B')
        self.cleaned_data['Quarter'] = self.cleaned_data[date_column].dt.quarter
        self.cleaned_data['Day'] = self.cleaned_data[date_column].dt.day
        self.cleaned_data['Day_of_Week'] = self.cleaned_data[date_column].dt.day_name()
        self.cleaned_data['Week_of_Year'] = self.cleaned_data[date_column].dt.isocalendar().week
        self.cleaned_data['Is_Weekend'] = self.cleaned_data[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        print(f"✅ Created time features from {date_column}")
        return self.cleaned_data
    
    def handle_outliers(self, columns, method='iqr', threshold=1.5):
        """
        Detect and handle outliers
        
        Parameters:
        -----------
        columns : list
            List of numeric columns to check
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        outliers_removed = 0
        
        for col in columns:
            if col not in self.cleaned_data.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.cleaned_data[col].quantile(0.25)
                Q3 = self.cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing
                before = len(self.cleaned_data)
                self.cleaned_data[col] = self.cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
                
                print(f"✅ {col}: Capped outliers at [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.cleaned_data[col]))
                self.cleaned_data = self.cleaned_data[z_scores < threshold]
        
        return self.cleaned_data
    
    def encode_categorical(self, columns, method='label'):
        """
        Encode categorical variables
        
        Parameters:
        -----------
        columns : list
            List of categorical columns
        method : str
            'label' or 'onehot'
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        from sklearn.preprocessing import LabelEncoder
        
        if method == 'label':
            le = LabelEncoder()
            for col in columns:
                if col in self.cleaned_data.columns:
                    self.cleaned_data[f'{col}_Encoded'] = le.fit_transform(self.cleaned_data[col])
                    print(f"✅ Label encoded: {col}")
        
        elif method == 'onehot':
            self.cleaned_data = pd.get_dummies(self.cleaned_data, columns=columns, prefix=columns)
            print(f"✅ One-hot encoded: {columns}")
        
        return self.cleaned_data
    
    def create_revenue_features(self, units_col='Units_Sold', price_col='Unit_Price'):
        """
        Create revenue-related features
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        # Total revenue
        if units_col in self.cleaned_data.columns and price_col in self.cleaned_data.columns:
            self.cleaned_data['Revenue'] = self.cleaned_data[units_col] * self.cleaned_data[price_col]
            print("✅ Created Revenue feature")
        
        # Revenue bins
        if 'Revenue' in self.cleaned_data.columns:
            self.cleaned_data['Revenue_Category'] = pd.cut(
                self.cleaned_data['Revenue'],
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            print("✅ Created Revenue_Category feature")
        
        return self.cleaned_data
    
    def normalize_numeric(self, columns, method='standardize'):
        """
        Normalize numeric columns
        
        Parameters:
        -----------
        columns : list
            Columns to normalize
        method : str
            'standardize' or 'minmax'
        """
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if method == 'standardize':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        for col in columns:
            if col in self.cleaned_data.columns:
                self.cleaned_data[f'{col}_Normalized'] = scaler.fit_transform(
                    self.cleaned_data[[col]]
                )
                print(f"✅ Normalized: {col}")
        
        return self.cleaned_data
    
    def save_cleaned_data(self, filepath, file_type='csv'):
        """
        Save cleaned data to file
        """
        if self.cleaned_data is None:
            print("❌ No cleaned data to save")
            return
        
        try:
            if file_type == 'csv':
                self.cleaned_data.to_csv(filepath, index=False)
            elif file_type == 'excel':
                self.cleaned_data.to_excel(filepath, index=False)
            elif file_type == 'json':
                self.cleaned_data.to_json(filepath, orient='records')
            
            print(f"✅ Cleaned data saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def get_preprocessing_summary(self):
        """
        Get summary of preprocessing steps
        """
        if self.data is None:
            print("❌ No data loaded")
            return
        
        print("\n" + "="*60)
        print("📋 PREPROCESSING SUMMARY")
        print("="*60)
        
        print(f"\n📊 Original data: {self.data.shape}")
        if self.cleaned_data is not None:
            print(f"📊 Cleaned data: {self.cleaned_data.shape}")
            print(f"📉 Rows removed: {len(self.data) - len(self.cleaned_data)}")
            print(f"📈 Columns added: {len(self.cleaned_data.columns) - len(self.data.columns)}")
        
        print("\n✅ Preprocessing pipeline completed")


def quick_preprocess(filepath, save_path=None):
    """
    Quick preprocessing pipeline with default settings
    
    Parameters:
    -----------
    filepath : str
        Path to input data
    save_path : str, optional
        Path to save cleaned data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    preprocessor = SalesDataPreprocessor()
    
    # Load data
    preprocessor.load_data(filepath)
    
    # Quality check
    preprocessor.check_data_quality()
    
    # Clean
    preprocessor.clean_missing_values(strategy='auto')
    preprocessor.remove_duplicates()
    
    # If date column exists, create time features
    if 'Date' in preprocessor.cleaned_data.columns:
        preprocessor.convert_date_columns(['Date'])
        preprocessor.create_time_features('Date')
    
    # Create revenue features if applicable
    if 'Units_Sold' in preprocessor.cleaned_data.columns:
        preprocessor.create_revenue_features()
    
    # Summary
    preprocessor.get_preprocessing_summary()
    
    # Save if path provided
    if save_path:
        preprocessor.save_cleaned_data(save_path)
    
    return preprocessor.cleaned_data


# Example usage
if __name__ == "__main__":
    print("Sales Data Preprocessing Module")
    print("="*60)
    print("\nExample usage:")
    print("  from data_preprocessing import SalesDataPreprocessor")
    print("  preprocessor = SalesDataPreprocessor()")
    print("  preprocessor.load_data('sales_data.csv')")
    print("  preprocessor.clean_missing_values()")
    print("  preprocessor.create_time_features()")
    print("\nOr use quick preprocessing:")
    print("  from data_preprocessing import quick_preprocess")
    print("  clean_data = quick_preprocess('sales_data.csv', 'clean_data.csv')")
