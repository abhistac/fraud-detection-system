import numpy as np
import pandas as pd
import logging


class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def create_time_features(self, df):
        if not self.config['features']['time_features']:
            return df

        df = df.copy()
        df['Time_hours'] = df['Time'] / 3600
        df['Hour_of_day'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] // (24 * 3600)).astype(int)
        df['Night_transaction'] = ((df['Hour_of_day'] >= 22) | (df['Hour_of_day'] <= 6)).astype(int)
        df['Is_weekend'] = (df['Day'] % 7 >= 5).astype(int)
        return df

    def create_amount_features(self, df):
        if not self.config['features']['amount_features']:
            return df

        df = df.copy()
        df['Amount_log'] = np.log1p(df['Amount'])

        amount_percentiles = np.percentile(df['Amount'], [25, 50, 75, 90, 95, 99])
        df['Amount_percentile'] = pd.cut(
            df['Amount'],
            bins=[-np.inf] + list(amount_percentiles) + [np.inf],
            labels=range(7)
        )
        df['Amount_category'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, 1000, np.inf],
            labels=['micro', 'small', 'medium', 'large', 'very_large', 'extreme']
        )
        df['High_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
        df['Zero_amount'] = (df['Amount'] == 0).astype(int)
        return df

    def create_pca_combinations(self, df):
        if not self.config['features']['pca_combinations']:
            return df

        df = df.copy()
        pca_cols = [col for col in df.columns if col.startswith('V')]

        df['V1_V2_interaction'] = df['V1'] * df['V2']
        df['V1_V3_interaction'] = df['V1'] * df['V3']
        df['V2_V3_interaction'] = df['V2'] * df['V3']
        df['V1_V5_sum'] = df[['V1', 'V2', 'V3', 'V4', 'V5']].sum(axis=1)
        df['V6_V10_sum'] = df[['V6', 'V7', 'V8', 'V9', 'V10']].sum(axis=1)
        df['PCA_mean'] = df[pca_cols].mean(axis=1)
        df['PCA_std'] = df[pca_cols].std(axis=1)
        df['PCA_skew'] = df[pca_cols].skew(axis=1)
        df['Negative_V_count'] = (df[pca_cols] < 0).sum(axis=1)
        return df

    def create_rolling_features(self, df):
        if not self.config['features']['rolling_statistics']:
            return df

        df = df.copy().sort_values('Time').reset_index(drop=True)

        for window in [10, 50, 100]:
            df[f'Amount_rolling_mean_{window}'] = df['Amount'].rolling(window=window, min_periods=1).mean()
            df[f'Amount_rolling_std_{window}'] = df['Amount'].rolling(window=window, min_periods=1).std()

        df['Trans_frequency_1h'] = df.groupby(df['Time'] // 3600)['Time'].transform('count')
        return df

    def engineer_features(self, df):
        logging.info("Starting feature engineering...")

        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_pca_combinations(df)
        df = self.create_rolling_features(df)

        for feature in ['Amount_percentile', 'Amount_category']:
            if feature in df.columns:
                df = pd.get_dummies(df, columns=[feature], prefix=feature)

        df = df.fillna(0)
        logging.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
