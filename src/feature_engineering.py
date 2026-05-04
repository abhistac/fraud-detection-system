import numpy as np
import pandas as pd
import logging


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self._amount_percentile_bins = None
        self._amount_high_threshold = None
        self._fitted_columns = None

    def fit(self, df):
        """Learn statistics from training data to prevent data leakage."""
        if self.config['features']['amount_features']:
            self._amount_percentile_bins = np.percentile(df['Amount'], [25, 50, 75, 90, 95, 99])
            self._amount_high_threshold = df['Amount'].quantile(0.95)
        return self

    def transform(self, df):
        """Apply feature engineering using statistics learned from training data."""
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_pca_combinations(df)
        df = self.create_rolling_features(df)

        for feature in ['Amount_percentile', 'Amount_category']:
            if feature in df.columns:
                df = pd.get_dummies(df, columns=[feature], prefix=feature)

        df = df.fillna(0)

        if self._fitted_columns is not None:
            df = df.reindex(columns=self._fitted_columns, fill_value=0)
        else:
            self._fitted_columns = df.columns.tolist()

        logging.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

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

        bins = (
            self._amount_percentile_bins
            if self._amount_percentile_bins is not None
            else np.percentile(df['Amount'], [25, 50, 75, 90, 95, 99])
        )
        df['Amount_percentile'] = pd.cut(
            df['Amount'],
            bins=[-np.inf] + list(bins) + [np.inf],
            labels=range(7)
        )
        df['Amount_category'] = pd.cut(
            df['Amount'],
            bins=[0, 10, 50, 100, 500, 1000, np.inf],
            labels=['micro', 'small', 'medium', 'large', 'very_large', 'extreme']
        )

        threshold = (
            self._amount_high_threshold
            if self._amount_high_threshold is not None
            else df['Amount'].quantile(0.95)
        )
        df['High_amount'] = (df['Amount'] > threshold).astype(int)
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
