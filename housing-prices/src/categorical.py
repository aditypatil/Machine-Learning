from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import pandas as pd
import category_encoders as ce
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import scipy.stats as stats


class CategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_features=None,
        ordinal_configs=None,
        binary_cols=None,
        one_hot_cols=None,
        auto_cols=None,
        handle_na=False,
        cardinality_threshold=5
    ):
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.ordinal_configs = ordinal_configs if ordinal_configs is not None else []
        self.binary_cols = binary_cols if binary_cols is not None else []
        self.one_hot_cols = one_hot_cols if one_hot_cols is not None else []
        self.auto_cols = auto_cols if auto_cols is not None else []
        self.handle_na = handle_na
        self.cardinality_threshold = cardinality_threshold
        self.ordinal_encoders = {}
        self.binary_encoders = None
        self.one_hot_encoders = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())

        if self.handle_na:
            for c in self.categorical_features:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df.loc[:, c] = df[c].fillna(-9999999).astype(int)
                else:
                    df.loc[:, c] = df[c].astype(str).fillna("-9999999")

        # Ordinal encoding
        for cols, categories in self.ordinal_configs:
            encoder = preprocessing.OrdinalEncoder(
                categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1
            )
            self.ordinal_encoders[cols] = encoder.fit(df[[cols]])

        # Binary encoding
        if self.binary_cols:
            self.binary_encoders = ce.BinaryEncoder(cols=self.binary_cols, return_df=True)
            self.binary_encoders.fit(df[self.binary_cols])

        # One-hot encoding
        if self.one_hot_cols:
            self.one_hot_encoders = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.one_hot_encoders.fit(df[self.one_hot_cols])

        # Auto encoding: Compute locally without modifying instance attributes
        if self.auto_cols:
            low_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() <= self.cardinality_threshold]
            high_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() > self.cardinality_threshold]
            if low_cardinality_cols:
                self.one_hot_encoders = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.one_hot_encoders.fit(df[low_cardinality_cols])
            if high_cardinality_cols:
                self.binary_encoders = ce.BinaryEncoder(cols=high_cardinality_cols, return_df=True)
                self.binary_encoders.fit(df[high_cardinality_cols])

        self.feature_names_out_ = self._get_feature_names(df)
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        out_df = df.copy(deep=True)

        if self.handle_na:
            for c in self.categorical_features:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df.loc[:, c] = df[c].fillna(-9999999).astype(int)
                else:
                    df.loc[:, c] = df[c].astype(str).fillna("-9999999")

        # Ordinal encoding
        for cols, _ in self.ordinal_configs:
            if cols in df.columns:
                encoded = self.ordinal_encoders[cols].transform(df[[cols]])
                out_df[cols] = encoded

        # Binary encoding
        if self.binary_cols and self.binary_encoders:
            binary_encoded = self.binary_encoders.transform(df[self.binary_cols])
            out_df = out_df.drop(columns=self.binary_cols)
            out_df = pd.concat([out_df, binary_encoded], axis=1)

        # One-hot encoding
        if self.one_hot_cols and self.one_hot_encoders:
            one_hot_encoded = self.one_hot_encoders.transform(df[self.one_hot_cols])
            one_hot_df = pd.DataFrame(
                one_hot_encoded,
                columns=self.one_hot_encoders.get_feature_names_out(self.one_hot_cols),
                index=df.index
            )
            out_df = out_df.drop(columns=self.one_hot_cols)
            out_df = pd.concat([out_df, one_hot_df], axis=1)

        # Auto encoding
        if self.auto_cols:
            low_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() <= self.cardinality_threshold]
            high_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() > self.cardinality_threshold]
            if low_cardinality_cols and self.one_hot_encoders:
                one_hot_encoded = self.one_hot_encoders.transform(df[low_cardinality_cols])
                one_hot_df = pd.DataFrame(
                    one_hot_encoded,
                    columns=self.one_hot_encoders.get_feature_names_out(low_cardinality_cols),
                    index=df.index
                )
                out_df = out_df.drop(columns=low_cardinality_cols)
                out_df = pd.concat([out_df, one_hot_df], axis=1)
            if high_cardinality_cols and self.binary_encoders:
                binary_encoded = self.binary_encoders.transform(df[high_cardinality_cols])
                out_df = out_df.drop(columns=high_cardinality_cols)
                out_df = pd.concat([out_df, binary_encoded], axis=1)

        return out_df[self.feature_names_out_]

    def _get_feature_names(self, df):
        feature_names = [col for col in df.columns if col not in self.categorical_features]
        for col, _ in self.ordinal_configs:
            if col in df.columns:
                feature_names.append(col)
        if self.one_hot_cols and self.one_hot_encoders:
            feature_names.extend(self.one_hot_encoders.get_feature_names_out(self.one_hot_cols))
        if self.binary_cols and self.binary_encoders:
            feature_names.extend(self.binary_encoders.get_feature_names_out())
        if self.auto_cols:
            low_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() <= self.cardinality_threshold]
            high_cardinality_cols = [c for c in self.auto_cols if df[c].nunique() > self.cardinality_threshold]
            if low_cardinality_cols and self.one_hot_encoders:
                feature_names.extend(self.one_hot_encoders.get_feature_names_out(low_cardinality_cols))
            if high_cardinality_cols and self.binary_encoders:
                feature_names.extend(self.binary_encoders.get_feature_names_out())
        return feature_names

    def get_feature_names_out(self, *args, **kwargs):
        return self.feature_names_out_

# Custom Correlation Threshold Transformer
class CorrelationThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.91):
        self.threshold = threshold
        self.to_drop = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.drop(self.to_drop, axis=1, errors='ignore').values

# Custom Preprocessor


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ordinal_configs=None,
        one_hot_cols=None,
        binary_cols=None,
        auto_cols=None,
        numeric_cols=None,
        impute_dict=None,
        binning_configs=None,
        cardinality_threshold=5
    ):
        # Initialize parameters with defaults
        self.ordinal_configs = ordinal_configs if ordinal_configs is not None else []
        self.one_hot_cols = one_hot_cols if one_hot_cols is not None else []
        self.binary_cols = binary_cols if binary_cols is not None else []
        self.auto_cols = auto_cols if auto_cols is not None else []
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.impute_dict = impute_dict if impute_dict is not None else {}
        self.binning_configs = binning_configs if binning_configs is not None else {}
        self.cardinality_threshold = cardinality_threshold

        # Define categorical features
        categorical_features = (
            self.one_hot_cols + self.binary_cols + self.auto_cols + 
            [cfg[0] for cfg in self.ordinal_configs]
        )

        # Initialize categorical transformer if there are categorical features
        self.categorical_transformer = (
            CategoricalFeatures(
                categorical_features=categorical_features,
                ordinal_configs=self.ordinal_configs,
                one_hot_cols=self.one_hot_cols,
                binary_cols=self.binary_cols,
                auto_cols=self.auto_cols,
                handle_na=True,
                cardinality_threshold=self.cardinality_threshold
            ) if categorical_features else None
        )

        # Numeric transformer pipeline
        self.numeric_transformer = (
            Pipeline([
                # ('power', PowerTransformer(method='yeo-johnson', standardize=True)),
                ('scaler', StandardScaler())
            ]) if self.numeric_cols else None
        )

        # Build ColumnTransformer
        transformers = []
        if self.numeric_cols and self.numeric_transformer:
            transformers.append(('numeric', self.numeric_transformer, self.numeric_cols))
        if categorical_features and self.categorical_transformer:
            transformers.append(('categorical', self.categorical_transformer, categorical_features))

        self.column_transformer = (
            ColumnTransformer(transformers, remainder='drop') if transformers else None
        )
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X.copy())

        # Imputation
        for col, value in self.impute_dict.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(value)

        # Binning
        for col, config in self.binning_configs.items():
            if col in X_df.columns:
                X_df[f"{col}_binned"] = pd.cut(
                    X_df[col], bins=config['bins'], labels=config['labels'], right=False
                ).cat.add_categories(config.get('fill', 'missing')).fillna(config.get('fill', 'missing')).astype('object')
                X_df = X_df.drop(columns=[col])

        if self.column_transformer:
            self.column_transformer.fit(X_df)
            self.feature_names_out_ = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X.copy())

        # Imputation
        for col, value in self.impute_dict.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(value)

        # Binning
        for col, config in self.binning_configs.items():
            if col in X_df.columns:
                X_df[f"{col}_binned"] = pd.cut(
                    X_df[col], bins=config['bins'], labels=config['labels'], right=False
                ).cat.add_categories(config.get('fill', 'missing')).fillna(config.get('fill', 'missing')).astype('object')
                X_df = X_df.drop(columns=[col])

        # Transform data
        if self.column_transformer:
            X_transformed = self.column_transformer.transform(X_df)
            return pd.DataFrame(X_transformed, columns=self.feature_names_out_, index=X_df.index)
        return X_df

    def get_feature_names_out(self):
        return self.feature_names_out_