"""
1. Label Encoding: Represent them as numbers. Works most of the times. 
2. Binary Encoding: Represend them in binary format (000, 001, 010, etc). 
3. One Hot Encoding: Represent them as a vector 
    (example assumes 4 distinct values in the column) [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]... 
    each representing if kth value is 1 or 0
    essentially creates k different columns (k = number of distinct categories in the column)

"""

from sklearn import preprocessing
import pandas as pd
import category_encoders as ce
import numpy as np

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False, cardinality_threshold=5):
        """
        df: pandas dataframe
        categorical features: lift of column names, e.g. ["ord_1", "nom_0"......] 
        encoding_type: label, binary, one hot
        handle_na: True/False
        """
        self.df = df
        self.out_df = self.df.copy(deep=True)
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.cardinality_threshold = cardinality_threshold
        self.label_encoders = dict()
        self.binary_encoders = None # Store Binary Encoders if used
        self.one_hot_encoders = None  # Store One-Hot Encoder if used
        self.binary_cols = []
        self.one_hot_cols = []


        if self.handle_na:
            for c in self.cat_feats:
                if pd.api.types.is_numeric_dtype(self.df[c]):  
                    # If column is numeric, keep it numeric and replace NaN with a sentinel value
                    self.df.loc[:, c] = self.df[c].fillna(-9999999).astype(int)
                else:
                    # If column is non-numeric (string, object, category), convert to string and fill NaN
                    self.df.loc[:, c] = self.df[c].astype(str).fillna("-9999999")


    def _label_encoding(self):
        """Applies label encoding (Integer Encoding)"""
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.out_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.out_df
    
    def _binary_encoding(self, cols):
        """Applies Binary Encoding (Efficient alternative to one hot in cases of high cardinality)"""
        encoder = ce.BinaryEncoder(cols=cols, return_df=True)
        encoded_df = encoder.fit_transform(self.df[cols])
        self.binary_encoders = encoder
        self.binary_cols = cols
        # Drop original columns and concatenate encoded ones
        self.out_df = self.df.drop(columns=cols)
        self.out_df = pd.concat([self.out_df, encoded_df], axis=1)
        return self.out_df
    
    def _one_hot_encoding(self, cols):
        """Applies One-Hot Encoding"""
        encoder = preprocessing.OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
        encoded_array = encoder.fit_transform(self.df[cols])
        new_oh_cols = encoder.get_feature_names_out(cols) 

        encoded_df = pd.DataFrame(encoded_array, columns=new_oh_cols, index=self.df.index)
        encoded_df = encoded_df.astype(int)
        encoded_df = encoded_df.reset_index(drop=True)

        self.df = self.df.drop(columns=cols).reset_index(drop=True)
        self.df = pd.concat([self.df, encoded_df], axis=1)
        # Store encoder and column names for later transformations
        self.one_hot_encoders = encoder
        self.one_hot_cols = cols
        return self.df
    
    def _auto_encoding(self):
        """Automatically applies One-Hot Encoding or Binary Encoding based on cardinality"""
        low_cardinality_cols = [c for c in self.cat_feats if self.df[c].nunique() <= self.cardinality_threshold]
        high_cardinality_cols = [c for c in self.cat_feats if self.df[c].nunique() > self.cardinality_threshold]
        print(f"One-Hot Encoding: {low_cardinality_cols}")
        print(f"Binary Encoding: {high_cardinality_cols}")
        if low_cardinality_cols:
            self.df = self._one_hot_encoding(low_cardinality_cols)
        if high_cardinality_cols:
            self.df = self._binary_encoding(high_cardinality_cols)
        return self.df

    def fit_transform(self):
        # print(f"Columns before fit_transform: {list(self.df.columns)}")
        self.original_cols = list(self.df.columns)  # Store original columns
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._binary_encoding(self.cat_feats)
        elif self.enc_type == "onehot":
            return self._one_hot_encoding(self.cat_feats)
        elif self.enc_type == "auto":
            result = self._auto_encoding()
            # print(f"Columns after fit_transform: {list(result.columns)}")
            return result
        else:
            raise Exception("Encoding type not found")
            
    def transform(self, dataframe):
        # print(f"Test data columns: {list(dataframe.columns)}")
        
        # Check for missing categorical columns
        missing_cat_cols = [c for c in self.cat_feats if c not in dataframe.columns]
        if missing_cat_cols:
            print(f"Warning: Missing categorical columns in test data: {missing_cat_cols}")
            for c in missing_cat_cols:
                dataframe[c] = "-9999999"

        # Handle NaN values if specified
        if self.handle_na:
            for c in self.cat_feats:
                if pd.api.types.is_numeric_dtype(dataframe[c]):
                    dataframe.loc[:, c] = dataframe[c].fillna(-9999999).astype(int)
                else:
                    dataframe.loc[:, c] = dataframe[c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = dataframe[c].map(lambda x: x if x in lbl.classes_ else "-9999999")
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            encoded_df = self.binary_encoders.transform(dataframe[self.binary_cols])
            dataframe = dataframe.drop(columns=self.binary_cols)
            dataframe = pd.concat([dataframe, encoded_df], axis=1)
            return dataframe

        elif self.enc_type == "onehot":
            encoded_array = self.one_hot_encoders.transform(dataframe[self.one_hot_cols])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.one_hot_encoders.get_feature_names_out(self.one_hot_cols),
                index=dataframe.index
            ).astype(int)
            dataframe = dataframe.drop(columns=self.one_hot_cols)
            dataframe = pd.concat([dataframe, encoded_df], axis=1)
            return dataframe

        elif self.enc_type == "auto":
            # Apply binary encoding (if applicable)
            if self.binary_cols is not None and len(self.binary_cols) > 0:
                encoded_df = self.binary_encoders.transform(dataframe[self.binary_cols])
                dataframe = dataframe.drop(columns=self.binary_cols)
                dataframe = pd.concat([dataframe, encoded_df], axis=1)

            # Apply one-hot encoding (if applicable)
            if self.one_hot_cols is not None and len(self.one_hot_cols) > 0:
                encoded_array = self.one_hot_encoders.transform(dataframe[self.one_hot_cols])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.one_hot_encoders.get_feature_names_out(self.one_hot_cols),
                    index=dataframe.index
                ).astype(int)
                dataframe = dataframe.drop(columns=self.one_hot_cols)
                dataframe = pd.concat([dataframe, encoded_df], axis=1)
            return dataframe

        else:
            raise Exception("Encoding type not found")


# if __name__=="__main__":
#     import pandas as pd
#     from sklearn import linear_model
#     df_train = pd.read_csv("../input/train.csv")
#     df_test = pd.read_csv("../input/test.csv")
#     df_test['target'] = -1

#     print(df_train.shape)
#     print(df_test.shape)

#     train_idx = df_train["id"].reset_index(drop=True).values
#     test_idx = df_test["id"].reset_index(drop=True).values

#     full_data = pd.concat([df_train, df_test])

#     cols = [c for c in full_data.columns if c not in ["id", "target"]]
#     cat_feats = CategoricalFeatures(full_data, 
#                                     categorical_features=cols, 
#                                     encoding_type="auto", 
#                                     handle_na=True)
#     full_data_transformed = cat_feats.fit_transform()
#     print(full_data_transformed.head())

#     train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)]
#     test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)]

#     print(train_df.shape)
#     print(test_df.shape)

#     clf = linear_model.LogisticRegression()
#     clf.fit(train_df, train_df.target.value)
#     preds = clf.predict_proba(test_df)[:,1]