import os
import numpy as np
import pandas as pd
from . import dispatcher
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

MODEL = os.environ.get("MODEL", "Lasso")

class FeatureSelector:
    def __init__(self, 
                 problem_type='classification',
                 n_features=None,
                 selection_methods=['all'],
                 model=None,
                 variance_threshold=0.1,
                 correlation_threshold=0.9,
                 auto_select=True):
        """        
        problem_type : 'classification' or 'regression'
        n_features : Number of features to select (None for auto-selection)
        selection_methods : ['variance', 'correlation', 'univariate', 'rfe', 
                           'importance', 'lasso', 'all']
        model : model object
        variance_threshold : (obvious)
        correlation_threshold : (obvious)
        auto_select : auto detect best features
        """
        self.problem_type = problem_type.lower()
        self.n_features = n_features
        self.selection_methods = ['variance', 'correlation', 'univariate', 'rfe', 
                                'importance', 'lasso'] if 'all' in selection_methods else selection_methods
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.auto_select = auto_select
        self.scaler = StandardScaler()
        self.model = dispatcher.MODELS[MODEL]
        
        # Model selection
        # if self.problem_type == "classification":
        #     model_name = MODEL
        # if model is None:
        #     if model_name.lower() == 'Lasso':
        #         self.model = Lasso() if problem_type == 'regression' else LogisticRegression(penalty='l1', solver='liblinear')
        #     else:
        #         self.model = (RandomForestRegressor() if problem_type == 'regression' 
        #                     else RandomForestClassifier())
        # else:
        #     self.model = model
            
        self.selectors = {}
        self.feature_mask = None
        self.feature_importance = {}
        
    def _variance_selection(self, X):
        selector = VarianceThreshold(threshold=self.variance_threshold)
        return selector.fit_transform(X), selector.get_support()
    
    def _correlation_selection(self, X, feature_names):
        if isinstance(X, np.ndarray):
            if len(feature_names) != X.shape[1]:
                raise ValueError("Length of feature_names does not match number of columns in X")
            X = pd.DataFrame(X, columns=feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame or NumPy array")
        # Check for numeric data
        if not X.dtypes.all() in [np.float64, np.int64, np.float32, np.int32]:
            raise ValueError("All columns in X must be numeric for correlation selection")
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.correlation_threshold)]
        keep_mask = ~X.columns.isin(to_drop)
        return X.loc[:, keep_mask].to_numpy(), keep_mask.to_numpy()
    
    def _univariate_selection(self, X, y):
        scoring = {
            'classification': {'f_classif': f_classif, 'chi2': chi2, 
                             'mutual_info': mutual_info_classif},
            'regression': {'f_regression': f_regression, 
                          'mutual_info': mutual_info_regression}
        }
        best_score = -np.inf
        best_selector = None
        
        for name, score_func in scoring[self.problem_type].items():
            selector = SelectKBest(score_func=score_func, 
                                 k=self.n_features or 'all')
            selector.fit(X, y)
            scores = selector.scores_
            if scores.max() > best_score:
                best_score = scores.max()
                best_selector = selector
                
        return best_selector.transform(X), best_selector.get_support()
    
    def _rfe_selection(self, X, y):
        selector = RFE(estimator=self.model, 
                      n_features_to_select=self.n_features)
        return selector.fit_transform(X, y), selector.get_support()
    
    def _importance_selection(self, X, y):
        selector = SelectFromModel(estimator=self.model, 
                                 prefit=False,
                                 max_features=self.n_features)
        return selector.fit_transform(X, y), selector.get_support()
    
    def _lasso_selection(self, X, y):
        lasso = Lasso() if self.problem_type == 'regression' else LogisticRegression(penalty='l1', solver='liblinear')
        selector = SelectFromModel(estimator=lasso, 
                                 prefit=False,
                                 max_features=self.n_features)
        return selector.fit_transform(X, y), selector.get_support()
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the feature selector to the data.
        """
        print("fitting feature selector")
        X = np.asarray(X)
        y = np.asarray(y)
        X_scaled = self.scaler.fit_transform(X)
        
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            
        current_X = X_scaled
        combined_mask = np.ones(X.shape[1], dtype=bool)
        method_results = {}
        
        for method in self.selection_methods:
            try:
                if method == 'variance':
                    X_transformed, mask = self._variance_selection(current_X)
                elif method == 'correlation':
                    X_transformed, mask = self._correlation_selection(current_X, feature_names)
                elif method == 'univariate':
                    X_transformed, mask = self._univariate_selection(current_X, y)
                elif method == 'rfe':
                    X_transformed, mask = self._rfe_selection(current_X, y)
                elif method == 'importance':
                    X_transformed, mask = self._importance_selection(current_X, y)
                elif method == 'lasso':
                    X_transformed, mask = self._lasso_selection(current_X, y)
                else:
                    continue
                    
                method_results[method] = {'X': X_transformed, 'mask': mask}
                combined_mask = combined_mask & mask
                current_X = current_X[:, mask]
                
            except Exception as e:
                warnings.warn(f"Method {method} failed: {str(e)}")
                
        self.selectors = method_results
        self.feature_mask = combined_mask
        
        # Calculate feature importance
        if 'importance' in method_results:
            self.feature_importance = dict(zip(feature_names, 
                                             self.model.fit(X_scaled, y).feature_importances_))
        elif 'univariate' in method_results:
            self.feature_importance = dict(zip(feature_names, 
                                             method_results['univariate']['selector'].scores_))
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        print("transforming using feature selector")
        if self.feature_mask is None:
            raise ValueError("Fit the selector first.")
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.feature_mask]
    
    def fit_transform(self, X, y, feature_names=None):
        """Fit and transform in one step."""
        return self.fit(X, y, feature_names).transform(X)
    
    def get_feature_names(self, feature_names=None):
        """Get names of selected features."""
        if self.feature_mask is None:
            raise ValueError("Fit the selector first.")
        if feature_names is None:
            return np.where(self.feature_mask)[0].tolist()
        return [feature_names[i] for i in np.where(self.feature_mask)[0]]
    
    def save_results(self, path='feature_selection_results'):
        """Save feature importance to file."""
        if self.feature_importance:
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            })
            importance_df.to_csv(f"{path}_importance.csv", index=False)

# Example usage
# if __name__ == "__main__":    
    # Load data
    # X, y = data.data, data.target
    # feature_names = data.feature_names
    
    # # Initialize selector
    # selector = CustomFeatureSelector(
    #     problem_type='regression',
    #     selection_methods=['variance', 'correlation', 'univariate', 'rfe'],
    #     n_features=None,  # Auto-select
    #     auto_select=True
    # )
    
    # # Fit and transform
    # X_transformed = selector.fit_transform(X, y, feature_names)
    # selected_features = selector.get_feature_names(feature_names)
    
    # print(f"Selected features: {selected_features}")
    # selector.save_results()
