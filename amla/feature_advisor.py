import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class FeatureAugmentationAdvisor:
    def __init__(self):
        self.recommendations = []
        
    def analyse(self, df, target_col):
        self.recommendations = []
        
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('missing')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        self._detect_weak_features(X, y_encoded)
        self._detect_redundant_features(X, y_encoded)
        self._detect_skewed_features(X)
        self._detect_missing_value_issues(df.drop(columns=[target_col]))
        self._detect_imbalance(y)
        
        return self.recommendations
    
    def _detect_weak_features(self, X, y_encoded):
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        
        for i, col in enumerate(X.columns):
            if X[col].dtype in [np.float64, np.int64]:
                variance = np.var(X[col].dropna())
                
                if variance < 0.01:
                    self.recommendations.append({
                        'feature': col,
                        'issue_type': 'weak_near_zero_variance',
                        'severity': 'high',
                        'metric_value': round(variance, 6),
                        'threshold': 0.01,
                        'action': f'Remove feature "{col}" - near-zero variance ({variance:.6f}) indicates no discriminative power'
                    })
                
                if mi_scores[i] < 0.01:
                    self.recommendations.append({
                        'feature': col,
                        'issue_type': 'weak_low_mutual_info',
                        'severity': 'medium',
                        'metric_value': round(mi_scores[i], 4),
                        'threshold': 0.01,
                        'action': f'Consider removing or transforming "{col}" - very low mutual information ({mi_scores[i]:.4f}) with target'
                    })
    
    def _detect_redundant_features(self, X, y_encoded):
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return
        
        corr_matrix = X[numerical_cols].corr().abs()
        
        mi_scores = {}
        for col in numerical_cols:
            try:
                mi = mutual_info_classif(X[[col]], y_encoded, random_state=42, discrete_features=False)[0]
                mi_scores[col] = mi
            except:
                mi_scores[col] = 0
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if corr_val > 0.92:
                    keep_col = col1 if mi_scores[col1] >= mi_scores[col2] else col2
                    remove_col = col2 if keep_col == col1 else col1
                    
                    self.recommendations.append({
                        'feature': f'{keep_col}/{remove_col}',
                        'issue_type': 'redundant_high_correlation',
                        'severity': 'high',
                        'metric_value': round(corr_val, 4),
                        'threshold': 0.92,
                        'action': f'Features "{col1}" and "{col2}" are highly correlated ({corr_val:.2%}). Keep "{keep_col}" (MI={mi_scores[keep_col]:.4f}), remove "{remove_col}"'
                    })
    
    def _detect_skewed_features(self, X):
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            col_data = X[col].dropna()
            if len(col_data) > 3:
                skewness_val = skew(col_data)
                
                if abs(skewness_val) > 1.5:
                    all_positive = (X[col] >= 0).all()
                    
                    if all_positive:
                        transform = 'log1p'
                        action = f'Apply log1p transformation to "{col}" - highly skewed (skewness={skewness_val:.2f})'
                    else:
                        transform = 'Box-Cox'
                        action = f'Apply Box-Cox or Yeo-Johnson transformation to "{col}" - highly skewed (skewness={skewness_val:.2f})'
                    
                    self.recommendations.append({
                        'feature': col,
                        'issue_type': 'highly_skewed',
                        'severity': 'medium',
                        'metric_value': round(skewness_val, 4),
                        'threshold': 1.5,
                        'action': action,
                        'recommended_transform': transform
                    })
    
    def _detect_missing_value_issues(self, X):
        for col in X.columns:
            missing_pct = X[col].isnull().sum() / len(X) * 100
            
            if missing_pct > 30:
                self.recommendations.append({
                    'feature': col,
                    'issue_type': 'high_missing',
                    'severity': 'high',
                    'metric_value': round(missing_pct, 2),
                    'threshold': 30.0,
                    'action': f'Consider removing column "{col}" - {missing_pct:.1f}% missing values exceeds 30% threshold'
                })
            elif missing_pct > 0:
                if X[col].dtype in [np.float64, np.int64]:
                    strategy = 'median'
                else:
                    strategy = 'mode'
                
                self.recommendations.append({
                    'feature': col,
                    'issue_type': 'moderate_missing',
                    'severity': 'low',
                    'metric_value': round(missing_pct, 2),
                    'threshold': 30.0,
                    'action': f'Impute "{col}" using {strategy} imputation - {missing_pct:.1f}% missing values'
                })
    
    def _detect_imbalance(self, y):
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        
        if imbalance_ratio > 10:
            self.recommendations.append({
                'feature': 'target',
                'issue_type': 'class_imbalance',
                'severity': 'high',
                'metric_value': round(imbalance_ratio, 2),
                'threshold': 10.0,
                'action': f'Class imbalance detected (ratio={imbalance_ratio:.1f}). Consider SMOTE oversampling, class weighting, or threshold adjustment'
            })
        elif imbalance_ratio > 5:
            self.recommendations.append({
                'feature': 'target',
                'issue_type': 'moderate_imbalance',
                'severity': 'medium',
                'metric_value': round(imbalance_ratio, 2),
                'threshold': 5.0,
                'action': f'Moderate class imbalance (ratio={imbalance_ratio:.1f}). Consider class_weight="balanced" in your model'
            })
    
    def get_health_summary(self):
        if not self.recommendations:
            return {
                'status': 'healthy',
                'total_issues': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'summary': 'No significant feature issues detected.'
            }
        
        high = sum(1 for r in self.recommendations if r['severity'] == 'high')
        medium = sum(1 for r in self.recommendations if r['severity'] == 'medium')
        low = sum(1 for r in self.recommendations if r['severity'] == 'low')
        
        issue_types = {}
        for r in self.recommendations:
            issue_types[r['issue_type']] = issue_types.get(r['issue_type'], 0) + 1
        
        if high > 0:
            status = 'critical'
        elif medium > 0:
            status = 'warning'
        else:
            status = 'minor'
        
        return {
            'status': status,
            'total_issues': len(self.recommendations),
            'high_severity': high,
            'medium_severity': medium,
            'low_severity': low,
            'issue_types': issue_types,
            'summary': f'Found {high} high, {medium} medium, and {low} low severity issues.'
        }
