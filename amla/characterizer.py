import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DatasetCharacterizer:
    def __init__(self):
        self.dmfv = {}
        self.dataset_hash = None
        
    def _compute_hash(self, df):
        return hex(hash(str(df.shape) + str(df.columns.tolist())))
    
    def _prepare_data(self, df, target_col):
        df_copy = df.copy()
        y = df_copy[target_col].copy()
        X = df_copy.drop(columns=[target_col])
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        X_imputed = X_imputed.fillna(0)
        
        return X_imputed, y
    
    def extract_layer1_simple_stats(self, df, target_col):
        X, y = self._prepare_data(df, target_col)
        
        n_instances = len(df)
        n_features = len(df.columns) - 1
        missing_pct = (df.isnull().sum().sum() / (n_instances * n_features)) * 100
        
        n_classes = y.nunique()
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        n_numerical = len(numerical_cols)
        n_categorical = len(categorical_cols)
        
        self.dmfv['l1_n_instances'] = round(n_instances, 4)
        self.dmfv['l1_n_features'] = round(n_features, 4)
        self.dmfv['l1_missing_pct'] = round(missing_pct, 4)
        self.dmfv['l1_n_classes'] = round(n_classes, 4)
        self.dmfv['l1_imbalance_ratio'] = round(min(imbalance_ratio, 1000), 4)
        self.dmfv['l1_n_numerical'] = round(n_numerical, 4)
        self.dmfv['l1_n_categorical'] = round(n_categorical, 4)
        
        return self.dmfv
    
    def extract_layer2_distribution_features(self, df, target_col):
        X, y = self._prepare_data(df, target_col)
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.dmfv['l2_mean_skewness'] = 0.0
            self.dmfv['l2_mean_kurtosis'] = 0.0
            self.dmfv['l2_mean_variance'] = 0.0
            self.dmfv['l2_mean_correlation'] = 0.0
            self.dmfv['l2_outlier_ratio'] = 0.0
            return self.dmfv
        
        skewness_vals = []
        kurtosis_vals = []
        variance_vals = []
        
        for col in numerical_cols:
            col_data = X[col].dropna()
            if len(col_data) > 3:
                skewness_vals.append(skew(col_data))
                kurtosis_vals.append(kurtosis(col_data))
                variance_vals.append(np.var(col_data))
        
        self.dmfv['l2_mean_skewness'] = round(np.mean(skewness_vals) if skewness_vals else 0, 4)
        self.dmfv['l2_mean_kurtosis'] = round(np.mean(kurtosis_vals) if kurtosis_vals else 0, 4)
        self.dmfv['l2_mean_variance'] = round(np.mean(variance_vals) if variance_vals else 0, 4)
        
        corr_matrix = X[numerical_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_corr = upper_tri.stack().mean() if not upper_tri.stack().empty else 0
        self.dmfv['l2_mean_correlation'] = round(mean_corr, 4)
        
        outlier_counts = 0
        total_points = 0
        for col in numerical_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((X[col] < lower) | (X[col] > upper)).sum()
            outlier_counts += outliers
            total_points += len(X[col])
        
        self.dmfv['l2_outlier_ratio'] = round(outlier_counts / total_points if total_points > 0 else 0, 4)
        
        return self.dmfv
    
    def extract_layer3_info_theoretic(self, df, target_col):
        X, y = self._prepare_data(df, target_col)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42, discrete_features=False)
        mean_mi = np.mean(mi_scores)
        max_mi = np.max(mi_scores)
        
        feature_pairs_mi = []
        for i in range(len(X.columns)):
            for j in range(i+1, len(X.columns)):
                pair_mi = mutual_info_classif(
                    X.iloc[:, [i, j]], y_encoded, random_state=42, discrete_features=False
                ).sum()
                feature_pairs_mi.append(pair_mi)
        
        redundancy_score = np.mean(feature_pairs_mi) if feature_pairs_mi else 0
        
        class_probs = pd.Series(y).value_counts(normalize=True)
        class_entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
        
        self.dmfv['l3_mean_mutual_info'] = round(mean_mi, 4)
        self.dmfv['l3_max_mutual_info'] = round(max_mi, 4)
        self.dmfv['l3_redundancy_score'] = round(redundancy_score, 4)
        self.dmfv['l3_class_entropy'] = round(class_entropy, 4)
        
        return self.dmfv
    
    def extract_layer4_landmarkers(self, df, target_col):
        X, y = self._prepare_data(df, target_col)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cv = 3
        
        dt_scores = []
        nb_scores = []
        knn_scores = []
        svm_scores = []
        
        try:
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt_scores = cross_val_score(dt, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
        except:
            dt_scores = [0.0]
        
        try:
            nb = GaussianNB()
            nb_scores = cross_val_score(nb, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
        except:
            nb_scores = [0.0]
        
        try:
            n_neighbors = min(3, len(X_scaled) - 1)
            if n_neighbors < 1:
                n_neighbors = 1
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
        except:
            knn_scores = [0.0]
        
        try:
            svm = LinearSVC(random_state=42, max_iter=2000)
            svm_scores = cross_val_score(svm, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
        except:
            svm_scores = [0.0]
        
        self.dmfv['l4_landmark_dt'] = round(np.mean(dt_scores) if len(dt_scores) > 0 else 0, 4)
        self.dmfv['l4_landmark_nb'] = round(np.mean(nb_scores) if len(nb_scores) > 0 else 0, 4)
        self.dmfv['l4_landmark_knn'] = round(np.mean(knn_scores) if len(knn_scores) > 0 else 0, 4)
        self.dmfv['l4_landmark_svm'] = round(np.mean(svm_scores) if len(svm_scores) > 0 else 0, 4)
        
        return self.dmfv
    
    def extract_all(self, df, target_col):
        self.dmfv = {}
        self.dataset_hash = self._compute_hash(df)
        
        self.extract_layer1_simple_stats(df, target_col)
        self.extract_layer2_distribution_features(df, target_col)
        self.extract_layer3_info_theoretic(df, target_col)
        self.extract_layer4_landmarkers(df, target_col)
        
        self.dmfv['dataset_hash'] = self.dataset_hash
        
        return self.dmfv
    
    def get_dmfv_dataframe(self):
        if not self.dmfv:
            return pd.DataFrame()
        
        dmfv_df = pd.DataFrame([self.dmfv])
        feature_cols = [col for col in dmfv_df.columns if col != 'dataset_hash']
        return dmfv_df[feature_cols]
