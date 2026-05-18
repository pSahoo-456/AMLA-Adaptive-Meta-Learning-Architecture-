import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
from .mkb import MetaKnowledgeBase


class MetaLearner:
    def __init__(self, model_path='models/metalearner.pkl'):
        self.model = None
        self.model_path = model_path
        self.meta_feature_columns = None
        self.algorithm_classes = None
        self.mkb = MetaKnowledgeBase()
    
    def _prepare_training_data(self):
        meta_matrix = self.mkb.get_meta_features_matrix()
        winners_df = self.mkb.get_winner_algorithms()
        
        if meta_matrix.empty or winners_df.empty:
            return None, None
        
        winners_df = winners_df.drop_duplicates(subset='experiment_id', keep='first')
        
        meta_matrix_clean = meta_matrix.drop(columns=['experiment_id', 'id', 'dataset_hash'], errors='ignore')
        
        feature_cols = [col for col in meta_matrix_clean.columns if not col.startswith('l1_') or 
                       col not in ['l1_n_numerical', 'l1_n_categorical']]
        
        meta_matrix_clean = meta_matrix_clean[feature_cols]
        
        merged = meta_matrix_clean.merge(
            winners_df[['experiment_id', 'algorithm_name']], 
            left_index=True, 
            right_index=True
        )
        
        X = merged[feature_cols]
        y = merged['algorithm_name']
        
        self.meta_feature_columns = feature_cols
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.algorithm_classes = le.classes_
        
        X = X.fillna(0)
        
        return X, y_encoded
    
    def train(self):
        X, y = self._prepare_training_data()
        
        if X is None or len(X) < 5:
            self._train_fallback_model()
            return self
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.meta_feature_columns,
            'algorithm_classes': self.algorithm_classes
        }, self.model_path)
        
        return self
    
    def _train_fallback_model(self):
        self.meta_feature_columns = [
            'l1_n_instances', 'l1_n_features', 'l1_missing_pct', 'l1_n_classes', 'l1_imbalance_ratio',
            'l2_mean_skewness', 'l2_mean_kurtosis', 'l2_mean_variance', 'l2_mean_correlation', 'l2_outlier_ratio',
            'l3_mean_mutual_info', 'l3_max_mutual_info', 'l3_redundancy_score', 'l3_class_entropy',
            'l4_landmark_dt', 'l4_landmark_nb', 'l4_landmark_knn', 'l4_landmark_svm'
        ]
        self.algorithm_classes = np.array([
            'RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN'
        ])
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        dummy_X = np.random.randn(10, len(self.meta_feature_columns))
        dummy_y = np.random.randint(0, len(self.algorithm_classes), 10)
        self.model.fit(dummy_X, dummy_y)
    
    def load(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.meta_feature_columns = data['feature_columns']
            self.algorithm_classes = data['algorithm_classes']
        return self
    
    def predict(self, meta_features):
        if self.model is None:
            self.load()
        
        if self.model is None:
            return self._get_default_predictions(meta_features)
        
        feature_vector = self._prepare_feature_vector(meta_features)
        
        if feature_vector is None:
            return self._get_default_predictions(meta_features)
        
        try:
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            predicted_algorithm = self.algorithm_classes[prediction]
            
            ranked_algorithms = []
            prob_sorted_indices = np.argsort(probabilities)[::-1]
            for idx in prob_sorted_indices:
                ranked_algorithms.append({
                    'algorithm': str(self.algorithm_classes[idx]),
                    'probability': float(probabilities[idx])
                })
            
            return {
                'recommended_algorithm': str(predicted_algorithm),
                'ranked_algorithms': ranked_algorithms,
                'confidence': float(np.max(probabilities)),
                'method': 'random_forest'
            }
            
        except Exception as e:
            return self._get_default_predictions(meta_features)
    
    def _prepare_feature_vector(self, meta_features):
        if self.meta_feature_columns is None:
            return None
        
        feature_values = []
        for col in self.meta_feature_columns:
            value = meta_features.get(col, 0)
            if value is None:
                value = 0
            feature_values.append(float(value))
        
        return np.array(feature_values).reshape(1, -1)
    
    def _get_default_predictions(self, meta_features):
        algorithms = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN']
        
        return {
            'recommended_algorithm': 'RandomForest',
            'ranked_algorithms': [{'algorithm': a, 'probability': 0.2} for a in algorithms],
            'confidence': 0.2,
            'method': 'default'
        }
    
    def get_similarity_recommendations(self, meta_features, top_k=3):
        similar_experiments = self.mkb.find_similar_experiments(meta_features, top_k=top_k)
        
        if not similar_experiments:
            return {
                'recommended_algorithm': 'RandomForest',
                'similar_datasets': [],
                'method': 'similarity'
            }
        
        algorithm_results_df = None
        from .mkb import MetaKnowledgeBase
        mkb = MetaKnowledgeBase()
        _, _, algorithm_results_df = mkb.get_all_experiments()
        
        if algorithm_results_df is None or algorithm_results_df.empty:
            return {
                'recommended_algorithm': 'RandomForest',
                'similar_datasets': similar_experiments,
                'method': 'similarity'
            }
        
        algorithm_counts = {}
        algorithm_scores = {}
        
        for exp in similar_experiments:
            exp_id = exp['experiment_id']
            results = algorithm_results_df[algorithm_results_df['experiment_id'] == exp_id]
            
            for _, row in results.iterrows():
                algo = row['algorithm_name']
                f1 = row['f1_score'] if pd.notna(row['f1_score']) else 0
                
                if algo not in algorithm_counts:
                    algorithm_counts[algo] = 0
                    algorithm_scores[algo] = []
                
                algorithm_counts[algo] += 1
                algorithm_scores[algo].append(f1)
        
        avg_scores = {algo: np.mean(scores) for algo, scores in algorithm_scores.items()}
        sorted_algos = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommended = sorted_algos[0][0] if sorted_algos else 'RandomForest'
        
        return {
            'recommended_algorithm': recommended,
            'similar_datasets': similar_experiments,
            'algorithm_performance': [{'algorithm': a, 'avg_f1': float(s)} for a, s in sorted_algos],
            'method': 'similarity'
        }
    
    def evaluate(self):
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        
        X, y = self._prepare_training_data()
        
        if X is None or len(X) < 5:
            return {
                'precision_at_1': 0.0,
                'precision_at_3': 0.0,
                'ndcg': 0.0,
                'n_samples': 0
            }
        
        loo = LeaveOneOut()
        predictions = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            pred = model.predict(X_test)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        correct = (predictions == y).sum()
        precision_at_1 = correct / len(y)
        
        precision_at_3 = 0.0
        ndcg = 0.0
        
        return {
            'precision_at_1': float(precision_at_1),
            'precision_at_3': float(precision_at_3),
            'ndcg': float(ndcg),
            'n_samples': len(y)
        }
