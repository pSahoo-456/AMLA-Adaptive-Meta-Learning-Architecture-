import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

from .characterizer import DatasetCharacterizer
from .metalearner import MetaLearner
from .feature_advisor import FeatureAugmentationAdvisor
from .mkb import MetaKnowledgeBase


class AMLAPipeline:
    def __init__(self, model_path='models/metalearner.pkl', mkb_path='data/mkb.db'):
        self.characterizer = DatasetCharacterizer()
        self.metalearner = MetaLearner(model_path=model_path)
        self.metalearner.load()
        self.feature_advisor = FeatureAugmentationAdvisor()
        self.mkb = MetaKnowledgeBase(db_path=mkb_path)
        self.meta_features = None
        self.algorithm_pool = [
            'RandomForest',
            'GradientBoosting', 
            'LogisticRegression',
            'SVM',
            'KNN'
        ]
        self.comprehensive_analyzer = None
    
    def run(self, df, target_col, dataset_name='Unknown', domain='general', return_details=False):
        result = {
            'status': 'success',
            'dataset_name': dataset_name,
            'target_column': target_col,
            'n_instances': len(df),
            'n_features': len(df.columns) - 1
        }
        
        try:
            self.meta_features = self.characterizer.extract_all(df, target_col)
            result['meta_features'] = self.meta_features
            
            rf_recommendation = self.metalearner.predict(self.meta_features)
            similarity_recommendation = self.metalearner.get_similarity_recommendations(self.meta_features)
            
            combined_recommendation = self._combine_recommendations(
                rf_recommendation, 
                similarity_recommendation
            )
            result['algorithm_recommendation'] = combined_recommendation
            
            feature_analysis = self.feature_advisor.analyse(df, target_col)
            health_summary = self.feature_advisor.get_health_summary()
            result['feature_analysis'] = {
                'recommendations': feature_analysis,
                'health_summary': health_summary
            }
            
            if return_details:
                result['details'] = {
                    'dmfv': self.characterizer.get_dmfv_dataframe().to_dict(orient='records')[0] if not self.characterizer.get_dmfv_dataframe().empty else {},
                    'rf_prediction': rf_recommendation,
                    'similarity_prediction': similarity_recommendation
                }
            
            return result
            
        except Exception as e:
            import traceback
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
    
    def _combine_recommendations(self, rf_rec, sim_rec):
        rf_algorithm = rf_rec['ranked_algorithms']
        sim_performance = {a['algorithm']: a['avg_f1'] for a in sim_rec.get('algorithm_performance', [])}
        
        combined_scores = {}
        
        for i, algo_dict in enumerate(rf_algorithm):
            algo = algo_dict['algorithm']
            rf_weight = 1.0 / (i + 1)
            
            sim_f1 = sim_performance.get(algo, 0.5)
            sim_weight = sim_f1
            
            combined_scores[algo] = rf_weight * rf_algorithm[i]['probability'] + sim_weight * 0.5
        
        sorted_algos = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranked_list = []
        for algo, score in sorted_algos:
            rf_prob = next((a['probability'] for a in rf_algorithm if a['algorithm'] == algo), 0)
            ranked_list.append({
                'algorithm': algo,
                'combined_score': float(score),
                'rf_probability': float(rf_prob),
                'similarity_f1': float(sim_performance.get(algo, 0))
            })
        
        return {
            'recommended_algorithm': ranked_list[0]['algorithm'] if ranked_list else 'RandomForest',
            'ranked_algorithms': ranked_list,
            'confidence': float(rf_rec['confidence']),
            'method': 'combined',
            'similar_datasets': sim_rec.get('similar_datasets', [])
        }
    
    def run_benchmark_algorithms(self, df, target_col, cv_folds=5):
        results = {}
        
        X, y = self._prepare_data(df, target_col)
        
        if X is None or y is None:
            return {'error': 'Failed to prepare data'}
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        algorithms = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        for algo_name, model in algorithms.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='f1_weighted')
                acc_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
                prec_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='precision_weighted')
                rec_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='recall_weighted')
                
                results[algo_name] = {
                    'f1': float(np.mean(scores)),
                    'f1_std': float(np.std(scores)),
                    'accuracy': float(np.mean(acc_scores)),
                    'precision': float(np.mean(prec_scores)),
                    'recall': float(np.mean(rec_scores)),
                    'cv_folds': cv_folds
                }
            except Exception as e:
                results[algo_name] = {
                    'error': str(e)
                }
        
        if results:
            best_algo = max(results.items(), key=lambda x: x[1].get('f1', 0))
            results['best_algorithm'] = best_algo[0]
            results['winning_score'] = best_algo[1]['f1']
        
        return results
    
    def _prepare_data(self, df, target_col):
        df_copy = df.copy()
        
        if target_col not in df_copy.columns:
            return None, None
        
        y = df_copy[target_col].copy()
        X = df_copy.drop(columns=[target_col])
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return X_imputed, y_encoded
    
    def add_feedback(self, df, target_col, actual_algorithm, actual_f1_score):
        try:
            if df is not None and target_col is not None:
                meta_features = self.characterizer.extract_all(df, target_col)
            elif self.meta_features is not None:
                meta_features = self.meta_features
            else:
                return {
                    'status': 'error',
                    'message': 'No dataset context available. Run an analysis before recording feedback.'
                }
            
            algorithm_results = {actual_algorithm: {'f1': actual_f1_score}}
            
            self.mkb.add_experiment(
                dataset_hash=meta_features['dataset_hash'],
                dataset_name='user_dataset',
                domain='user',
                meta_features=meta_features,
                algorithm_results=algorithm_results
            )
            
            return {'status': 'success', 'message': 'Feedback recorded'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def retrain(self):
        try:
            self.metalearner.train()
            return {'status': 'success', 'message': 'Meta-learner retrained'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_system_stats(self):
        mkb_stats = self.mkb.get_algorithm_statistics()
        experiment_count = self.mkb.get_experiment_count()
        
        return {
            'total_experiments': experiment_count,
            'algorithm_statistics': mkb_stats.to_dict(orient='records') if not mkb_stats.empty else [],
            'model_loaded': self.metalearner.model is not None
        }
    
    def run_comprehensive_analysis(self, df, target_col, output_dir='analysis_results', cv=5):
        from .comprehensive_analyzer import ComprehensiveAnalyzer

        self.comprehensive_analyzer = ComprehensiveAnalyzer(output_dir=output_dir)
        results = self.comprehensive_analyzer.analyze_dataset(df, target_col, cv=cv)
        return results
