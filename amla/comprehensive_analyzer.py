import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from .model_evaluator import ModelEvaluator
from .visualizations import VisualizationGenerator


class ComprehensiveAnalyzer:
    def __init__(self, output_dir='analysis_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluator = ModelEvaluator()
        self.visualizer = VisualizationGenerator()
        self.algorithm_pool = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'NaiveBayes': GaussianNB()
        }
    
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
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        return X_scaled, y_encoded, X_imputed.columns.tolist(), le.classes_.tolist()
    
    def analyze_dataset(self, df, target_col, algorithms=None, cv=5):
        X, y, feature_names, classes = self._prepare_data(df, target_col)
        
        if algorithms is None:
            algorithms = list(self.algorithm_pool.keys())
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,
                'n_classes': len(classes),
                'class_distribution': pd.Series(y).value_counts().to_dict(),
                'target_column': target_col
            },
            'models': {},
            'best_model': None,
            'best_score': 0,
            'visualizations': {},
            'all_figures': {}
        }
        
        print("="*70)
        print("COMPREHENSIVE ML ANALYSIS")
        print("="*70)
        print(f"\nDataset: {len(df)} samples, {len(df.columns)-1} features, {len(classes)} classes")
        
        dataset_figs = self.visualizer.generate_dataset_overview(df, target_col)
        for fig_name, fig in dataset_figs.items():
            fig_path = os.path.join(self.output_dir, f'dataset_{fig_name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            results['visualizations'][f'dataset_{fig_name}'] = fig_path
        
        print("\n" + "-"*70)
        print("TRAINING AND EVALUATING MODELS")
        print("-"*70)
        
        for algo_name in algorithms:
            if algo_name not in self.algorithm_pool:
                continue
            
            print(f"\n>>> Evaluating: {algo_name}")
            
            model = self.algorithm_pool[algo_name]
            
            model_results = self.evaluator.evaluate_all(model, X, y, algo_name, cv=cv)
            
            results['models'][algo_name] = model_results
            
            metrics = model_results.get('metrics', {})
            f1 = metrics.get('f1_weighted', 0)
            
            if f1 > results['best_score']:
                results['best_score'] = f1
                results['best_model'] = algo_name
            
            print(f"    Accuracy:   {metrics.get('accuracy', 0):.4f}")
            print(f"    F1-Score:  {f1:.4f}")
            print(f"    ROC-AUC:   {metrics.get('roc_auc', metrics.get('roc_auc_macro', 0)):.4f}")
            
            model_figs = self.evaluator.figures.get(algo_name, {})
            for fig_name, fig in model_figs.items():
                fig_path = os.path.join(self.output_dir, f'{algo_name}_{fig_name}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                if algo_name not in results['all_figures']:
                    results['all_figures'][algo_name] = {}
                results['all_figures'][algo_name][fig_name] = fig_path
            
            learning_fig, learning_data = self.visualizer.generate_learning_curve(
                model, X, y, algo_name, cv=cv
            )
            learning_path = os.path.join(self.output_dir, f'{algo_name}_learning_curve.png')
            learning_fig.savefig(learning_path, dpi=300, bbox_inches='tight')
            results['all_figures'][algo_name]['learning_curve'] = learning_path
            results['models'][algo_name]['learning_curve_data'] = learning_data
            
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importance_fig = self.visualizer.generate_feature_importance(
                    model, feature_names, algo_name
                )
                importance_path = os.path.join(self.output_dir, f'{algo_name}_feature_importance.png')
                importance_fig.savefig(importance_path, dpi=300, bbox_inches='tight')
                results['all_figures'][algo_name]['feature_importance'] = importance_path
            
            pred_figs = self.visualizer.generate_prediction_analysis(
                y, model_results['predictions'], model_results.get('proba'), algo_name
            )
            for fig_name, fig in pred_figs.items():
                fig_path = os.path.join(self.output_dir, f'{algo_name}_{fig_name}.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                results['all_figures'][algo_name][fig_name] = fig_path
            
            if len(classes) > 2:
                multi_roc = self.visualizer.generate_multiclass_roc_curves(
                    y, model_results.get('proba'), algo_name, classes
                )
                if multi_roc is not None:
                    multi_roc_path = os.path.join(self.output_dir, f'{algo_name}_multiclass_roc.png')
                    multi_roc.savefig(multi_roc_path, dpi=300, bbox_inches='tight')
                    results['all_figures'][algo_name]['multiclass_roc'] = multi_roc_path
            
            calibration_fig = self.visualizer.generate_calibration_curve(model, X, y, algo_name, cv=cv)
            calibration_path = os.path.join(self.output_dir, f'{algo_name}_calibration.png')
            calibration_fig.savefig(calibration_path, dpi=300, bbox_inches='tight')
            results['all_figures'][algo_name]['calibration'] = calibration_path
            
            per_class_fig = self.visualizer.generate_per_class_analysis(
                y, model_results['predictions'], algo_name, classes
            )
            per_class_path = os.path.join(self.output_dir, f'{algo_name}_per_class.png')
            per_class_fig.savefig(per_class_path, dpi=300, bbox_inches='tight')
            results['all_figures'][algo_name]['per_class_analysis'] = per_class_path
            
            dashboard_figs = self.visualizer.generate_comprehensive_dashboard(model_results, algo_name)
            for fig_name, fig in dashboard_figs.items():
                dashboard_path = os.path.join(self.output_dir, f'{algo_name}_{fig_name}.png')
                fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                results['all_figures'][algo_name][fig_name] = dashboard_path
        
        print("\n" + "-"*70)
        print("GENERATING COMPARISON PLOTS")
        print("-"*70)
        
        comparison_figs = self.evaluator.generate_comparison_plots()
        for fig_name, fig in comparison_figs.items():
            fig_path = os.path.join(self.output_dir, f'comparison_{fig_name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            results['visualizations'][f'comparison_{fig_name}'] = fig_path
        
        metrics_df = self.evaluator.get_metrics_dataframe()
        metrics_csv_path = os.path.join(self.output_dir, 'metrics_comparison.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        results['visualizations']['metrics_comparison_csv'] = metrics_csv_path
        
        results['summary'] = self.evaluator.get_summary()
        
        self._generate_report(results)
        
        plt.close('all')
        
        return results
    
    def _generate_report(self, results):
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ML MODEL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {results['timestamp']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("DATASET INFORMATION\n")
            f.write("-"*80 + "\n")
            info = results['dataset_info']
            f.write(f"Samples:        {info['n_samples']}\n")
            f.write(f"Features:       {info['n_features']}\n")
            f.write(f"Classes:        {info['n_classes']}\n")
            f.write(f"Target Column:  {info['target_column']}\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("-"*80 + "\n")
            
            metrics_df = self.evaluator.get_metrics_dataframe()
            f.write(metrics_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("-"*80 + "\n")
            f.write("BEST MODEL\n")
            f.write("-"*80 + "\n")
            f.write(f"Model:          {results['best_model']}\n")
            f.write(f"F1-Score:       {results['best_score']:.4f}\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("GENERATED FILES\n")
            f.write("-"*80 + "\n")
            for section, files in results['all_figures'].items():
                f.write(f"\n{section}:\n")
                for file_type, path in files.items():
                    f.write(f"  - {file_type}: {os.path.basename(path)}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        results['report_path'] = report_path
    
    def benchmark_algorithms(self, df, target_col, cv=5):
        X, y, _, classes = self._prepare_data(df, target_col)
        
        results = {}
        
        print("\n" + "="*70)
        print("ALGORITHM BENCHMARK")
        print("="*70)
        
        for algo_name, model in self.algorithm_pool.items():
            print(f"\nBenchmarking {algo_name}...")
            
            scores = {
                'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
                'f1_weighted': cross_val_score(model, X, y, cv=cv, scoring='f1_weighted'),
                'f1_macro': cross_val_score(model, X, y, cv=cv, scoring='f1_macro'),
                'precision_weighted': cross_val_score(model, X, y, cv=cv, scoring='precision_weighted'),
                'recall_weighted': cross_val_score(model, X, y, cv=cv, scoring='recall_weighted'),
                'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc') if len(classes) == 2 else None
            }
            
            results[algo_name] = {
                metric: {
                    'mean': float(np.mean(scores[metric])) if scores[metric] is not None else None,
                    'std': float(np.std(scores[metric])) if scores[metric] is not None else None,
                    'scores': scores[metric].tolist() if scores[metric] is not None else None
                }
                for metric in scores
            }
            
            print(f"  Accuracy:  {results[algo_name]['accuracy']['mean']:.4f} ± {results[algo_name]['accuracy']['std']:.4f}")
            print(f"  F1:        {results[algo_name]['f1_weighted']['mean']:.4f} ± {results[algo_name]['f1_weighted']['std']:.4f}")
        
        return results
    
    def compare_best_models(self, df, target_col, top_n=3, cv=5):
        benchmark_results = self.benchmark_algorithms(df, target_col, cv)
        
        sorted_models = sorted(
            benchmark_results.items(),
            key=lambda x: x[1]['f1_weighted']['mean'],
            reverse=True
        )[:top_n]
        
        print("\n" + "="*70)
        print(f"TOP {top_n} MODELS - DETAILED COMPARISON")
        print("="*70)
        
        comparison_results = {}
        
        for rank, (model_name, scores) in enumerate(sorted_models, 1):
            print(f"\n#{rank} {model_name}")
            print(f"    F1-Score:      {scores['f1_weighted']['mean']:.4f} ± {scores['f1_weighted']['std']:.4f}")
            print(f"    Accuracy:      {scores['accuracy']['mean']:.4f} ± {scores['accuracy']['std']:.4f}")
            print(f"    Precision:     {scores['precision_weighted']['mean']:.4f} ± {scores['precision_weighted']['std']:.4f}")
            print(f"    Recall:        {scores['recall_weighted']['mean']:.4f} ± {scores['recall_weighted']['std']:.4f}")
            
            comparison_results[model_name] = scores
        
        return comparison_results, sorted_models[0][0] if sorted_models else None
