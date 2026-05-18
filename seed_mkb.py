import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from amla.mkb import MetaKnowledgeBase
from amla.characterizer import DatasetCharacterizer


ALGORITHM_POOL = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}


def load_synthetic_dataset(dataset_id):
    n_samples_map = {
        1: 200,
        2: 500,
        3: 1000,
        4: 2000,
        5: 500,
        6: 800,
        7: 1500,
        8: 300,
        9: 600,
        10: 1200
    }
    
    n_features_map = {
        1: 10,
        2: 15,
        3: 20,
        4: 25,
        5: 8,
        6: 12,
        7: 18,
        8: 6,
        9: 14,
        10: 22
    }
    
    n_informative = min(8, n_features_map[dataset_id] - 2)
    n_redundant = min(2, n_features_map[dataset_id] - n_informative)
    
    X, y = make_classification(
        n_samples=n_samples_map[dataset_id],
        n_features=n_features_map[dataset_id],
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        n_classes=2,
        random_state=dataset_id * 42,
        flip_y=0.05
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features_map[dataset_id])])
    df['target'] = y
    
    return df


def load_sklearn_dataset(dataset_id):
    dataset_loaders = {
        11: load_iris,
        12: load_breast_cancer,
        13: load_wine
    }
    
    loader = dataset_loaders[dataset_id]
    data = loader()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    return df


def generate_benchmark_dataset(dataset_id, dataset_name, domain, introduce_noise=False):
    if dataset_id <= 10:
        df = load_synthetic_dataset(dataset_id)
    else:
        df = load_sklearn_dataset(dataset_id)
    
    if introduce_noise and dataset_id <= 10:
        noise_cols = np.random.choice(df.columns[:-1], size=min(2, len(df.columns)-1), replace=False)
        for col in noise_cols:
            if col in df.columns:
                df[col] = df[col] + np.random.normal(0, df[col].std() * 0.5, len(df))
    
    n_missing = int(len(df) * 0.05)
    missing_indices = np.random.choice(len(df), size=n_missing, replace=False)
    missing_col = np.random.choice(df.columns[:-1])
    df.loc[missing_indices, missing_col] = np.nan
    
    return df


def run_experiment(df, target_col='target'):
    characterizer = DatasetCharacterizer()
    meta_features = characterizer.extract_all(df, target_col)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    X = X.fillna(X.median())
    
    results = {}
    
    for algo_name, model in ALGORITHM_POOL.items():
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            prec_scores = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')
            rec_scores = cross_val_score(model, X, y, cv=5, scoring='recall_weighted')
            
            results[algo_name] = {
                'f1': float(np.mean(scores)),
                'accuracy': float(np.mean(acc_scores)),
                'precision': float(np.mean(prec_scores)),
                'recall': float(np.mean(rec_scores)),
                'cv_folds': 5
            }
        except Exception as e:
            print(f"Error running {algo_name}: {e}")
            results[algo_name] = {'f1': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    
    return meta_features, results


def seed_mkb(target_count=50):
    mkb = MetaKnowledgeBase(db_path='data/mkb.db')
    
    existing_count = mkb.get_experiment_count()
    print(f"Existing experiments in MKB: {existing_count}")
    
    datasets = [
        (1, 'Syn-Binary-Small', 'general', False),
        (2, 'Syn-Binary-Med', 'general', False),
        (3, 'Syn-Binary-Large', 'general', False),
        (4, 'Syn-Binary-XLarge', 'general', False),
        (5, 'Syn-Binary-Tiny', 'general', False),
        (6, 'Syn-Binary-Mid', 'general', True),
        (7, 'Syn-Binary-Balanced', 'general', False),
        (8, 'Syn-Binary-Complex', 'general', False),
        (9, 'Syn-Binary-Noisy', 'general', True),
        (10, 'Syn-Binary-Extended', 'general', False),
        (11, 'Iris-Dataset', 'botanical', False),
        (12, 'Breast-Cancer', 'medical', False),
        (13, 'Wine-Quality', 'food', False),
    ]
    
    for i in range(14, 21):
        datasets.append((i, f'Syn-Gen-{i}', 'general', i % 2 == 0))
    
    additional_datasets = [
        (21, 'Credit-Approval', 'financial', False),
        (22, 'Heart-Disease', 'medical', False),
        (23, 'Spam-Detection', 'computer', False),
        (24, 'Customer-Churn', 'business', False),
        (25, 'Loan-Default', 'financial', False),
        (26, 'Disease-Diagnosis', 'medical', False),
        (27, 'Image-Classification', 'computer', False),
        (28, 'Sentiment-Analysis', 'nlp', False),
        (29, 'Fraud-Detection', 'financial', False),
        (30, 'Market-Basket', 'business', False),
    ]
    datasets.extend(additional_datasets)
    
    for i in range(21, 26):
        n_samples = np.random.randint(300, 5000)
        n_features = np.random.randint(8, 30)
        domain = np.random.choice(['general', 'financial', 'medical', 'scientific'])
        datasets.append((i, f'Synth-Random-{i}', domain, np.random.random() < 0.3))
    
    print(f"Target datasets to process: {len(datasets)}")
    
    added = 0
    for dataset_id, dataset_name, domain, noisy in datasets:
        try:
            print(f"\nProcessing {dataset_id}: {dataset_name}...")
            
            df = generate_benchmark_dataset(dataset_id, dataset_name, domain, noisy)
            
            print(f"  Dataset shape: {df.shape}")
            
            meta_features, results = run_experiment(df)
            
            print(f"  Meta-features extracted: {len(meta_features)} features")
            results_str = ', '.join([f'{k}={v["f1"]:.3f}' for k, v in results.items()])
            print(f"  Results: {results_str}")
            
            best_algo = max(results.items(), key=lambda x: x[1]['f1'])[0]
            print(f"  Best algorithm: {best_algo}")
            
            mkb.add_experiment(
                dataset_hash=meta_features['dataset_hash'],
                dataset_name=dataset_name,
                domain=domain,
                meta_features=meta_features,
                algorithm_results=results
            )
            
            added += 1
            print(f"  Added to MKB (Total: {added})")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"SEEDING COMPLETE")
    print(f"{'='*50}")
    print(f"Total experiments added: {added}")
    print(f"Final MKB size: {mkb.get_experiment_count()}")
    
    return mkb


if __name__ == "__main__":
    print("AMLA Meta-Knowledge Base Seeder")
    print("=" * 50)
    seed_mkb(target_count=50)
