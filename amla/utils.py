import numpy as np
import pandas as pd
import hashlib
from datetime import datetime


def compute_dataset_hash(df):
    content = str(df.shape) + str(df.columns.tolist()) + str(df.dtypes.to_dict())
    return hashlib.md5(content.encode()).hexdigest()


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))


def ndcg_at_k(relevance_scores, k=None):
    if k is None:
        k = len(relevance_scores)
    
    relevance_scores = relevance_scores[:k]
    
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))
    
    if idcg == 0:
        return 0.0
    
    return float(dcg / idcg)


def precision_at_k(true_items, predicted_items, k):
    if k <= 0 or not true_items or not predicted_items:
        return 0.0

    # AMLA stores full ranked predictions, so we score how many of the
    # true top-k items appear anywhere in the produced ranking.
    relevant_targets = set(true_items[:k])
    retrieved_items = set(predicted_items)
    relevant = len(relevant_targets & retrieved_items)
    return float(relevant / min(k, len(true_items)))


def format_confidence(score):
    if score >= 0.8:
        return 'Very High'
    elif score >= 0.6:
        return 'High'
    elif score >= 0.4:
        return 'Medium'
    elif score >= 0.2:
        return 'Low'
    else:
        return 'Very Low'


def generate_explanation(meta_features, recommended_algorithm, similar_datasets=None):
    explanations = []
    
    n_instances = meta_features.get('l1_n_instances', 0)
    n_features = meta_features.get('l1_n_features', 0)
    
    if n_instances < 500:
        explanations.append(f"Small dataset ({int(n_instances)} samples) - simpler models may generalize better")
    elif n_instances > 10000:
        explanations.append(f"Large dataset ({int(n_instances)} samples) - complex ensemble methods can leverage the data")
    
    imbalance = meta_features.get('l1_imbalance_ratio', 1)
    if imbalance > 5:
        explanations.append(f"Class imbalance detected (ratio: {imbalance:.1f}) - ensemble methods recommended")
    
    landmark_dt = meta_features.get('l4_landmark_dt', 0)
    landmark_nb = meta_features.get('l4_landmark_nb', 0)
    
    if landmark_nb > landmark_dt:
        explanations.append("Naive Bayes outperforms Decision Stump - suggests simpler decision boundaries")
    else:
        explanations.append("Decision Tree outperforms Naive Bayes - suggests non-linear relationships")
    
    if similar_datasets:
        explanations.append(f"Based on analysis of {len(similar_datasets)} similar historical datasets")
    
    return explanations


def create_summary_report(analysis_result):
    report = {
        'title': 'AMLA Analysis Report',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'name': analysis_result.get('dataset_name', 'Unknown'),
            'instances': analysis_result.get('n_instances', 0),
            'features': analysis_result.get('n_features', 0)
        },
        'recommendation': analysis_result.get('algorithm_recommendation', {}),
        'feature_health': analysis_result.get('feature_analysis', {}).get('health_summary', {}),
        'feature_issues': analysis_result.get('feature_analysis', {}).get('recommendations', [])
    }
    
    return report


def validate_dataset(df, target_col):
    issues = []
    
    if df is None or df.empty:
        issues.append("Dataset is empty")
        return False, issues
    
    if target_col not in df.columns:
        issues.append(f"Target column '{target_col}' not found in dataset")
        return False, issues
    
    if df[target_col].nunique() < 2:
        issues.append("Target column must have at least 2 unique classes")
    
    if len(df) < 10:
        issues.append("Dataset too small for reliable analysis (minimum 10 samples)")
    
    return len(issues) == 0, issues


def get_recommended_preprocessing(issues):
    recommendations = {
        'high_missing': 'Consider removing columns with >30% missing values',
        'highly_skewed': 'Apply log or Box-Cox transformation',
        'near_zero_variance': 'Remove features with near-zero variance',
        'high_correlation': 'Remove one of highly correlated feature pairs',
        'class_imbalance': 'Use SMOTE, class weighting, or adjust decision threshold'
    }
    
    applicable = []
    for issue in issues:
        issue_type = issue.get('issue_type', '')
        if issue_type in recommendations:
            applicable.append(recommendations[issue_type])
    
    return applicable
