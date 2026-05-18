import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amla.characterizer import DatasetCharacterizer
from amla.feature_advisor import FeatureAugmentationAdvisor
from amla.mkb import MetaKnowledgeBase
from amla import utils


@pytest.fixture
def sample_dataframe():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                n_redundant=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def characterizer():
    return DatasetCharacterizer()


@pytest.fixture
def feature_advisor():
    return FeatureAugmentationAdvisor()


@pytest.fixture
def temp_mkb():
    import tempfile
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_mkb.db')
    mkb = MetaKnowledgeBase(db_path=db_path)
    yield mkb
    import shutil
    shutil.rmtree(temp_dir)


class TestDatasetCharacterizer:
    
    def test_initialization(self, characterizer):
        assert characterizer.dmfv == {}
        assert characterizer.dataset_hash is None
    
    def test_extract_all(self, characterizer, sample_dataframe):
        result = characterizer.extract_all(sample_dataframe, 'target')
        
        assert isinstance(result, dict)
        assert 'dataset_hash' in result
        assert result['dataset_hash'] is not None
        
        assert 'l1_n_instances' in result
        assert result['l1_n_instances'] == 200
        
        assert 'l1_n_features' in result
        assert result['l1_n_features'] == 10
    
    def test_layer1_simple_stats(self, characterizer, sample_dataframe):
        characterizer.extract_layer1_simple_stats(sample_dataframe, 'target')
        
        assert 'l1_n_instances' in characterizer.dmfv
        assert 'l1_n_features' in characterizer.dmfv
        assert 'l1_n_classes' in characterizer.dmfv
        assert 'l1_imbalance_ratio' in characterizer.dmfv
        assert characterizer.dmfv['l1_n_instances'] == 200
        assert characterizer.dmfv['l1_n_features'] == 10
        assert characterizer.dmfv['l1_n_classes'] == 2
    
    def test_layer2_distribution(self, characterizer, sample_dataframe):
        characterizer.extract_layer2_distribution_features(sample_dataframe, 'target')
        
        assert 'l2_mean_skewness' in characterizer.dmfv
        assert 'l2_mean_kurtosis' in characterizer.dmfv
        assert 'l2_mean_variance' in characterizer.dmfv
        assert 'l2_outlier_ratio' in characterizer.dmfv
    
    def test_layer3_info_theoretic(self, characterizer, sample_dataframe):
        characterizer.extract_layer3_info_theoretic(sample_dataframe, 'target')
        
        assert 'l3_mean_mutual_info' in characterizer.dmfv
        assert 'l3_class_entropy' in characterizer.dmfv
    
    def test_layer4_landmarkers(self, characterizer, sample_dataframe):
        characterizer.extract_layer4_landmarkers(sample_dataframe, 'target')
        
        assert 'l4_landmark_dt' in characterizer.dmfv
        assert 'l4_landmark_nb' in characterizer.dmfv
        assert 'l4_landmark_knn' in characterizer.dmfv
        assert 'l4_landmark_svm' in characterizer.dmfv
        
        assert 0 <= characterizer.dmfv['l4_landmark_dt'] <= 1
        assert 0 <= characterizer.dmfv['l4_landmark_nb'] <= 1
    
    def test_dmfv_dataframe(self, characterizer, sample_dataframe):
        characterizer.extract_all(sample_dataframe, 'target')
        dmfv_df = characterizer.get_dmfv_dataframe()
        
        assert isinstance(dmfv_df, pd.DataFrame)
        assert len(dmfv_df) == 1
        assert 'dataset_hash' not in dmfv_df.columns


class TestFeatureAugmentationAdvisor:
    
    def test_initialization(self, feature_advisor):
        assert feature_advisor.recommendations == []
    
    def test_analyse(self, feature_advisor, sample_dataframe):
        recommendations = feature_advisor.analyse(sample_dataframe, 'target')
        
        assert isinstance(recommendations, list)
    
    def test_health_summary(self, feature_advisor, sample_dataframe):
        feature_advisor.analyse(sample_dataframe, 'target')
        health = feature_advisor.get_health_summary()
        
        assert 'status' in health
        assert 'total_issues' in health
        assert health['status'] in ['healthy', 'warning', 'critical', 'minor']
    
    def test_near_zero_variance_detection(self, feature_advisor):
        df = pd.DataFrame({
            'constant_feature': [1.0] * 100,
            'varying_feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        recommendations = feature_advisor.analyse(df, 'target')
        weak_features = [r for r in recommendations if r['issue_type'] == 'weak_near_zero_variance']
        
        assert len(weak_features) >= 1
        assert any('constant_feature' in r['feature'] for r in weak_features)
    
    def test_skewed_feature_detection(self, feature_advisor):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            'normal_feature': rng.normal(size=200),
            'skewed_feature': rng.exponential(2, 200),
            'target': rng.integers(0, 2, 200)
        })
        
        recommendations = feature_advisor.analyse(df, 'target')
        skewed_features = [r for r in recommendations if r['issue_type'] == 'highly_skewed']
        
        assert len(skewed_features) >= 1
    
    def test_missing_value_detection(self, feature_advisor):
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': [1.0] * 50 + [np.nan] * 50,
            'target': np.random.randint(0, 2, 100)
        })
        
        recommendations = feature_advisor.analyse(df, 'target')
        missing_issues = [r for r in recommendations if 'missing' in r['issue_type']]
        
        assert len(missing_issues) >= 1


class TestMetaKnowledgeBase:
    
    def test_initialization(self, temp_mkb):
        assert temp_mkb.db_path.endswith('test_mkb.db')
        assert os.path.exists(temp_mkb.db_path)
    
    def test_add_experiment(self, temp_mkb):
        meta_features = {
            'l1_n_instances': 100,
            'l1_n_features': 10,
            'l4_landmark_dt': 0.8,
            'l4_landmark_nb': 0.75
        }
        
        algorithm_results = {
            'RandomForest': {'f1': 0.85, 'accuracy': 0.87, 'precision': 0.86, 'recall': 0.84}
        }
        
        exp_id = temp_mkb.add_experiment(
            dataset_hash='test_hash_123',
            dataset_name='test_dataset',
            domain='general',
            meta_features=meta_features,
            algorithm_results=algorithm_results
        )
        
        assert exp_id is not None
        assert exp_id > 0
    
    def test_get_experiment_count(self, temp_mkb):
        initial_count = temp_mkb.get_experiment_count()
        
        meta_features = {'l1_n_instances': 100, 'l1_n_features': 10}
        algorithm_results = {'RandomForest': {'f1': 0.85}}
        
        temp_mkb.add_experiment('hash1', 'dataset1', 'general', meta_features, algorithm_results)
        
        assert temp_mkb.get_experiment_count() == initial_count + 1
    
    def test_find_similar_experiments(self, temp_mkb):
        meta_features1 = {
            'l1_n_instances': 100, 'l1_n_features': 10, 'l2_mean_skewness': 0.5
        }
        meta_features2 = {
            'l1_n_instances': 105, 'l1_n_features': 12, 'l2_mean_skewness': 0.6
        }
        
        temp_mkb.add_experiment('hash1', 'dataset1', 'general', meta_features1, 
                               {'RandomForest': {'f1': 0.85}})
        temp_mkb.add_experiment('hash2', 'dataset2', 'general', meta_features2,
                               {'GradientBoosting': {'f1': 0.87}})
        
        similar = temp_mkb.find_similar_experiments(meta_features1, top_k=2)
        
        assert len(similar) <= 2
        assert all('similarity' in exp for exp in similar)


class TestUtils:
    
    def test_cosine_similarity(self):
        v1 = [1, 0, 0]
        v2 = [1, 0, 0]
        assert utils.cosine_similarity(v1, v2) == pytest.approx(1.0)
        
        v1 = [1, 0]
        v2 = [0, 1]
        assert utils.cosine_similarity(v1, v2) == pytest.approx(0.0)
    
    def test_precision_at_k(self):
        true_items = ['A', 'B', 'C']
        predicted = ['A', 'D', 'E', 'B', 'F']
        
        p1 = utils.precision_at_k(true_items, predicted, k=1)
        assert p1 == 1.0
        
        p3 = utils.precision_at_k(true_items, predicted, k=3)
        assert p3 == 2/3
    
    def test_ndcg_at_k(self):
        relevance = [3, 2, 3, 0, 1]
        ndcg = utils.ndcg_at_k(relevance, k=5)
        assert 0 <= ndcg <= 1
    
    def test_format_confidence(self):
        assert utils.format_confidence(0.9) == 'Very High'
        assert utils.format_confidence(0.7) == 'High'
        assert utils.format_confidence(0.5) == 'Medium'
        assert utils.format_confidence(0.3) == 'Low'
        assert utils.format_confidence(0.1) == 'Very Low'
    
    def test_validate_dataset(self, sample_dataframe):
        valid, issues = utils.validate_dataset(sample_dataframe, 'target')
        assert valid is True
        assert len(issues) == 0
        
        empty_df = pd.DataFrame()
        valid, issues = utils.validate_dataset(empty_df, 'target')
        assert valid is False
    
    def test_compute_dataset_hash(self, sample_dataframe):
        hash1 = utils.compute_dataset_hash(sample_dataframe)
        hash2 = utils.compute_dataset_hash(sample_dataframe)
        
        assert hash1 == hash2
        assert len(hash1) == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
