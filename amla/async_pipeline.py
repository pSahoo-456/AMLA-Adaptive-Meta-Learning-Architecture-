"""
Async Pipeline for responsive, real-time dataset analysis
Enables streaming results and progress tracking
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, AsyncGenerator
import time
import json

from .responsive_characterizer import ResponsiveDatasetCharacterizer
from .metalearner import MetaLearner
from .feature_advisor import FeatureAugmentationAdvisor
from .mkb import MetaKnowledgeBase


class ProgressContext:
    """Track and broadcast analysis progress"""
    
    def __init__(self):
        self.stages = [
            ("Loading dataset", 5),
            ("Characterizing features", 30),
            ("Analyzing feature quality", 60),
            ("Running algorithm selector", 85),
            ("Generating recommendations", 95),
            ("Complete", 100)
        ]
        self.current_stage = 0
        self.progress = 0
        self.start_time = time.time()
    
    def update(self, stage_idx, progress=None):
        """Update progress"""
        if stage_idx < len(self.stages):
            self.current_stage = stage_idx
            self.progress = progress if progress else self.stages[stage_idx][1]
    
    def get_state(self):
        """Get current progress state"""
        elapsed = time.time() - self.start_time
        return {
            "stage": self.stages[self.current_stage][0],
            "progress": self.progress,
            "elapsed_seconds": round(elapsed, 2),
            "stage_index": self.current_stage,
            "total_stages": len(self.stages)
        }


class AsyncAMLAPipeline:
    """Async-compatible AMLA pipeline for responsive analysis"""
    
    def __init__(self, model_path='models/metalearner.pkl', mkb_path='data/mkb.db'):
        self.characterizer = ResponsiveDatasetCharacterizer(cache_size=100, n_workers=4)
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
            'KNN',
            'XGBoost',
            'LightGBM'
        ]
        self.last_results = {}  # Cache last analysis
    
    async def run_async(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        dataset_name: str = 'Unknown',
        domain: str = 'general',
        return_details: bool = False
    ) -> Dict:
        """Async analysis pipeline with real-time progress"""
        
        progress = ProgressContext()
        result = {
            'status': 'success',
            'dataset_name': dataset_name,
            'domain': domain,
            'target_column': target_col,
            'n_instances': len(df),
            'n_features': len(df.columns) - 1,
            'progress': progress.get_state()
        }
        
        try:
            # Stage 1: Dataset loading (sync but fast)
            progress.update(0)
            result['progress'] = progress.get_state()
            await asyncio.sleep(0.001)  # Yield control
            
            # Stage 2: Characterization (using optimized sync with async wrapper)
            progress.update(1)
            result['progress'] = progress.get_state()
            
            self.meta_features = await asyncio.to_thread(
                self.characterizer.extract_all, df, target_col, True
            )
            result['meta_features'] = self.meta_features
            
            progress.update(2)
            result['progress'] = progress.get_state()
            await asyncio.sleep(0.001)
            
            # Stage 3: Feature analysis (async)
            feature_analysis = await asyncio.to_thread(
                self.feature_advisor.analyse, df, target_col
            )
            health_summary = await asyncio.to_thread(
                self.feature_advisor.get_health_summary
            )
            result['feature_analysis'] = {
                'recommendations': feature_analysis,
                'health_summary': health_summary
            }
            
            progress.update(3)
            result['progress'] = progress.get_state()
            await asyncio.sleep(0.001)
            
            # Stage 4: Algorithm selection (async)
            rf_recommendation = await asyncio.to_thread(
                self.metalearner.predict, self.meta_features
            )
            similarity_recommendation = await asyncio.to_thread(
                self.metalearner.get_similarity_recommendations, self.meta_features
            )
            
            combined_recommendation = self._combine_recommendations(
                rf_recommendation, 
                similarity_recommendation
            )
            result['algorithm_recommendation'] = combined_recommendation
            
            progress.update(4)
            result['progress'] = progress.get_state()
            
            # Final stage
            progress.update(5)
            result['progress'] = progress.get_state()
            
            if return_details:
                result['details'] = {
                    'dmfv': self.characterizer.get_dmfv_dict(),
                    'rf_prediction': rf_recommendation,
                    'similarity_prediction': similarity_recommendation
                }
            
            # Cache results
            self.last_results = result.copy()
            
            return result
            
        except Exception as e:
            import traceback
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            return result
    
    async def run_streaming(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        dataset_name: str = 'Unknown',
        domain: str = 'general'
    ) -> AsyncGenerator[Dict, None]:
        """
        Streaming analysis - yields progress updates in real-time
        Useful for Streamlit and WebSockets
        """
        
        progress = ProgressContext()
        
        try:
            # Yield initial state
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            await asyncio.sleep(0.001)
            
            # Stage 1: Dataset validation
            progress.update(0, 5)
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            await asyncio.sleep(0.001)
            
            # Stage 2: Characterization
            progress.update(1, 25)
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            
            self.meta_features = await asyncio.to_thread(
                self.characterizer.extract_all, df, target_col, True
            )
            
            yield {
                "type": "meta_features",
                "data": self.meta_features
            }
            await asyncio.sleep(0.001)
            
            # Stage 3: Feature analysis
            progress.update(2, 50)
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            
            feature_analysis = await asyncio.to_thread(
                self.feature_advisor.analyse, df, target_col
            )
            health_summary = await asyncio.to_thread(
                self.feature_advisor.get_health_summary
            )
            
            yield {
                "type": "feature_analysis",
                "data": {
                    "recommendations": feature_analysis,
                    "health_summary": health_summary
                }
            }
            await asyncio.sleep(0.001)
            
            # Stage 4: Algorithm selection
            progress.update(3, 75)
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            
            rf_recommendation = await asyncio.to_thread(
                self.metalearner.predict, self.meta_features
            )
            similarity_recommendation = await asyncio.to_thread(
                self.metalearner.get_similarity_recommendations, self.meta_features
            )
            
            combined_recommendation = self._combine_recommendations(
                rf_recommendation,
                similarity_recommendation
            )
            
            yield {
                "type": "algorithm_recommendation",
                "data": combined_recommendation
            }
            await asyncio.sleep(0.001)
            
            # Final completion
            progress.update(5)
            yield {
                "type": "progress",
                "data": progress.get_state()
            }
            
            yield {
                "type": "complete",
                "data": {
                    "status": "success",
                    "dataset_name": dataset_name,
                    "domain": domain,
                    "target_column": target_col,
                    "n_instances": len(df),
                    "n_features": len(df.columns) - 1,
                    "meta_features": self.meta_features,
                    "feature_analysis": {
                        "recommendations": feature_analysis,
                        "health_summary": health_summary
                    },
                    "algorithm_recommendation": combined_recommendation
                }
            }
            
        except Exception as e:
            import traceback
            yield {
                "type": "error",
                "data": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
    
    def _combine_recommendations(self, rf_rec, sim_rec):
        """Combine RF and similarity-based recommendations"""
        rf_algorithm = rf_rec.get('ranked_algorithms', [])
        sim_performance = {
            a['algorithm']: a['avg_f1'] 
            for a in sim_rec.get('algorithm_performance', [])
        }
        
        combined_scores = {}
        
        for i, algo_dict in enumerate(rf_algorithm):
            algo = algo_dict['algorithm']
            rf_weight = 1.0 / (i + 1)
            sim_f1 = sim_performance.get(algo, 0.5)
            sim_weight = sim_f1
            
            combined_scores[algo] = (
                rf_weight * algo_dict.get('probability', 0) + 
                sim_weight * 0.5
            )
        
        sorted_algos = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        confidence = float(rf_rec.get('confidence', sorted_algos[0][1] if sorted_algos else 0.0))

        ranked_algorithms = []
        for rank, (algo, score) in enumerate(sorted_algos, start=1):
            rf_probability = next(
                (item.get('probability', 0.0) for item in rf_algorithm if item.get('algorithm') == algo),
                0.0
            )
            ranked_algorithms.append({
                'rank': rank,
                'algorithm': algo,
                'combined_score': float(score),
                'score': float(score),
                'rf_probability': float(rf_probability),
                'similarity_f1': float(sim_performance.get(algo, 0.0))
            })
        
        return {
            'recommended_algorithm': sorted_algos[0][0] if sorted_algos else 'RandomForest',
            'confidence': confidence,
            'confidence_score': confidence,
            'method': 'combined_async',
            'ranked_algorithms': ranked_algorithms,
            'similar_datasets': sim_rec.get('similar_datasets', []),
            'explanation': f"Recommended {sorted_algos[0][0]} based on meta-features and historical performance on similar datasets." if sorted_algos else "Defaulted to RandomForest due to limited training history."
        }
    
    def get_last_results(self):
        """Get cached last analysis results"""
        return self.last_results

    def record_feedback(self, algorithm: str, f1_score: float, dataset_hash: str = None):
        """Persist feedback for the most recent analysis result."""
        last_meta_features = self.last_results.get('meta_features')
        if not last_meta_features:
            return {
                'status': 'error',
                'message': 'No recent analysis context available. Run an analysis before submitting feedback.'
            }

        cached_hash = last_meta_features.get('dataset_hash')
        if dataset_hash and cached_hash and dataset_hash != cached_hash:
            return {
                'status': 'error',
                'message': 'The provided dataset hash does not match the most recent analysis.'
            }

        self.mkb.add_experiment(
            dataset_hash=cached_hash or dataset_hash,
            dataset_name=self.last_results.get('dataset_name', 'user_dataset'),
            domain=self.last_results.get('domain', 'user'),
            meta_features=last_meta_features,
            algorithm_results={
                algorithm: {
                    'f1': f1_score
                }
            }
        )

        return {
            'status': 'success',
            'message': f'Recorded feedback for {algorithm} with F1={f1_score:.4f}.'
        }
    
    def get_system_stats(self):
        """Get system statistics"""
        return {
            'total_experiments': self.mkb.get_experiment_count() if self.mkb else 0,
            'model_loaded': self.metalearner.model is not None,
            'cache_size': len(self.characterizer.cache),
            'cached_datasets': len(self.characterizer.cache)
        }
