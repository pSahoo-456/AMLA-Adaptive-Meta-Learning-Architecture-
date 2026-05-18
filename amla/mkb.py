import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime


class MetaKnowledgeBase:
    def __init__(self, db_path='data/mkb.db'):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_hash TEXT UNIQUE NOT NULL,
                dataset_name TEXT,
                domain TEXT,
                n_instances INTEGER,
                n_features INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                feature_name TEXT NOT NULL,
                feature_value REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, feature_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                algorithm_name TEXT NOT NULL,
                f1_score REAL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                cv_folds INTEGER DEFAULT 5,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_experiment_hash ON experiments(dataset_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_algorithm_name ON algorithm_results(algorithm_name)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_experiment(self, dataset_hash, dataset_name, domain, meta_features, algorithm_results):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id FROM experiments WHERE dataset_hash = ?
            ''', (dataset_hash,))
            existing = cursor.fetchone()
            
            if existing:
                experiment_id = existing[0]
                cursor.execute('''
                    UPDATE experiments 
                    SET dataset_name = ?, domain = ?, n_instances = ?, n_features = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (dataset_name, domain, 
                      int(meta_features.get('l1_n_instances', 0)),
                      int(meta_features.get('l1_n_features', 0)),
                      experiment_id))
                
                cursor.execute('DELETE FROM meta_features WHERE experiment_id = ?', (experiment_id,))
                cursor.execute('DELETE FROM algorithm_results WHERE experiment_id = ?', (experiment_id,))
            else:
                cursor.execute('''
                    INSERT INTO experiments (dataset_hash, dataset_name, domain, n_instances, n_features)
                    VALUES (?, ?, ?, ?, ?)
                ''', (dataset_hash, dataset_name, domain,
                      int(meta_features.get('l1_n_instances', 0)),
                      int(meta_features.get('l1_n_features', 0))))
                experiment_id = cursor.lastrowid
            
            for feature_name, feature_value in meta_features.items():
                if feature_name != 'dataset_hash':
                    cursor.execute('''
                        INSERT OR REPLACE INTO meta_features (experiment_id, feature_name, feature_value)
                        VALUES (?, ?, ?)
                    ''', (experiment_id, feature_name, float(feature_value) if feature_value is not None else None))
            
            for algo_name, metrics in algorithm_results.items():
                cursor.execute('''
                    INSERT INTO algorithm_results (experiment_id, algorithm_name, f1_score, accuracy, precision_score, recall_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (experiment_id, algo_name,
                      metrics.get('f1', None),
                      metrics.get('accuracy', None),
                      metrics.get('precision', None),
                      metrics.get('recall', None)))
            
            conn.commit()
            conn.close()
            return experiment_id
            
        except Exception as e:
            conn.rollback()
            conn.close()
            raise e
    
    def get_all_experiments(self):
        conn = self._get_connection()
        
        experiments_df = pd.read_sql_query('''
            SELECT * FROM experiments ORDER BY updated_at DESC
        ''', conn)
        
        if experiments_df.empty:
            conn.close()
            return experiments_df, pd.DataFrame(), pd.DataFrame()
        
        experiment_ids = experiments_df['id'].tolist()
        
        meta_features_df = pd.read_sql_query('''
            SELECT * FROM meta_features WHERE experiment_id IN ({})
        '''.format(','.join('?' * len(experiment_ids))), conn, params=experiment_ids)
        
        algorithm_results_df = pd.read_sql_query('''
            SELECT * FROM algorithm_results WHERE experiment_id IN ({})
        '''.format(','.join('?' * len(experiment_ids))), conn, params=experiment_ids)
        
        conn.close()
        
        return experiments_df, meta_features_df, algorithm_results_df
    
    def get_meta_features_matrix(self):
        experiments_df, meta_features_df, _ = self.get_all_experiments()
        
        if meta_features_df.empty:
            return pd.DataFrame()
        
        pivot_df = meta_features_df.pivot(index='experiment_id', columns='feature_name', values='feature_value')
        pivot_df = pivot_df.reset_index()
        pivot_df = pivot_df.merge(experiments_df[['id', 'dataset_hash']], left_on='experiment_id', right_on='id')
        
        return pivot_df
    
    def get_winner_algorithms(self):
        conn = self._get_connection()
        
        query = '''
            SELECT ar.experiment_id, ar.algorithm_name, ar.f1_score
            FROM algorithm_results ar
            INNER JOIN (
                SELECT experiment_id, MAX(f1_score) as max_f1
                FROM algorithm_results
                GROUP BY experiment_id
            ) best ON ar.experiment_id = best.experiment_id AND ar.f1_score = best.max_f1
        '''
        
        winners_df = pd.read_sql_query(query, conn)
        conn.close()
        
        return winners_df
    
    def find_similar_experiments(self, meta_features, top_k=5):
        experiments_df, meta_features_df, _ = self.get_all_experiments()
        
        if meta_features_df.empty:
            return []
        
        pivot_df = meta_features_df.pivot(index='experiment_id', columns='feature_name', values='feature_value')
        
        query_features = []
        for col in pivot_df.columns:
            if col in meta_features:
                query_features.append(meta_features[col])
            else:
                query_features.append(0)
        
        query_vector = np.array(query_features)
        dataset_vectors = pivot_df.values
        
        norms = np.linalg.norm(dataset_vectors, axis=1)
        norms[norms == 0] = 1
        dataset_vectors_normalized = dataset_vectors / norms[:, np.newaxis]
        
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_normalized = query_vector / query_norm
        else:
            query_normalized = query_vector
        
        similarities = np.dot(dataset_vectors_normalized, query_normalized)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            exp_id = pivot_df.index[idx]
            exp_info = experiments_df[experiments_df['id'] == exp_id].iloc[0] if len(experiments_df[experiments_df['id'] == exp_id]) > 0 else None
            if exp_info is not None:
                results.append({
                    'experiment_id': int(exp_id),
                    'dataset_hash': exp_info['dataset_hash'],
                    'dataset_name': exp_info['dataset_name'],
                    'domain': exp_info['domain'],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def get_experiment_count(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM experiments')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def export_to_csv(self, output_path):
        experiments_df, meta_features_df, algorithm_results_df = self.get_all_experiments()
        
        experiments_df.to_csv(os.path.join(output_path, 'experiments.csv'), index=False)
        meta_features_df.to_csv(os.path.join(output_path, 'meta_features.csv'), index=False)
        algorithm_results_df.to_csv(os.path.join(output_path, 'algorithm_results.csv'), index=False)
    
    def get_algorithm_statistics(self):
        conn = self._get_connection()
        results_df = pd.read_sql_query('''
            SELECT algorithm_name, f1_score, accuracy
            FROM algorithm_results
        ''', conn)
        conn.close()
        
        if results_df.empty:
            return pd.DataFrame(
                columns=[
                    'algorithm_name', 'experiment_count', 'avg_f1',
                    'avg_accuracy', 'max_f1', 'min_f1', 'std_f1'
                ]
            )

        stats_df = (
            results_df.groupby('algorithm_name')
            .agg(
                experiment_count=('algorithm_name', 'size'),
                avg_f1=('f1_score', 'mean'),
                avg_accuracy=('accuracy', 'mean'),
                max_f1=('f1_score', 'max'),
                min_f1=('f1_score', 'min'),
                std_f1=('f1_score', 'std')
            )
            .reset_index()
        )
        stats_df['std_f1'] = stats_df['std_f1'].fillna(0.0)
        return stats_df
