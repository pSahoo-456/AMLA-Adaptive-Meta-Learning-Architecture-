import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

# Lazy import SHAP and LIME inside their summary functions to avoid heavy imports at module load
shap = None
LimeTabularExplainer = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors
except Exception:
    SimpleDocTemplate = None


RANDOM_STATE = 42


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list
    categorical_features: list
    target_encoder: LabelEncoder | None = None


def read_tabular_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith('.csv'):
        return pd.read_csv(BytesIO(data))
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(BytesIO(data))
    raise ValueError('Only CSV and Excel files are supported.')


def infer_task_type(df, target_col):
    y = df[target_col].dropna()
    unique_count = y.nunique()
    unique_ratio = unique_count / max(len(y), 1)

    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return 'classification', 0.95, 'Target is categorical.'
    if pd.api.types.is_integer_dtype(y) and (unique_count <= 20 or unique_ratio <= 0.1):
        return 'classification', 0.8, 'Integer target has few unique values.'
    if pd.api.types.is_numeric_dtype(y) and unique_count > 20 and unique_ratio > 0.1:
        return 'regression', 0.85, 'Numeric target has many continuous values.'
    return 'classification', 0.55, 'Task is ambiguous; review the target manually.'


def prepare_data(df, target_col, task_type):
    clean = df.dropna(subset=[target_col]).copy()
    X = clean.drop(columns=[target_col])
    y = clean[target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    target_encoder = None

    if task_type == 'classification':
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index, name=target_col)
    else:
        y = pd.to_numeric(y, errors='coerce')
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

    return PreparedData(X, y, numeric_features, categorical_features, target_encoder)


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
    ])


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        names.extend(encoder.get_feature_names_out(categorical_features).tolist())
    return names


def model_catalog(task_type, n_classes=None, fast_mode=True):
    """Model catalog with optimized hyperparameters for faster training.
    
    Args:
        task_type: 'classification' or 'regression'
        n_classes: number of classes for classification
        fast_mode: if True, use reduced hyperparameters for speed (default)
    """
    if fast_mode:
        n_estimators = 80  # reduced from 200
        nn_max_iter = 80   # reduced from 500
        nn_hidden = (32, 16)  # smaller architecture
        max_depth = 4
        learning_rate = 0.05
    else:
        n_estimators = 200
        nn_max_iter = 200
        nn_hidden = (64, 32)
        max_depth = 5
        learning_rate = 0.05
    
    if task_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=nn_hidden, max_iter=nn_max_iter, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE, n_iter_no_change=10),
        }
        # Skip SVM on fast mode for large datasets (can be very slow)
        if not fast_mode:
            models['SVM'] = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE, gamma='scale')
        
        if XGBClassifier is not None:
            models['XGBoost'] = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric='logloss' if n_classes == 2 else 'mlogloss',
                random_state=RANDOM_STATE,
                tree_method='hist' if fast_mode else 'auto',
            )
        if LGBMClassifier is not None:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_STATE,
                verbose=-1,
                num_leaves=31 if not fast_mode else 25,
            )
        return models

    # Regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_STATE, max_depth=max_depth),
        'Neural Network Regressor': MLPRegressor(hidden_layer_sizes=nn_hidden, max_iter=nn_max_iter, early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE, n_iter_no_change=10),
    }
    if not fast_mode:
        models['SVR'] = SVR(kernel='rbf', gamma='scale')
    
    if XGBRegressor is not None:
        models['XGBoost Regressor'] = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=RANDOM_STATE,
            tree_method='hist' if fast_mode else 'auto',
        )
    if LGBMRegressor is not None:
        models['LightGBM Regressor'] = LGBMRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            verbose=-1,
            num_leaves=31 if not fast_mode else 25,
        )
    return models


def train_models(df, target_col, task_type, test_size=0.2, progress_callback=None, max_workers=2, fast_mode=True):
    """Train multiple AutoML models with optimized speed.
    
    Args:
        df: input DataFrame
        target_col: target column name
        task_type: 'classification' or 'regression'
        test_size: train-test split ratio
        progress_callback: callback function for progress updates
        max_workers: max parallel workers (reduced from 3 to 2 for less contention)
        fast_mode: if True, use reduced hyperparameters for faster training
    """
    prepared = prepare_data(df, target_col, task_type)
    if len(prepared.X) < 10:
        raise ValueError('At least 10 valid rows are required for AutoML training.')

    stratify = prepared.y if task_type == 'classification' and prepared.y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # Create preprocessor once and reuse
    preprocessor = make_preprocessor(prepared.numeric_features, prepared.categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = model_catalog(task_type, n_classes=prepared.y.nunique(), fast_mode=fast_mode)
    rows = []
    fitted = {}
    artifacts = {}

    started_at = time.perf_counter()

    def fit_one(name, estimator):
        model_started = time.perf_counter()
        try:
            # Use pre-processed data to avoid redundant preprocessing
            pipeline = clone(estimator)
            pipeline.fit(X_train_processed, y_train)
            y_pred = pipeline.predict(X_test_processed)
            y_proba = pipeline.predict_proba(X_test_processed) if hasattr(pipeline, 'predict_proba') else None
            row = evaluate_model(task_type, name, y_test, y_pred, y_proba)
            row['runtime_seconds'] = round(time.perf_counter() - model_started, 3)
            
            # Wrap for consistency
            full_pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('model', pipeline),
            ])
            
            return name, row, full_pipeline, {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'X_test': X_test,
            }
        except Exception as exc:
            return name, {
                'model': name,
                'status': f'failed: {str(exc)[:200]}',
                'runtime_seconds': round(time.perf_counter() - model_started, 3),
            }, None, None

    # Use fewer workers to reduce thread contention
    actual_workers = max(1, min(max_workers, len(models)))
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_map = {executor.submit(fit_one, name, estimator): name for name, estimator in models.items()}
        completed = 0
        total = len(future_map)
        for future in as_completed(future_map):
            name, row, pipeline, artifact = future.result()
            rows.append(row)
            if pipeline is not None:
                fitted[name] = pipeline
                artifacts[name] = artifact
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, name, row)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError('No model completed successfully.')

    ok_metrics = metrics[metrics['status'].eq('ok')]
    if ok_metrics.empty:
        error_msgs = metrics['status'].tolist()
        unique_errors = list(set([e.replace('failed: ', '') for e in error_msgs]))
        err_detail = unique_errors[0] if unique_errors else 'Unknown error'
        raise ValueError(f'All models failed to train. Common error: {err_detail}')

    sort_metric = 'f1' if task_type == 'classification' else 'r2'
    ranked = ok_metrics.sort_values(sort_metric, ascending=False)

    best_name = ranked.iloc[0]['model']
    return {
        'prepared': prepared,
        'metrics': metrics,
        'ranked': ranked,
        'best_model_name': best_name,
        'best_pipeline': fitted[best_name],
        'fitted_models': fitted,
        'artifacts': artifacts,
        'sort_metric': sort_metric,
        'total_runtime_seconds': round(time.perf_counter() - started_at, 3),
    }


def evaluate_model(task_type, model_name, y_true, y_pred, y_proba):
    if task_type == 'classification':
        row = {
            'model': model_name,
            'status': 'ok',
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f2': fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0),
        }
        row['roc_auc'] = safe_roc_auc(y_true, y_proba)
        return row

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        'model': model_name,
        'status': 'ok',
        'rmse': rmse,
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }


def safe_roc_auc(y_true, y_proba):
    if y_proba is None:
        return np.nan
    try:
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        return np.nan


def dataset_profile(df, target_col):
    numeric = df.select_dtypes(include=[np.number])
    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_rows = int(df.duplicated().sum())
    outlier_counts = {}
    box_plot_stats = {}
    histograms = {}

    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((series < low) | (series > high)).sum())
        
        box_plot_stats[col] = {
            'min': float(series.min()),
            'q1': float(q1),
            'median': float(series.median()),
            'q3': float(q3),
            'max': float(series.max()),
            'outliers': series[(series < low) | (series > high)].sample(min(50, outlier_counts[col])).tolist() if outlier_counts[col] > 0 else []
        }
        
        counts, bins = np.histogram(series, bins=min(20, max(1, series.nunique())))
        histograms[col] = {
            'counts': counts.tolist(),
            'bins': bins.tolist()
        }

    corr_matrix = numeric.corr().fillna(0).to_dict()

    stats_table = numeric.agg(['mean', 'median', 'std', 'skew', 'kurt']).T.reset_index().rename(columns={'index': 'feature'})

    return {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(numeric.columns),
        'categorical_columns': len(df.columns) - len(numeric.columns),
        'missing_total': int(df.isna().sum().sum()),
        'missing_by_column': missing[missing > 0],
        'duplicate_rows': duplicate_rows,
        'outlier_counts': pd.Series(outlier_counts).sort_values(ascending=False),
        'box_plot_stats': box_plot_stats,
        'histograms': histograms,
        'correlation_matrix': corr_matrix,
        'stats_table': stats_table,
        'target_unique': int(df[target_col].nunique()) if target_col in df else 0,
    }


def feature_signal_scores(df, target_col, task_type):
    prepared = prepare_data(df, target_col, task_type)
    preprocessor = make_preprocessor(prepared.numeric_features, prepared.categorical_features)
    X_transformed = preprocessor.fit_transform(prepared.X)
    feature_names = get_feature_names(preprocessor, prepared.numeric_features, prepared.categorical_features)

    try:
        if task_type == 'classification':
            scores = mutual_info_classif(X_transformed, prepared.y, random_state=RANDOM_STATE)
        else:
            scores = mutual_info_regression(X_transformed, prepared.y, random_state=RANDOM_STATE)
    except Exception:
        scores = np.zeros(len(feature_names))

    return pd.DataFrame({'feature': feature_names, 'score': scores}).sort_values('score', ascending=False)


def feature_importance(best_pipeline):
    preprocessor = best_pipeline.named_steps['preprocess']
    model = best_pipeline.named_steps['model']
    feature_names = get_feature_names(
        preprocessor,
        preprocessor.transformers_[0][2],
        preprocessor.transformers_[1][2],
    )

    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        values = np.ravel(np.abs(model.coef_))
        if len(values) != len(feature_names):
            values = values[:len(feature_names)]
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    return pd.DataFrame({'feature': feature_names[:len(values)], 'importance': values}).sort_values('importance', ascending=False)


def shap_summary(best_pipeline, X_sample, max_rows=80):
    # Lazy-import shap to avoid slowing app startup
    try:
        import shap as _shap
    except Exception:
        return None, 'SHAP is not installed.'
    global shap
    shap = _shap
    model = best_pipeline.named_steps['model']
    if not hasattr(model, 'feature_importances_'):
        return None, 'SHAP summary is available for tree-based models in this dashboard.'

    try:
        preprocessor = best_pipeline.named_steps['preprocess']
        X_small = X_sample.head(max_rows)
        X_transformed = preprocessor.transform(X_small)
        feature_names = get_feature_names(
            preprocessor,
            preprocessor.transformers_[0][2],
            preprocessor.transformers_[1][2],
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
        if values.ndim == 3:
            values = values[:, :, 0]
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame({'feature': feature_names, 'shap_importance': importance}).sort_values('shap_importance', ascending=False), None
    except Exception as exc:
        return None, f'SHAP could not be computed: {exc}'


def lime_summary(best_pipeline, prepared, row_index=0, max_features=12):
    # Lazy-import LIME to avoid slowing app startup
    try:
        from lime.lime_tabular import LimeTabularExplainer as _LimeTabularExplainer
    except Exception:
        return None, 'LIME is not installed.'
    try:
        preprocessor = best_pipeline.named_steps['preprocess']
        model = best_pipeline.named_steps['model']
        X_transformed = preprocessor.transform(prepared.X)
        feature_names = get_feature_names(
            preprocessor,
            preprocessor.transformers_[0][2],
            preprocessor.transformers_[1][2],
        )
        mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
        explainer = _LimeTabularExplainer(
            X_transformed,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )
        instance = X_transformed[min(row_index, len(X_transformed) - 1)]
        predict_fn = model.predict_proba if mode == 'classification' else model.predict
        explanation = explainer.explain_instance(instance, predict_fn, num_features=max_features)
        return pd.DataFrame(explanation.as_list(), columns=['feature_rule', 'contribution']), None
    except Exception as exc:
        return None, f'LIME could not be computed: {exc}'


def stability_curves(best_pipeline, X, y, task_type):
    scoring = 'f1_weighted' if task_type == 'classification' else 'r2'
    cv = min(5, max(2, len(y) // 5))
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.3, 1.0, 5),
        n_jobs=-1,
    )

    learning = pd.DataFrame({
        'train_size': train_sizes,
        'train_score': train_scores.mean(axis=1),
        'validation_score': test_scores.mean(axis=1),
    })

    validation = None
    if 'Random Forest' in best_pipeline.named_steps['model'].__class__.__name__:
        try:
            param_range = [3, 5, 8, 12, None]
            train_v, test_v = validation_curve(
                best_pipeline,
                X,
                y,
                param_name='model__max_depth',
                param_range=param_range,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            validation = pd.DataFrame({
                'max_depth': [str(v) for v in param_range],
                'train_score': train_v.mean(axis=1),
                'validation_score': test_v.mean(axis=1),
            })
        except Exception:
            validation = None

    return learning, validation


def improvement_suggestions(df, target_col, task_type, metrics_df):
    suggestions = []
    missing_pct = df.isna().mean().sort_values(ascending=False)
    if (missing_pct > 0).any():
        cols = ', '.join(missing_pct[missing_pct > 0].head(5).index)
        suggestions.append(f'Clean missing values in: {cols}. Use median/mode imputation or model-aware imputation.')

    numeric = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    skewed = [col for col in numeric.columns if abs(stats.skew(numeric[col].dropna())) > 1.5 and numeric[col].dropna().nunique() > 3]
    if skewed:
        suggestions.append(f'Apply log/Yeo-Johnson transforms to skewed features: {", ".join(skewed[:5])}.')

    if task_type == 'classification':
        class_counts = df[target_col].value_counts()
        if len(class_counts) > 1 and class_counts.max() / class_counts.min() > 3:
            suggestions.append('Class imbalance detected. Try class weights, SMOTE, or threshold tuning.')
        best = metrics_df[metrics_df['status'].eq('ok')].sort_values('f1', ascending=False).iloc[0]
        if best['f1'] < 0.75:
            suggestions.append('F1 is below 0.75. Add domain features, remove leakage/noise, tune hyperparameters, and check labeling quality.')
        else:
            suggestions.append('Performance is strong. Run repeated cross-validation and calibration before deployment.')
    else:
        best = metrics_df[metrics_df['status'].eq('ok')].sort_values('r2', ascending=False).iloc[0]
        if best['r2'] < 0.6:
            suggestions.append('R2 is modest. Add interaction features, transform the target, inspect outliers, and tune tree depth/learning rate.')
        else:
            suggestions.append('Regression fit is promising. Validate residual assumptions and test on a holdout dataset.')

    suggestions.append('Export the best model only after checking learning curves, feature importance, and data drift across versions.')
    return suggestions


def what_if_scenarios(metrics_df, task_type, rows):
    ok = metrics_df[metrics_df['status'].eq('ok')]
    if ok.empty:
        return []
    if task_type == 'classification':
        best = float(ok.sort_values('f1', ascending=False).iloc[0]['f1'])
        headroom = max(0.0, 1.0 - best)
        gain_20 = min(0.08, headroom * 0.25)
        gain_clean = min(0.06, headroom * 0.2)
        return [
            f'If you add about 20% more representative labeled rows ({int(rows * 0.2)} rows), F1 may improve by roughly {gain_20:.1%}.',
            f'If missing values/outliers are cleaned before training, F1 may improve by roughly {gain_clean:.1%}.',
        ]
    best = float(ok.sort_values('r2', ascending=False).iloc[0]['r2'])
    headroom = max(0.0, 1.0 - best)
    gain_20 = min(0.10, headroom * 0.25)
    return [
        f'If you add about 20% more representative rows ({int(rows * 0.2)} rows), R2 may improve by roughly {gain_20:.1%}.',
        'If residual outliers are investigated and target transformations are tested, regression stability may improve.',
    ]


def auto_fix_dataset(df, target_col, task_type, aggressive=False):
    """Apply automatic data cleaning and improvements based on quality issues detected."""
    df_clean = df.copy()
    changes_log = []

    # 1. Handle missing values: impute numeric with median, categorical with mode
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isna().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            changes_log.append(f'✓ Filled {col} (numeric) missing values with median.')

    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if col != target_col and df_clean[col].isna().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            changes_log.append(f'✓ Filled {col} (categorical) missing values with mode.')

    # 2. Remove duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        changes_log.append(f'✓ Removed {dup_count} duplicate rows.')

    # 3. Handle skewness (if aggressive): log transform highly skewed numeric features
    if aggressive:
        for col in numeric_cols:
            if col != target_col and df_clean[col].min() > 0:
                skewness = abs(stats.skew(df_clean[col].dropna()))
                if skewness > 1.5 and df_clean[col].nunique() > 10:
                    df_clean[col] = np.log1p(df_clean[col])
                    changes_log.append(f'✓ Applied log transform to skewed feature: {col}.')

    # 4. Handle outliers (if aggressive): clip to IQR bounds
    if aggressive:
        for col in numeric_cols:
            if col != target_col:
                q1, q3 = df_clean[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_count = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
                if outlier_count > 0:
                    df_clean[col] = df_clean[col].clip(lower, upper)
                    changes_log.append(f'✓ Clipped {outlier_count} outliers in {col} to IQR bounds.')

    # 5. Class imbalance handling (classification only, if aggressive)
    if aggressive and task_type == 'classification':
        class_counts = df_clean[target_col].value_counts()
        if len(class_counts) > 1:
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 3:
                changes_log.append(f'⚠ Class imbalance detected (ratio: {imbalance_ratio:.1f}). Consider SMOTE or class weights in model training.')

    return df_clean, changes_log


def dataset_fingerprint(df):
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_total': int(df.isna().sum().sum()),
        'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
        'duplicate_rows': int(df.duplicated().sum()),
    }


def export_model(pipeline, path):
    joblib.dump(pipeline, path)
    return path


def dataframe_download(df):
    csv = df.to_csv(index=False).encode('utf-8')
    return base64.b64encode(csv).decode('utf-8')


def generate_html_report(path, dataset_name, target_col, task_type, profile, metrics_df, suggestions):
    html = f"""
    <html>
    <head><title>AMLA AutoML Report</title></head>
    <body>
    <h1>AMLA AutoML Report</h1>
    <h2>Dataset</h2>
    <p><b>Name:</b> {dataset_name}</p>
    <p><b>Target:</b> {target_col}</p>
    <p><b>Task:</b> {task_type}</p>
    <p><b>Rows:</b> {profile['rows']} | <b>Columns:</b> {profile['columns']} | <b>Missing Cells:</b> {profile['missing_total']}</p>
    <h2>Model Metrics</h2>
    {metrics_df.to_html(index=False)}
    <h2>Suggestions</h2>
    <ul>{''.join(f'<li>{item}</li>' for item in suggestions)}</ul>
    </body>
    </html>
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return str(output_path)


def generate_pdf_report(path, dataset_name, target_col, task_type, profile, metrics_df, suggestions):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if SimpleDocTemplate is None:
        output_path.write_text(
            f"AMLA AutoML Report\nDataset: {dataset_name}\nTarget: {target_col}\nTask: {task_type}\n\n"
            + metrics_df.to_string(index=False)
            + "\n\nSuggestions\n"
            + "\n".join(f"- {item}" for item in suggestions),
            encoding='utf-8',
        )
        return str(output_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    elements = [
        Paragraph('AMLA AutoML Report', styles['Title']),
        Spacer(1, 12),
        Paragraph(f'Dataset: {dataset_name}', styles['Normal']),
        Paragraph(f'Target: {target_col}', styles['Normal']),
        Paragraph(f'Task: {task_type}', styles['Normal']),
        Paragraph(f"Rows: {profile['rows']} | Columns: {profile['columns']} | Missing Cells: {profile['missing_total']}", styles['Normal']),
        Spacer(1, 12),
        Paragraph('Model Metrics', styles['Heading2']),
    ]
    table_data = [metrics_df.columns.tolist()] + metrics_df.fillna('').round(4).astype(str).values.tolist()
    table = Table(table_data[:12], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
    ]))
    elements.extend([table, Spacer(1, 12), Paragraph('Improvement Suggestions', styles['Heading2'])])
    for item in suggestions:
        elements.append(Paragraph(f'- {item}', styles['Normal']))
    doc.build(elements)
    return str(output_path)


def generate_deployment_files(output_dir, model_filename='best_automl_model.joblib'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api_file = output / 'serve_model_fastapi.py'
    docker_file = output / 'Dockerfile'
    requirements_file = output / 'requirements-serving.txt'

    api_file.write_text(f"""from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title='AMLA Model Serving API')
model = joblib.load('{model_filename}')

class PredictionRequest(BaseModel):
    rows: list[dict]

@app.get('/health')
def health():
    return {{'status': 'healthy', 'model_loaded': model is not None}}

@app.post('/predict')
def predict(request: PredictionRequest):
    df = pd.DataFrame(request.rows)
    predictions = model.predict(df)
    return {{'predictions': predictions.tolist()}}
""", encoding='utf-8')

    docker_file.write_text("""FROM python:3.11-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "serve_model_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
""", encoding='utf-8')

    requirements_file.write_text("""fastapi
uvicorn
pandas
scikit-learn
joblib
xgboost
lightgbm
""", encoding='utf-8')

    return {
        'api_file': str(api_file),
        'dockerfile': str(docker_file),
        'requirements': str(requirements_file),
    }
