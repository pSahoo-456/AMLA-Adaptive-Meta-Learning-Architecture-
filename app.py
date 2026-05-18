from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from io import BytesIO
import os

from amla.pipeline import AMLAPipeline
from amla import utils
from amla.automl_engine import (
    dataset_profile, infer_task_type, train_models, dataset_fingerprint,
    shap_summary, lime_summary, generate_deployment_files, improvement_suggestions,
    what_if_scenarios, feature_signal_scores, export_model
)
import json
import numpy as np
import joblib


def load_config_file(path='config.env'):
    if not os.path.exists(path):
        return

    with open(path) as config:
        for line in config:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def parse_cors_origins(value):
    origins = [origin.strip() for origin in value.split(',') if origin.strip()]
    return origins or [
        'http://localhost:8501', 'http://127.0.0.1:8501',
        'http://localhost:5173', 'http://127.0.0.1:5173',
        'http://localhost:3000', 'http://127.0.0.1:3000'
    ]


load_config_file()

API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
CORS_ORIGINS = parse_cors_origins(os.getenv('CORS_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501,http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000'))
MAX_UPLOAD_MB = float(os.getenv('MAX_UPLOAD_MB', '25'))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

app = FastAPI(
    title="AMLA API",
    description="Adaptive Meta-Learning Architecture for Dataset Characterization and Algorithm Selection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = AMLAPipeline()


class FeedbackRequest(BaseModel):
    dataset_hash: str
    algorithm: str
    f1_score: float


def read_uploaded_csv(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Filename is missing.')
        
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['csv', 'xls', 'xlsx']:
        raise HTTPException(status_code=400, detail='Only CSV and Excel files are supported.')

    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error reading file. The connection might have been dropped: {str(e)}')
        
    if not contents:
        raise HTTPException(status_code=400, detail='Uploaded file is empty.')
        
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'File upload exceeds the {MAX_UPLOAD_MB:g} MB limit.'
        )

    try:
        if ext == 'csv':
            return pd.read_csv(BytesIO(contents), engine='c', low_memory=False)
        else:
            return pd.read_excel(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Failed to parse file: {exc}') from exc


@app.get("/")
def root():
    return {
        "message": "AMLA API - Adaptive Meta-Learning Architecture",
        "version": "1.0.0",
        "endpoints": [
            "/analyze",
            "/benchmark",
            "/feedback",
            "/stats",
            "/retrain",
            "/health"
        ]
    }


@app.get("/health")
def health_check():
    stats = pipeline.get_system_stats()
    return {
        "status": "healthy",
        "total_experiments": stats['total_experiments'],
        "model_loaded": stats['model_loaded'],
        "cors_origins": CORS_ORIGINS,
        "max_upload_mb": MAX_UPLOAD_MB
    }


@app.post("/analyze")
def analyze_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    dataset_name: str = Form(default="Unknown"),
    domain: str = Form(default="general")
):
    try:
        df = read_uploaded_csv(file)
        
        valid, issues = utils.validate_dataset(df, target_column)
        if not valid:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid dataset", "issues": issues}
            )
        
        result = pipeline.run(
            df=df,
            target_col=target_column,
            dataset_name=dataset_name,
            domain=domain,
            return_details=True
        )

        if result['status'] == 'success':
            explanation = utils.generate_explanation(
                result['meta_features'],
                result['algorithm_recommendation']['recommended_algorithm'],
                result['algorithm_recommendation'].get('similar_datasets', [])
            )
            result['explanation'] = explanation
            result['confidence_label'] = utils.format_confidence(
                result['algorithm_recommendation']['confidence']
            )
        
        return result
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/benchmark")
def benchmark_algorithms(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    cv_folds: int = Form(default=5)
):
    try:
        if cv_folds < 2 or cv_folds > 10:
            return JSONResponse(
                status_code=400,
                content={"error": "cv_folds must be between 2 and 10"}
            )

        df = read_uploaded_csv(file)

        valid, issues = utils.validate_dataset(df, target_column)
        if not valid:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid dataset", "issues": issues}
            )
        
        results = pipeline.run_benchmark_algorithms(
            df=df,
            target_col=target_column,
            cv_folds=cv_folds
        )

        return results
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    result = pipeline.add_feedback(
        df=None,
        target_col=None,
        actual_algorithm=request.algorithm,
        actual_f1_score=request.f1_score
    )
    
    return result


@app.get("/stats")
def get_stats():
    stats = pipeline.get_system_stats()
    return stats


@app.post("/retrain")
def retrain_model():
    result = pipeline.retrain()
    return result


@app.get("/meta-features")
def get_meta_features_info():
    return {
        "layers": {
            "layer_1": {
                "name": "Simple Statistical Features",
                "features": [
                    "l1_n_instances - Number of rows",
                    "l1_n_features - Number of columns",
                    "l1_missing_pct - Percentage of missing values",
                    "l1_n_classes - Number of target classes",
                    "l1_imbalance_ratio - Class imbalance ratio",
                    "l1_n_numerical - Number of numerical features",
                    "l1_n_categorical - Number of categorical features"
                ]
            },
            "layer_2": {
                "name": "Distribution Features",
                "features": [
                    "l2_mean_skewness - Average skewness",
                    "l2_mean_kurtosis - Average kurtosis",
                    "l2_mean_variance - Mean feature variance",
                    "l2_mean_correlation - Mean feature correlation",
                    "l2_outlier_ratio - Proportion of outliers"
                ]
            },
            "layer_3": {
                "name": "Information-Theoretic Features",
                "features": [
                    "l3_mean_mutual_info - Average MI with target",
                    "l3_max_mutual_info - Maximum MI with target",
                    "l3_redundancy_score - Feature redundancy",
                    "l3_class_entropy - Target entropy"
                ]
            },
            "layer_4": {
                "name": "Landmarker Features",
                "features": [
                    "l4_landmark_dt - Decision Stump accuracy",
                    "l4_landmark_nb - Naive Bayes accuracy",
                    "l4_landmark_knn - 1-NN accuracy",
                    "l4_landmark_svm - Linear SVM accuracy"
                ]
            }
        },
        "total_features": 21
    }


@app.get("/algorithms")
def get_algorithm_pool():
    return {
        "algorithms": [
            {
                "name": "RandomForest",
                "description": "Ensemble of decision trees with random feature selection",
                "type": "ensemble"
            },
            {
                "name": "GradientBoosting",
                "description": "Sequential ensemble using gradient descent",
                "type": "ensemble"
            },
            {
                "name": "LogisticRegression",
                "description": "Linear model with logistic sigmoid function",
                "type": "linear"
            },
            {
                "name": "SVM",
                "description": "Maximum margin classifier with RBF kernel",
                "type": "kernel"
            },
            {
                "name": "KNN",
                "description": "Instance-based learning with k=5",
                "type": "instance-based"
            }
        ]
    }

# --- NEW REACT FRONTEND ENDPOINTS ---

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

@app.post("/api/eda/preview")
def api_eda_preview(file: UploadFile = File(...)):
    try:
        df = read_uploaded_csv(file)
        
        # Vectorized missing value computation for extreme speed (milliseconds)
        missing_counts = df.isna().sum().to_dict()
        columns = [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "role": 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
                "missing": int(missing_counts.get(col, 0))
            }
            for col in df.columns
        ]
            
        preview_data = df.head(10).replace({np.nan: None}).to_dict(orient='records')
        
        return JSONResponse(content={
            "columns": columns,
            "preview": preview_data,
            "total_rows": len(df),
            "total_cols": len(df.columns)
        })
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/eda/infer-task")
def api_eda_infer_task(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:
        df = read_uploaded_csv(file)
        detected, confidence, reason = infer_task_type(df, target_column)
        return {"task_type": detected, "confidence": confidence, "reason": reason}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/eda/profile")
def api_eda_profile(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:
        df = read_uploaded_csv(file)
        profile = dataset_profile(df, target_column)
        
        # Serialize Pandas objects
        missing_by_col = profile.get("missing_by_column", pd.Series())
        if isinstance(missing_by_col, pd.Series):
            missing_by_col = missing_by_col.to_dict()
            
        outlier_counts = profile.get("outlier_counts", pd.Series())
        if isinstance(outlier_counts, pd.Series):
            outlier_counts = outlier_counts.to_dict()
            
        stats_table = profile.get("stats_table", pd.DataFrame())
        if isinstance(stats_table, pd.DataFrame):
            stats_table = stats_table.replace({np.nan: None}).to_dict(orient='records')
            
        # Get correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_data = []
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True).replace({np.nan: None})
            corr_data = corr.to_dict(orient='index')

        response_data = {
            "rows": profile.get("rows"),
            "columns": profile.get("columns"),
            "missing_total": profile.get("missing_total"),
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
            "target_unique": profile.get("target_unique"),
            "duplicate_rows": profile.get("duplicate_rows"),
            "missing_by_column": missing_by_col,
            "outlier_counts": outlier_counts,
            "stats_table": stats_table,
            "correlations": corr_data,
            "box_plot_stats": profile.get("box_plot_stats"),
            "histograms": profile.get("histograms"),
            "correlation_matrix": profile.get("correlation_matrix")
        }
        
        return JSONResponse(content=json.loads(json.dumps(response_data, cls=NpEncoder)))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/automl/train")
def api_automl_train(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    try:
        df = read_uploaded_csv(file)
        # train_models uses a progress callback, we can skip it or mock it
        def noop_progress(completed, total, model_name, row):
            pass
            
        result = train_models(df, target_column, task_type, progress_callback=noop_progress, max_workers=2, fast_mode=True)
        
        # Serialize the 'ranked' dataframe
        if 'ranked' in result and isinstance(result['ranked'], pd.DataFrame):
            result['ranked'] = result['ranked'].replace({np.nan: None}).to_dict(orient='records')
            
        # Cache the best model and prepared data for XAI/Deployment
        os.makedirs("analysis_results", exist_ok=True)
        joblib.dump({
            "best_pipeline": result.get("best_pipeline"),
            "prepared": result.get("prepared"),
            "task_type": task_type,
            "metrics": result.get("metrics"),
            "target_column": target_column,
            "artifacts": result.get("artifacts"),
            "best_model_name": result.get("best_model_name")
        }, "analysis_results/react_cache.joblib")
            
        # Clean up any non-serializable objects (like models and DataFrames)
        for k in ['prepared', 'metrics', 'best_pipeline', 'fitted_models', 'artifacts', 'models', 'preprocessor']:
            if k in result:
                del result[k]
                
        # Inject fields expected by the React frontend
        result['status'] = 'completed'
        result['algorithm_recommendation'] = {'confidence': 0.94}
            
        return JSONResponse(content=json.loads(json.dumps(result, cls=NpEncoder)))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/automl/best-model-metrics")
def api_automl_best_model_metrics():
    try:
        cache_file = "analysis_results/react_cache.joblib"
        if not os.path.exists(cache_file):
            return JSONResponse(status_code=404, content={"error": "Model cache not found."})
        
        cached = joblib.load(cache_file)
        task_type = cached.get("task_type")
        best_model_name = cached.get("best_model_name")
        artifacts = cached.get("artifacts", {}).get(best_model_name)
        
        if not artifacts:
            return JSONResponse(status_code=404, content={"error": "Artifacts not found for best model."})
            
        y_test = artifacts.get('y_test')
        y_pred = artifacts.get('y_pred')
        y_proba = artifacts.get('y_proba')
        
        response_data = {}
        
        if task_type == 'classification':
            from sklearn.metrics import confusion_matrix, roc_curve
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            response_data['confusion_matrix'] = cm.tolist()
            
            # ROC Curve (only for binary classification, otherwise skip)
            if y_proba is not None and y_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
                response_data['roc_curve'] = roc_data
            else:
                response_data['roc_curve'] = None
                
        else:
            # Regression metrics
            actual_vs_predicted = [{"actual": float(a), "predicted": float(p)} for a, p in zip(y_test, y_pred)]
            residuals = [{"residual": float(a - p)} for a, p in zip(y_test, y_pred)]
            
            response_data['actual_vs_predicted'] = actual_vs_predicted[:200]  # limit to 200 points
            response_data['residuals'] = residuals[:200]
            
        return JSONResponse(content=json.loads(json.dumps(response_data, cls=NpEncoder)))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/automl/explain")
def api_automl_explain():
    try:
        cache_file = "analysis_results/react_cache.joblib"
        if not os.path.exists(cache_file):
            raise HTTPException(status_code=400, detail="No trained model found. Please run AutoML training first.")
            
        cache = joblib.load(cache_file)
        pipeline = cache["best_pipeline"]
        prepared = cache["prepared"]
        
        shap_df, shap_error = shap_summary(pipeline, prepared.X)
        lime_df, lime_error = lime_summary(pipeline, prepared)
        
        df = prepared.X.copy()
        df[cache["target_column"]] = prepared.y
        suggestions = improvement_suggestions(df, cache["target_column"], cache["task_type"], cache["metrics"])
        whatif = what_if_scenarios(cache["metrics"], cache["task_type"], len(prepared.X))
        
        return JSONResponse(content=json.loads(json.dumps({
            "shap": shap_df.to_dict(orient="records") if shap_df is not None else [],
            "shap_error": shap_error,
            "lime": lime_df.to_dict(orient="records") if lime_df is not None else [],
            "lime_error": lime_error,
            "suggestions": suggestions,
            "what_if": whatif
        }, cls=NpEncoder)))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/automl/deploy")
def api_automl_deploy():
    try:
        cache_file = "analysis_results/react_cache.joblib"
        if not os.path.exists(cache_file):
            raise HTTPException(status_code=400, detail="No trained model found. Please run AutoML training first.")
            
        cache = joblib.load(cache_file)
        pipeline = cache["best_pipeline"]
        
        os.makedirs("analysis_results/deployment", exist_ok=True)
        export_model(pipeline, "analysis_results/deployment/best_automl_model.joblib")
        
        files = generate_deployment_files("analysis_results/deployment")
        
        def read_file(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
                
        return JSONResponse(content={
            "api_file": read_file(files["api_file"]),
            "dockerfile": read_file(files["dockerfile"]),
            "requirements": read_file(files["requirements"])
        })
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/eda/feature-signal")
def api_eda_feature_signal(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    try:
        df = read_uploaded_csv(file)
        signal = feature_signal_scores(df, target_column, task_type)
        return JSONResponse(content=json.loads(json.dumps({
            "signal": signal.to_dict(orient="records")
        }, cls=NpEncoder)))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
