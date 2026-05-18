import os
import json
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

from amla.automl_engine import (
    dataset_fingerprint,
    dataset_profile,
    export_model,
    feature_importance,
    feature_signal_scores,
    generate_deployment_files,
    generate_html_report,
    generate_pdf_report,
    improvement_suggestions,
    infer_task_type,
    lime_summary,
    read_tabular_file,
    shap_summary,
    stability_curves,
    make_preprocessor,
    prepare_data,
    train_models,
    what_if_scenarios,
)


API_URL = os.getenv('AMLA_API_URL', 'http://localhost:8000')
PREFERENCES_PATH = Path('analysis_results/ui_preferences.json')
PLOT_CONFIG = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'amla_chart',
        'height': 720,
        'width': 1280,
        'scale': 2,
    },
}

# ===== CACHING DECORATORS FOR PERFORMANCE =====
@st.cache_data
def load_data(uploaded_file):
    """Cache file loading to avoid re-reading CSV/Excel multiple times."""
    return read_tabular_file(uploaded_file)

@st.cache_data
def cache_dataset_profile(_df, target_col):
    """Cache expensive dataset profiling operations."""
    return dataset_profile(_df, target_col)

@st.cache_data(ttl=3600)
def cache_correlation_matrix(_df, numeric_cols):
    """Cache correlation matrix computation (1 hour TTL)."""
    return _df[numeric_cols].corr(numeric_only=True)

@st.cache_data(ttl=3600)
def cache_feature_importance(importance_df):
    """Cache feature importance computation."""
    return importance_df.head(20)

# ===== ENHANCED CSS FOR UI FIXES =====
ENHANCED_CSS = """
<style>
    /* Fix text truncation in metrics */
    [data-testid="stMetricLabel"] {
        word-break: break-word;
        white-space: normal;
        max-width: 100%;
    }
    
    [data-testid="stMetricValue"] {
        word-break: break-word;
        white-space: normal;
    }
    
    /* Improve dataframe display */
    [data-testid="stDataFrameResizable"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
    
    /* Fix card truncation */
    .card-shell, .card-body {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        line-height: 1.6;
    }
    
    /* Better expander styling */
    [data-testid="stExpander"] {
        width: 100%;
    }
    
    /* Responsive improvements */
    @media (max-width: 1024px) {
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
"""

st.set_page_config(
    page_title='AMLA AutoML Platform',
    page_icon='AMLA',
    layout='wide',
    initial_sidebar_state='expanded',
)


def init_state():
    stored_theme = 'Dark'
    if PREFERENCES_PATH.exists():
        try:
            stored_theme = json.loads(PREFERENCES_PATH.read_text()).get('theme', 'Dark')
        except Exception:
            stored_theme = 'Dark'

    defaults = {
        'df': None,
        'target_col': None,
        'task_type': None,
        'task_confidence': None,
        'task_reason': None,
        'profile': None,
        'automl': None,
        'history': [],
        'theme': stored_theme,
        'dataset_type': 'tabular',  # 'tabular' or 'image'
        'image_dataset': None,
        'image_metadata': None,
        'n_classes': 2,
        'df_original': None,
        'nav_page': 'Home',
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_theme():
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(json.dumps({'theme': st.session_state.theme}), encoding='utf-8')


def chart(fig, **kwargs):
    st.plotly_chart(fig, width='stretch', config=PLOT_CONFIG, **kwargs)


def notebook_style_benchmarks(df, target_col):
    """Run notebook-style classifier variants for easier result parity with EDA notebooks."""
    task_type = 'classification'
    prepared = prepare_data(df, target_col, task_type)
    if prepared.y.nunique() < 2:
        return pd.DataFrame(), 'Notebook benchmark requires at least two target classes.'

    stratify = prepared.y if prepared.y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = make_preprocessor(prepared.numeric_features, prepared.categorical_features)

    results = []
    lr = LogisticRegression(max_iter=700, class_weight='balanced')
    lr_pipe = Pipeline([('preprocess', preprocessor), ('model', lr)])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    results.append({'Model': 'Baseline Logistic Regression', 'Accuracy': accuracy_score(y_test, lr_pred), 'Key Detail': 'Standard model using all features.'})

    if SMOTE is not None:
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train_proc, y_train)
        lr_smote = LogisticRegression(max_iter=700, class_weight='balanced')
        lr_smote.fit(X_res, y_res)
        smote_pred = lr_smote.predict(X_test_proc)
        results.append({'Model': 'Logistic Regression (SMOTE)', 'Accuracy': accuracy_score(y_test, smote_pred), 'Key Detail': 'Used oversampling to handle class imbalance; usually improves churn recall.'})
    else:
        results.append({'Model': 'Logistic Regression (SMOTE)', 'Accuracy': np.nan, 'Key Detail': 'imblearn not installed; SMOTE variant skipped.'})

    selector = RFE(estimator=LogisticRegression(max_iter=700), n_features_to_select=min(10, max(2, prepared.X.shape[1] // 2)))
    rfe_pipe = Pipeline([('preprocess', preprocessor), ('rfe', selector), ('model', LogisticRegression(max_iter=700, class_weight='balanced'))])
    rfe_pipe.fit(X_train, y_train)
    rfe_pred = rfe_pipe.predict(X_test)
    results.append({'Model': 'Logistic Regression (RFE)', 'Accuracy': accuracy_score(y_test, rfe_pred), 'Key Detail': 'Recursive Feature Elimination for compact feature set.'})

    rf_pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced', n_jobs=-1)),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    results.append({'Model': 'Random Forest Classifier', 'Accuracy': accuracy_score(y_test, rf_pred), 'Key Detail': 'Tree ensemble baseline for non-linear patterns.'})

    benchmark_df = pd.DataFrame(results)
    benchmark_df['Accuracy (%)'] = (benchmark_df['Accuracy'] * 100).round(2)
    return benchmark_df[['Model', 'Accuracy (%)', 'Key Detail']], None


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #090f1f;
            --bg-surface: #121a31;
            --bg-surface-soft: #182341;
            --border-color: #2a3759;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.16);
        }

        .stApp {
            background: radial-gradient(circle at 12% 8%, #162349 0%, #0e1730 35%, #090f1f 100%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }

        /* Prevent top content from sitting under the header bar */
        [data-testid="stAppViewContainer"] {
            overflow: visible;
        }

        section.main > div {
            overflow: visible;
        }

        [data-testid="stAlert"] {
            overflow: visible;
        }

        /* Prevent uploader truncation and allow wrapping */
        [data-testid="stFileUploader"] {
            width: 100%;
        }

        [data-testid="stFileUploader"] label {
            width: 100%;
        }

        [data-testid="stFileUploaderDropzone"] {
            width: 100%;
            max-width: 100%;
            overflow: visible;
        }

        [data-testid="stFileUploaderDropzone"] > div {
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        [data-testid="stFileUploaderDropzone"] small {
            white-space: normal;
            word-break: break-word;
        }

        /* Allow horizontal radio options to wrap instead of clipping */
        [data-testid="stRadio"] div[role="radiogroup"] {
            flex-wrap: wrap;
            gap: 0.5rem 1rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1730 0%, #0b1020 100%);
            border-right: 1px solid var(--border-color);
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
            border-color: var(--accent) !important;
            background: var(--accent) !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span {
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        .amla-hero {
            padding: 0.6rem 0 2rem 0;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
            margin-bottom: 0.45rem;
        }

        .hero-copy {
            color: var(--text-muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 70ch;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0.45rem 0 1rem 0;
            letter-spacing: 0.01em;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(165deg, var(--bg-surface) 0%, var(--bg-surface-soft) 100%);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.1rem;
            min-height: auto;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.65);
            box-shadow: 0 16px 32px rgba(8, 47, 73, 0.4), 0 0 0 1px rgba(99, 102, 241, 0.2);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.01em;
            word-break: break-word;
            white-space: normal;
            max-width: 100%;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-main);
            font-size: 1.6rem;
            word-break: break-word;
            white-space: normal;
        }

        .metric-empty-state {
            margin-top: 0.45rem;
            color: var(--text-muted);
            font-size: 0.84rem;
            padding: 0.24rem 0.55rem;
            border: 1px dashed rgba(148, 163, 184, 0.42);
            border-radius: 999px;
            display: inline-block;
            background: rgba(15, 23, 42, 0.4);
        }

        .card-shell {
            background: linear-gradient(165deg, rgba(18, 26, 49, 0.94) 0%, rgba(23, 33, 61, 0.86) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.15rem 1.1rem;
            min-height: auto;
            height: 100%;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.32);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

        .card-shell:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.7);
            box-shadow: 0 18px 30px rgba(15, 23, 42, 0.5);
        }

        .card-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 10px;
            background: var(--accent-soft);
            margin-bottom: 0.6rem;
            font-size: 1rem;
        }

        .card-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .card-body {
            color: var(--text-muted);
            line-height: 1.6;
            font-size: 0.94rem;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

        .section-gap {
            margin-top: 2rem;
        }

        .stButton > button {
            background: var(--accent);
            color: white;
            border: 1px solid transparent;
            border-radius: 10px;
        }

        .stButton > button:hover {
            border-color: rgba(255, 255, 255, 0.35);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.25);
        }

        /* Fix dataframe display */
        [data-testid="stDataFrameResizable"] {
            width: 100% !important;
            overflow-x: auto !important;
        }

        /* Fix expander display */
        [data-testid="stExpander"] {
            width: 100%;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 3rem;
            }
            [data-testid="stMetric"] {
                min-height: auto;
            }
            .card-shell {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_status():
    try:
        response = requests.get(f'{API_URL}/health', timeout=3)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def landing_page():
    st.markdown('<div class="amla-hero">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">AMLA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Adaptive Meta-Learning Architecture for end-to-end AutoML: upload, analyze, train, compare, improve, track, and deploy from one workspace.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    health = api_status()
    cols = st.columns(4)
    cols[0].metric('Runtime', 'Live', 'Ready')
    cols[1].metric('API', 'Healthy' if health else 'Offline', 'Connected' if health else 'Unavailable', delta_color='off')
    cols[2].metric('History Runs', len(st.session_state.history), 'Tracked')
    if st.session_state.automl:
        cols[3].metric('Best Model', st.session_state.automl['best_model_name'], 'Ready')
    else:
        cols[3].metric('Best Model', 'Pending', 'Awaiting training', delta_color='off')
        cols[3].markdown('<span class="metric-empty-state">Empty State: no trained model yet</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    quick_actions = st.container()
    with quick_actions:
        st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3, gap='large')
        q1.markdown(
            '<div class="card-shell"><div class="card-icon">1</div><div class="card-title">Step 1: Upload</div><div class="card-body">Import CSV or Excel data, then select your target column to initialize task detection.</div></div>',
            unsafe_allow_html=True,
        )
        q2.markdown(
            '<div class="card-shell"><div class="card-icon">2</div><div class="card-title">Step 2: Analyze + Train</div><div class="card-body">Use EDA and AutoML ranking to surface strong candidate models with explainable performance.</div></div>',
            unsafe_allow_html=True,
        )
        q3.markdown(
            '<div class="card-shell"><div class="card-icon">3</div><div class="card-title">Step 3: Deploy</div><div class="card-body">Export reports, model artifacts, and FastAPI deployment files for production handoff.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    workflow = st.container()
    with workflow:
        st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    steps = [
        ('Upload', '📤', 'Bring data into AMLA with schema preview and target selection.'),
        ('EDA', '📊', 'Inspect quality signals, missingness, distributions, and correlations.'),
        ('Train', '🤖', 'Run ranked model training for classification or regression tasks.'),
        ('Compare', '🧪', 'Benchmark models with side-by-side metrics and visual diagnostics.'),
        ('Improve', '💡', 'Receive data and model enhancement suggestions with explainability insights.'),
        ('Track / Deploy', '🚀', 'Track run history and export deployable artifacts for production.'),
    ]
    for row in range(0, len(steps), 3):
        cards = st.columns(3, gap='large')
        for card, (title, icon, body) in zip(cards, steps[row:row + 3]):
            card.markdown(
                f'<div class="card-shell"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


def upload_page():
    """Unified upload page supporting tabular and image datasets."""
    st.header('📤 Upload Dataset')
    
    dataset_type_choice = st.radio(
        'Dataset Type',
        ['📊 Tabular (CSV/Excel)', '📸 Images (PNG/JPG/etc)'],
        horizontal=True,
        key='dataset_type_radio'
    )
    
    if 'Tabular' in dataset_type_choice:
        upload_tabular_data()
    else:
        image_upload_page()


def upload_tabular_data():
    """Upload and analyze tabular dataset."""
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])

    df = None
    if uploaded is not None:
        try:
            df = load_data(uploaded)
            st.session_state.df = df
            st.session_state.dataset_type = 'tabular'
        except Exception as exc:
            st.error(str(exc))
            return
    elif st.session_state.df is not None and st.session_state.dataset_type == 'tabular':
        df = st.session_state.df
        st.info('Showing the current cleaned dataset from session. Upload a new file to replace it.')
    else:
        st.info('Upload a dataset to unlock analysis and model training.')
        return
    
    st.subheader('Dataset Preview')
    preview = df.head(10).style.highlight_null(color='#fca5a5')
    st.dataframe(preview, width='stretch')

    dtype_df = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[col].dtype) for col in df.columns],
        'role': ['numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical' for col in df.columns],
        'missing': [int(df[col].isna().sum()) for col in df.columns],
    })
    with st.expander('📋 Column Types and Missing Values', expanded=False):
        st.dataframe(dtype_df, width='stretch')

    target_guess = df.columns[-1]
    target_col = st.selectbox('Target Column', df.columns.tolist(), index=df.columns.get_loc(target_guess))
    detected, confidence, reason = infer_task_type(df, target_col)
    task_options = ['classification', 'regression']
    selected_task = st.radio(
        f'Task Type: auto-detected {detected} ({confidence:.0%})',
        task_options,
        index=task_options.index(detected),
        horizontal=True,
        help=reason,
    )

    st.session_state.target_col = target_col
    st.session_state.task_type = selected_task
    st.session_state.task_confidence = confidence
    st.session_state.task_reason = reason
    
    # Use caching for profile
    profile = cache_dataset_profile(df, target_col)
    st.session_state.profile = profile

    cols = st.columns(5)
    cols[0].metric('Rows', profile['rows'], help='Number of rows')
    cols[1].metric('Columns', profile['columns'], help='Number of columns')
    cols[2].metric('Missing', profile['missing_total'], help='Missing cells')
    cols[3].metric('Numeric', profile['numeric_columns'], help='Numeric features')
    cols[4].metric('Categorical', profile['categorical_columns'], help='Categorical features')

    if st.session_state.df_original is not None and uploaded is None:
        st.success('✅ Cleaned dataset is loaded from session state. Download it or re-train directly from here.')

    current_csv = df.to_csv(index=False)
    st.download_button(
        '📥 Download Current Dataset',
        current_csv,
        f'current_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        'text/csv',
        use_container_width=True,
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🤖 Run AutoML Pipeline', type='primary', use_container_width=True):
            run_automl()
    
    with col2:
        csv_sample = df.head(100).to_csv(index=False)
        st.download_button(
            '📥 Download Sample',
            csv_sample,
            f'dataset_sample_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            'text/csv',
            use_container_width=True
        )


def run_automl():
    df = st.session_state.df
    target_col = st.session_state.target_col
    task_type = st.session_state.task_type
    if df is None or target_col is None:
        st.error('Upload a dataset first.')
        return

    progress = st.progress(0)
    status = st.empty()
    runtime_rows = []

    def on_progress(completed, total, model_name, row):
        progress.progress(completed / total)
        status.write(f"Finished {model_name} in {row.get('runtime_seconds', 0):.2f}s")
        runtime_rows.append({
            'model': model_name,
            'status': row.get('status'),
            'runtime_seconds': row.get('runtime_seconds', 0),
        })

    with st.spinner('Training models in parallel, ranking performance, and generating explanations...'):
        result = train_models(df, target_col, task_type, progress_callback=on_progress, max_workers=2, fast_mode=True)
        result['runtime_table'] = pd.DataFrame(runtime_rows)
        st.session_state.automl = result
        ranked = result['ranked'].iloc[0]
        score = float(ranked['f1'] if task_type == 'classification' else ranked['r2'])
        st.session_state.history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'task': task_type,
            'target': target_col,
            'best_model': result['best_model_name'],
            'score': score,
            **dataset_fingerprint(df),
        })
    st.success(f"Best model: {result['best_model_name']}")


def analysis_page():
    df = require_dataset()
    if df is None:
        return
    target_col = st.session_state.target_col
    task_type = st.session_state.task_type
    profile = st.session_state.profile or dataset_profile(df, target_col)

    st.header('Dataset Analysis')
    cols = st.columns(4)
    cols[0].metric('Target', target_col)
    cols[1].metric('Task', task_type.title())
    cols[2].metric('Target Unique', profile['target_unique'])
    cols[3].metric('Duplicate Rows', profile['duplicate_rows'])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    st.subheader('Interactive Filters')
    selected_features = st.multiselect('Feature subset', feature_cols, default=feature_cols[:min(6, len(feature_cols))])
    filtered_df = df.copy()
    group_col = st.selectbox('Group/filter by categorical column', ['None'] + categorical_cols)
    if group_col != 'None':
        allowed = st.multiselect(f'Values in {group_col}', sorted(df[group_col].dropna().astype(str).unique().tolist()))
        if allowed:
            filtered_df = filtered_df[filtered_df[group_col].astype(str).isin(allowed)]
    st.caption(f'Active rows after filters: {len(filtered_df)}')

    tab_dist, tab_missing, tab_corr, tab_outliers, tab_stats, tab_signal, tab_notebook = st.tabs([
        'Distributions', 'Missing Values', 'Correlation', 'Outliers', 'Stats Infographics', 'Feature Signal', 'Notebook-Style EDA'
    ])

    with tab_dist:
        if selected_features:
            selected = selected_features
            for col in selected:
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(filtered_df, x=col, color=target_col if task_type == 'classification' else None, marginal='box'), width='stretch', config=PLOT_CONFIG)
                c2.plotly_chart(px.box(filtered_df, y=col, x=target_col if task_type == 'classification' else None), width='stretch', config=PLOT_CONFIG)
            if len(selected) >= 2:
                chart(px.scatter_matrix(filtered_df, dimensions=selected[:5], color=target_col if task_type == 'classification' else None))
        else:
            st.info('No numeric feature columns found for distribution plots.')

    with tab_missing:
        missing = profile['missing_by_column']
        if len(missing):
            chart(px.bar(missing.reset_index(), x='index', y=0, labels={'index': 'Column', '0': 'Missing Count'}))
        else:
            st.success('No missing values detected.')
        missing_map = df.isna().astype(int).iloc[:200]
        chart(px.imshow(missing_map.T, aspect='auto', title='Missing Value Map (first 200 rows)'))

    with tab_corr:
        if len(numeric_cols) >= 2:
            corr = filtered_df[numeric_cols].corr(numeric_only=True)
            chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1))
        else:
            st.info('At least two numeric columns are required for correlation analysis.')

    with tab_outliers:
        outliers = profile['outlier_counts']
        if len(outliers):
            chart(px.bar(outliers.head(20).reset_index(), x='index', y=0, labels={'index': 'Feature', '0': 'Outlier Count'}))
        else:
            st.success('No numeric outliers detected by the IQR rule.')

    with tab_stats:
        st.dataframe(profile['stats_table'], width='stretch')
        if not profile['stats_table'].empty:
            chart(px.bar(profile['stats_table'].head(25), x='feature', y=['mean', 'median', 'skew', 'kurt'], barmode='group'))

    with tab_signal:
        signal = feature_signal_scores(df, target_col, task_type)
        st.dataframe(signal.head(30), width='stretch')
        chart(px.bar(signal.head(20), x='score', y='feature', orientation='h', title='Mutual Information Feature Signal'))

    with tab_notebook:
        st.caption('Aligned with your notebook chart family: count plots, box/violin, heatmap, distributions, pair plots, bar chart, pie/donut, and sankey.')

        if task_type == 'classification':
            churn_counts = filtered_df[target_col].astype(str).value_counts().reset_index()
            churn_counts.columns = ['class', 'count']
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(churn_counts, x='class', y='count', title='Count Plot: Target Class Distribution'), width='stretch', config=PLOT_CONFIG, key='nb_count_target')
            c2.plotly_chart(px.pie(churn_counts, names='class', values='count', hole=0.45, title='Donut: Churn Composition'), width='stretch', config=PLOT_CONFIG, key='nb_donut_target')

        cat_candidates = [c for c in categorical_cols if c != target_col]
        if cat_candidates and task_type == 'classification':
            cat_col = st.selectbox('Notebook Count Plot by Category', cat_candidates, key='nb_cat_col')
            count_df = filtered_df.groupby([cat_col, target_col]).size().reset_index(name='count')
            chart(px.bar(count_df, x=cat_col, y='count', color=target_col, barmode='group', title=f'Count Plot: {cat_col} by {target_col}'), key='nb_count_grouped')

        if feature_cols and task_type == 'classification':
            num_for_box = st.selectbox('Numeric Feature for Box/Violin', feature_cols, key='nb_num_col')
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.box(filtered_df, x=target_col, y=num_for_box, title=f'Box Plot: {num_for_box} by {target_col}'), width='stretch', config=PLOT_CONFIG, key='nb_box')
            c2.plotly_chart(px.violin(filtered_df, x=target_col, y=num_for_box, box=True, points='outliers', title=f'Violin Plot: {num_for_box} by {target_col}'), width='stretch', config=PLOT_CONFIG, key='nb_violin')

        if feature_cols:
            dist_col = st.selectbox('Distribution Feature (Histogram + KDE-like shape)', feature_cols, key='nb_dist_col')
            chart(px.histogram(filtered_df, x=dist_col, color=target_col if task_type == 'classification' else None, marginal='box', opacity=0.75, title=f'Distribution Plot: {dist_col}'), key='nb_dist')

        if len(feature_cols) >= 2:
            chart(px.scatter_matrix(filtered_df, dimensions=feature_cols[:5], color=target_col if task_type == 'classification' else None, title='Pair Plot (Scatter Matrix)'), key='nb_pair')

        if 'tenure' in filtered_df.columns and 'MonthlyCharges' in filtered_df.columns:
            tenure_bins = pd.cut(filtered_df['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
            avg_df = filtered_df.assign(tenure_group=tenure_bins).groupby('tenure_group', observed=False)['MonthlyCharges'].mean().reset_index()
            chart(px.bar(avg_df, x='tenure_group', y='MonthlyCharges', title='Bar Plot: Avg Monthly Charges by Tenure Group'), key='nb_bar_tenure')

        if task_type == 'classification':
            sankey_sources = ['Contract', 'PaymentMethod']
            if all(col in filtered_df.columns for col in sankey_sources):
                source_col, mid_col = sankey_sources
                target_col_name = target_col
                s1 = filtered_df.groupby([source_col, mid_col]).size().reset_index(name='value')
                s2 = filtered_df.groupby([mid_col, target_col_name]).size().reset_index(name='value')
                nodes = pd.Index(pd.concat([s1[source_col], s1[mid_col], s2[target_col_name]]).astype(str).unique())
                node_index = {v: i for i, v in enumerate(nodes)}
                link_src = [node_index[v] for v in s1[source_col].astype(str)] + [node_index[v] for v in s2[mid_col].astype(str)]
                link_tgt = [node_index[v] for v in s1[mid_col].astype(str)] + [node_index[v] for v in s2[target_col_name].astype(str)]
                link_val = s1['value'].tolist() + s2['value'].tolist()
                fig = go.Figure(data=[go.Sankey(node={'label': nodes.tolist()}, link={'source': link_src, 'target': link_tgt, 'value': link_val})])
                fig.update_layout(title='Sankey: Service Path to Churn Outcome')
                chart(fig, key='nb_sankey')


def training_page():
    df = require_dataset()
    if df is None:
        return
    st.header('Model Training & Ranking')

    if st.button('Train / Re-run Models', type='primary'):
        run_automl()

    result = st.session_state.automl
    if not result:
        st.info('Run the AutoML pipeline from Upload Dataset or click Train / Re-run Models.')
        return

    metrics = result['metrics']
    ranked = result['ranked']
    best = ranked.iloc[0]
    metric_name = result['sort_metric']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Best Model', result['best_model_name'])
    c2.metric(metric_name.upper(), f'{best[metric_name]:.4f}')
    c3.metric('Models Trained', len(ranked))
    c4.metric('Total Runtime', f"{result.get('total_runtime_seconds', 0):.2f}s")

    st.dataframe(metrics, width='stretch')
    if 'runtime_table' in result and not result['runtime_table'].empty:
        st.subheader('Parallel Training Runtime')
        st.dataframe(result['runtime_table'], width='stretch')
        chart(px.bar(result['runtime_table'], x='model', y='runtime_seconds', color='status', title='Model Runtime Estimates'))
    plot_cols = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc'] if st.session_state.task_type == 'classification' else ['rmse', 'mae', 'r2']
    plot_cols = [col for col in plot_cols if col in ranked.columns]
    chart(px.bar(ranked, x='model', y=plot_cols, barmode='group', title='Model Ranking'))

    st.subheader('Notebook-Style Model Variants (Parity Block)')
    st.caption('This section mirrors your notebook-style comparisons so model suggestions and reported analysis stay aligned.')
    if st.session_state.task_type == 'classification':
        if st.button('Run Notebook-Style Benchmark Variants'):
            with st.spinner('Running Baseline LR, LR+SMOTE, LR+RFE, and RF variants...'):
                benchmark_df, benchmark_error = notebook_style_benchmarks(st.session_state.df, st.session_state.target_col)
                if benchmark_error:
                    st.warning(benchmark_error)
                else:
                    st.dataframe(benchmark_df, width='stretch')
                    chart(px.bar(benchmark_df, x='Model', y='Accuracy (%)', title='Notebook Variant Accuracy Comparison'))
    else:
        st.info('Notebook variant block is currently enabled for classification tasks.')

    model_path = 'analysis_results/best_automl_model.joblib'
    if st.button('Export Best Model as Joblib'):
        export_model(result['best_pipeline'], model_path)
        st.success(f'Exported to {model_path}. Serve it with FastAPI for deployment.')


def comparison_page():
    result = require_automl()
    if result is None:
        return
    st.header('Model Comparison')

    models = result['ranked']['model'].tolist()
    selected = st.multiselect('Models to compare', models, default=models[:min(3, len(models))])
    focus_model = st.selectbox('Instant model focus', selected if selected else models)
    if not selected:
        return

    filtered = result['metrics'][result['metrics']['model'].isin(selected)]
    st.dataframe(filtered, width='stretch')

    task_type = st.session_state.task_type
    st.subheader(f'Focused Analysis: {focus_model}')
    if st.session_state.task_type == 'classification':
        show_classification_visuals(result['artifacts'][focus_model], key_prefix=f'focus_{focus_model}')
    else:
        show_regression_visuals(result['artifacts'][focus_model], key_prefix=f'focus_{focus_model}')

    with st.expander('Side-by-side model visualizations', expanded=False):
        for model_name in selected:
            artifact = result['artifacts'][model_name]
            st.subheader(model_name)
            if task_type == 'classification':
                show_classification_visuals(artifact, key_prefix=f'side_{model_name}')
            else:
                show_regression_visuals(artifact, key_prefix=f'side_{model_name}')

    profile = st.session_state.profile or dataset_profile(st.session_state.df, st.session_state.target_col)
    suggestions = improvement_suggestions(st.session_state.df, st.session_state.target_col, st.session_state.task_type, result['metrics'])
    html_path = generate_html_report('analysis_results/automl_report.html', 'uploaded_dataset', st.session_state.target_col, st.session_state.task_type, profile, filtered, suggestions)
    pdf_path = generate_pdf_report('analysis_results/automl_report.pdf', 'uploaded_dataset', st.session_state.target_col, st.session_state.task_type, profile, filtered, suggestions)
    report_html = Path(html_path).read_text(encoding='utf-8')
    st.download_button('Download Comparison Report (HTML)', report_html, 'model_comparison_report.html', 'text/html')
    with open(pdf_path, 'rb') as pdf_file:
        st.download_button('Download Comparison Report (PDF)', pdf_file, 'model_comparison_report.pdf', 'application/pdf')


def show_classification_visuals(artifact, key_prefix='classification'):
    y_true = artifact['y_test']
    y_pred = artifact['y_pred']
    y_proba = artifact['y_proba']
    cm = confusion_matrix(y_true, y_pred)
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.imshow(cm, text_auto=True, title='Confusion Matrix', labels={'x': 'Predicted', 'y': 'Actual'}),
        width='stretch',
        config=PLOT_CONFIG,
        key=f'{key_prefix}_cm',
    )
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        c2.plotly_chart(
            px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}, title='ROC Curve'),
            width='stretch',
            config=PLOT_CONFIG,
            key=f'{key_prefix}_roc',
        )
        chart(
            px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'}, title='Precision-Recall Curve'),
            key=f'{key_prefix}_pr',
        )
    else:
        c2.info('ROC/PR curves require binary classification probabilities.')


def show_regression_visuals(artifact, key_prefix='regression'):
    y_true = np.asarray(artifact['y_test'])
    y_pred = np.asarray(artifact['y_pred'])
    residuals = y_true - y_pred
    df_plot = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Residual': residuals})
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.scatter(df_plot, x='Actual', y='Predicted', trendline='ols', title='Predicted vs Actual'),
        width='stretch',
        config=PLOT_CONFIG,
        key=f'{key_prefix}_pred_vs_actual',
    )
    c2.plotly_chart(
        px.scatter(df_plot, x='Predicted', y='Residual', title='Residual Plot'),
        width='stretch',
        config=PLOT_CONFIG,
        key=f'{key_prefix}_residual',
    )
    chart(px.histogram(df_plot, x='Residual', nbins=30, title='Error Distribution'), key=f'{key_prefix}_err_dist')


def suggestions_page():
    result = require_automl()
    if result is None:
        return
    st.header('Improvement Suggestions')
    df = st.session_state.df
    target_col = st.session_state.target_col
    task_type = st.session_state.task_type

    suggestions = improvement_suggestions(df, target_col, task_type, result['metrics'])
    for item in suggestions:
        st.info(item)

    # Data Cleaning & Re-upload Button
    st.divider()
    st.subheader('🔧 Auto-Fix Dataset')
    col1, col2 = st.columns(2)
    with col1:
        aggressive_cleaning = st.checkbox('Enable aggressive cleaning (log transforms, outlier clipping)', value=False, help='Applies log transforms to skewed features and clips outliers to IQR bounds.')
    with col2:
        st.write('')  # Spacer for alignment
    
    if st.button('✨ Apply Suggestions & Download Cleaned Dataset', type='primary', use_container_width=True):
        with st.spinner('🔄 Cleaning dataset and applying suggestions...'):
            from amla.automl_engine import auto_fix_dataset
            df_clean, changes_log = auto_fix_dataset(df, target_col, task_type, aggressive=aggressive_cleaning)
            
            # Store original for potential rollback
            if 'df_original' not in st.session_state:
                st.session_state.df_original = df.copy()
            
            # Update session state with cleaned dataset
            st.session_state.df = df_clean
            st.session_state.dataset_type = 'tabular'
            st.session_state.nav_page = 'Upload'
            
            # Show before/after comparison
            st.success('✅ Dataset cleaned and ready!')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Rows (Before)', len(df), delta=len(df_clean) - len(df))
            with col2:
                missing_before = int(df.isna().sum().sum())
                missing_after = int(df_clean.isna().sum().sum())
                st.metric('Missing Cells (Before)', missing_before, delta=missing_after - missing_before)
            with col3:
                dup_before = int(df.duplicated().sum())
                dup_after = int(df_clean.duplicated().sum())
                st.metric('Duplicates (Before)', dup_before, delta=dup_after - dup_before)
            
            # Show detailed changes
            st.info('**Changes Applied:**')
            for change in changes_log:
                st.write(f"• {change}")
            
            # ===== DOWNLOAD SECTION =====
            st.divider()
            st.subheader('📥 Download Cleaned Dataset')
            
            # CSV download
            csv_buffer = io.StringIO()
            df_clean.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label='📊 Download as CSV',
                data=csv_data,
                file_name=f'cleaned_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                key='download_csv_cleaned'
            )
            
            # Excel download
            excel_buffer = io.BytesIO()
            try:
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_clean.to_excel(writer, sheet_name='Cleaned Data', index=False)
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Original Rows', 'Cleaned Rows', 'Rows Removed', 'Missing Cells Before', 'Missing Cells After', 'Duplicates Before', 'Duplicates After'],
                        'Value': [len(df), len(df_clean), len(df) - len(df_clean), missing_before, missing_after, dup_before, dup_after]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                excel_buffer.seek(0)
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label='📈 Download as Excel (with Summary)',
                    data=excel_data,
                    file_name=f'cleaned_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                    mime='application/vnd.ms-excel',
                    key='download_xlsx_cleaned'
                )
            except Exception as e:
                st.warning(f'Excel export not available: {e}')
            
            st.divider()
            
            # ===== NEXT STEPS =====
            st.subheader('🔄 Next Steps')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button('📤 Go to Upload Page (cleaned data ready)', use_container_width=True, key='goto_upload'):
                    st.session_state.nav_page = 'Upload'
                    st.rerun()
            
            with col2:
                if st.button('🤖 Re-train Models', use_container_width=True, key='retrain_models'):
                    run_automl()
                    st.success('Models training with cleaned data...')
                    st.rerun()
            
            with col3:
                if st.button('↩️ Revert to Original', use_container_width=True, key='revert_original'):
                    st.session_state.df = st.session_state.df_original.copy()
                    st.success('Dataset reverted to original.')
                    st.rerun()
            
            st.warning('💡 Tip: Run AutoML again on the cleaned dataset to see if model performance improves!', icon='💡')
    
    st.divider()

    st.subheader('AI-Generated What-if Scenarios')
    for item in what_if_scenarios(result['metrics'], task_type, len(df)):
        st.success(item)

    st.subheader('Feature Importance')
    importance = feature_importance(result['best_pipeline'])
    if not importance.empty:
        chart(px.bar(importance.head(20), x='importance', y='feature', orientation='h'))
    else:
        st.write('Feature importance is not available for this best model.')

    st.subheader('Explainable AI: SHAP')
    with st.expander('Load SHAP explanation', expanded=False):
        shap_df, shap_error = shap_summary(result['best_pipeline'], st.session_state.df.drop(columns=[target_col]))
        if shap_df is not None:
            chart(px.bar(shap_df.head(20), x='shap_importance', y='feature', orientation='h'))
        else:
            st.warning(shap_error)

    st.subheader('Explainable AI: LIME')
    with st.expander('Load LIME explanation', expanded=False):
        lime_df, lime_error = lime_summary(result['best_pipeline'], result['prepared'])
        if lime_df is not None:
            st.dataframe(lime_df, width='stretch')
            chart(px.bar(lime_df, x='contribution', y='feature_rule', orientation='h'))
        else:
            st.warning(lime_error)

    st.subheader('Model Stability')
    with st.expander('Load stability curves', expanded=False):
        try:
            prepared = result['prepared']
            learning, validation = stability_curves(result['best_pipeline'], prepared.X, prepared.y, task_type)
            chart(px.line(learning, x='train_size', y=['train_score', 'validation_score'], title='Learning Curve / Bias-Variance Tradeoff'))
            if validation is not None:
                chart(px.line(validation, x='max_depth', y=['train_score', 'validation_score'], title='Validation Curve'))
        except Exception as exc:
            st.warning(f'Stability curves could not be generated: {exc}')


def deployment_page():
    result = require_automl()
    if result is None:
        return
    st.header('Deployment & Export')
    deployment_dir = Path('analysis_results/deployment')
    model_path = deployment_dir / 'best_automl_model.joblib'
    if st.button('Export Model + FastAPI + Docker Template', type='primary'):
        try:
            deployment_dir.mkdir(parents=True, exist_ok=True)
            export_model(result['best_pipeline'], str(model_path))
            files = generate_deployment_files(str(deployment_dir))
            st.success('Deployment package generated in analysis_results/deployment')
            st.json({'model': str(model_path), **files})
        except Exception as exc:
            st.error(f'Deployment export failed: {exc}')
            return

    for file_name in ['serve_model_fastapi.py', 'Dockerfile', 'requirements-serving.txt']:
        file_path = deployment_dir / file_name
        if file_path.exists():
            st.download_button(
                f'Download {file_name}',
                file_path.read_bytes(),
                file_name,
                'text/plain',
            )


def tracking_page():
    st.header('Performance Tracking')
    history = pd.DataFrame(st.session_state.history)
    if history.empty:
        st.info('Run the pipeline on one or more dataset versions to track changes.')
        return

    st.dataframe(history, width='stretch')
    chart(px.line(history, x='time', y='score', markers=True, color='task', title='Performance Across Dataset Versions'))
    timeline = go.Figure()
    timeline.add_trace(go.Scatter(
        x=history['time'],
        y=history['score'],
        mode='lines+markers+text',
        text=history['best_model'],
        textposition='top center',
        name='Best Score',
    ))
    timeline.update_layout(title='Dataset Version Timeline', xaxis_title='Run Time', yaxis_title='Accuracy / R2 / F1')
    chart(timeline)
    chart(px.bar(history, x='time', y=['rows', 'columns', 'missing_total', 'duplicate_rows'], barmode='group', title='Dataset Change Impact'))
    st.download_button('Download Tracking CSV', history.to_csv(index=False), 'performance_tracking.csv', 'text/csv')


def image_upload_page():
    """Upload and analyze image datasets for computer vision tasks."""
    st.header('📸 Upload Image Dataset')
    st.caption('Upload images for classification/detection tasks. Supports PNG, JPG, BMP, GIF, WebP')
    
    if Image is None:
        st.error('PIL (Pillow) not installed. Please run: pip install Pillow')
        return
    
    try:
        from amla.image_processor import validate_image_dataset, ImageDataset
    except ImportError:
        st.error('Image processor module not available.')
        return
    
    image_files = st.file_uploader(
        'Upload images',
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'],
        accept_multiple_files=True
    )
    
    if not image_files:
        st.info('Upload image files to get started.')
        return
    
    # Validate images
    metadata, errors = validate_image_dataset(image_files)
    
    if errors:
        st.warning('⚠️ Validation Issues:')
        for error in errors:
            st.write(error)
    
    if metadata:
        # Metadata summary
        meta_df = pd.DataFrame(metadata)
        
        st.subheader('📊 Image Metadata Summary')
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric('Total Images', len(meta_df))
        col2.metric('Avg Width', f"{meta_df['width'].mean():.0f}px")
        col3.metric('Avg Height', f"{meta_df['height'].mean():.0f}px")
        col4.metric('Avg Size', f"{meta_df['size_bytes'].mean()/1024:.1f}KB")
        col5.metric('Formats', f"{meta_df['format'].nunique()}")
        
        with st.expander('📋 Detailed Metadata', expanded=False):
            st.dataframe(meta_df, width='stretch')
        
        # Image gallery
        st.subheader('🖼️ Image Gallery (First 9)')
        gallery_cols = st.columns(3)
        for idx, (col, file) in enumerate(zip(gallery_cols * 3, image_files[:9])):
            with col:
                try:
                    img = Image.open(file)
                    st.image(img, use_column_width=True, caption=file.name[:20])
                    file.seek(0)
                except Exception as e:
                    st.error(f'Cannot load {file.name}: {e}')
        
        # Store for analysis
        st.session_state.image_dataset = ImageDataset(image_files)
        st.session_state.dataset_type = 'image'
        st.session_state.image_metadata = metadata
        
        # Task configuration
        st.divider()
        st.subheader('⚙️ Task Configuration')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            task = st.radio('Task Type', ['Classification', 'Detection', 'Segmentation'], horizontal=True)
        with col2:
            n_classes = st.number_input('Number of Classes', min_value=2, max_value=100, value=2)
        with col3:
            use_cnn = st.checkbox('Extract CNN Features', value=True, help='Use pre-trained deep learning model')
        
        st.session_state.task_type = task.lower()
        st.session_state.n_classes = n_classes
        st.session_state.use_cnn_features = use_cnn
        
        st.success('✅ Image dataset loaded successfully!')
        st.info(f'Ready to proceed with {task} task on {len(meta_df)} images.')
    else:
        st.error('❌ No valid images found. Please check file formats.')


def image_analysis_page():
    """Analyze image dataset characteristics."""
    if st.session_state.image_metadata is None:
        st.warning('No image dataset loaded. Go to Upload Images first.')
        return
    
    st.header('📊 Image Dataset Analysis')
    
    metadata = st.session_state.image_metadata
    meta_df = pd.DataFrame(metadata)
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Total Images', len(meta_df))
    col2.metric('Image Types', meta_df['format'].nunique())
    col3.metric('Total Size', f"{meta_df['size_bytes'].sum() / (1024*1024):.2f}MB")
    col4.metric('Avg Dimensions', f"{meta_df['width'].mean():.0f}×{meta_df['height'].mean():.0f}px")
    col5.metric('Color Channels', f"{meta_df['channels'].mode()[0] if len(meta_df['channels'].mode()) > 0 else 3}")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Distributions', 'Quality', 'Characteristics', 'Preview'])
    
    with tab1:
        st.subheader('📈 Image Size Distributions')
        col1, col2 = st.columns(2)
        
        with col1:
            chart(px.histogram(meta_df, x='width', nbins=30, title='Width Distribution'))
        with col2:
            chart(px.histogram(meta_df, x='height', nbins=30, title='Height Distribution'))
        
        chart(px.scatter(meta_df, x='width', y='height', color='channels', title='Image Dimensions (colored by channels)'))
    
    with tab2:
        st.subheader('🔍 Image Quality Metrics')
        
        col1, col2 = st.columns(2)
        with col1:
            chart(px.histogram(meta_df, x='size_bytes', nbins=20, title='File Size Distribution'))
        with col2:
            format_counts = meta_df['format'].value_counts().reset_index()
            format_counts.columns = ['Format', 'Count']
            chart(px.pie(format_counts, names='Format', values='Count', title='Format Distribution'))
        
        # Size anomalies
        size_mean = meta_df['size_bytes'].mean()
        size_std = meta_df['size_bytes'].std()
        outliers = meta_df[(meta_df['size_bytes'] > size_mean + 3 * size_std) | (meta_df['size_bytes'] < size_mean - 3 * size_std)]
        if len(outliers) > 0:
            st.warning(f'⚠️ Found {len(outliers)} size outliers (>3σ from mean). Review for potential issues.')
    
    with tab3:
        st.subheader('🎨 Image Characteristics')
        
        chart(px.box(meta_df, y='width', title='Width Statistics'))
        chart(px.box(meta_df, y='height', title='Height Statistics'))
        
        # Aspect ratio
        meta_df['aspect_ratio'] = meta_df['width'] / meta_df['height']
        chart(px.histogram(meta_df, x='aspect_ratio', nbins=25, title='Aspect Ratio Distribution'))
    
    with tab4:
        st.subheader('🖼️ Full Image Gallery')
        if st.session_state.image_dataset:
            image_files = st.session_state.image_dataset.image_files
            n_cols = st.slider('Columns in gallery', min_value=2, max_value=6, value=3)
            
            gallery_cols = st.columns(n_cols)
            for idx, file in enumerate(image_files):
                col = gallery_cols[idx % n_cols]
                with col:
                    try:
                        img = Image.open(file)
                        st.image(img, use_column_width=True, caption=file.name)
                        file.seek(0)
                    except Exception as e:
                        st.error(f'Error: {e}')


def require_dataset():
    if st.session_state.df is None:
        st.warning('Upload a dataset first.')
        return None
    return st.session_state.df


def require_automl():
    if st.session_state.automl is None:
        st.warning('Run model training first.')
        return None
    return st.session_state.automl


def main():
    init_state()
    with st.sidebar:
        st.title('AMLA')
        st.caption('Adaptive Meta-Learning Architecture')
        st.markdown('---')
        page = st.radio(
            'Navigation',
            [
                'Home',
                'Upload',
                'EDA (Tabular)',
                'Image Analysis',
                'Train',
                'Compare',
                'Improve',
                'Track',
                'Deploy',
            ],
            key='nav_page',
        )
        if st.session_state.df is not None:
            st.caption(f"Tabular: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
        if st.session_state.image_metadata is not None:
            st.caption(f"Images: {len(st.session_state.image_metadata)} files")

    if st.session_state.theme != 'Dark':
        st.session_state.theme = 'Dark'
        persist_theme()

    if st.session_state.nav_page != page:
        page = st.session_state.nav_page

    apply_theme()

    pages = {
        'Home': landing_page,
        'Upload': upload_page,
        'EDA (Tabular)': analysis_page,
        'Image Analysis': image_analysis_page,
        'Train': training_page,
        'Compare': comparison_page,
        'Improve': suggestions_page,
        'Track': tracking_page,
        'Deploy': deployment_page,
    }
    pages[page]()


if __name__ == '__main__':
    main()
