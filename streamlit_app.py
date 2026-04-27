import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

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

st.set_page_config(
    page_title='AMLA AutoML Platform',
    page_icon='AMLA',
    layout='wide',
    initial_sidebar_state='expanded',
)


def init_state():
    stored_theme = 'Light'
    if PREFERENCES_PATH.exists():
        try:
            stored_theme = json.loads(PREFERENCES_PATH.read_text()).get('theme', 'Light')
        except Exception:
            stored_theme = 'Light'

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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persist_theme():
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(json.dumps({'theme': st.session_state.theme}), encoding='utf-8')


def chart(fig, **kwargs):
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, **kwargs)


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #0b1020;
            --bg-surface: #121a31;
            --bg-surface-soft: #17213d;
            --border-color: #283453;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.15);
        }

        .stApp {
            background: radial-gradient(circle at 10% 10%, #121b39 0%, #0b1020 45%, #070b16 100%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1730 0%, #0b1020 100%);
            border-right: 1px solid var(--border-color);
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
            min-height: 120px;
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
        }

        [data-testid="stMetricValue"] {
            color: var(--text-main);
            font-size: 1.6rem;
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
            min-height: 170px;
            height: 100%;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.32);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
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
            line-height: 1.55;
            font-size: 0.94rem;
        }

        .section-gap {
            margin-top: 1.9rem;
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

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            [data-testid="stMetric"] {
                min-height: 110px;
            }
            .card-shell {
                min-height: 145px;
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
    st.markdown('<div class="hero-title">AMLA AutoML Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Build, compare, improve, and deploy machine learning workflows with a unified AutoML dashboard designed for speed and clarity.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    health = api_status()
    cols = st.columns(4)
    cols[0].metric('Runtime', 'Live', 'Optimized')
    cols[1].metric('API', 'Healthy' if health else 'Offline', 'Connected' if health else 'Unavailable', delta_color='off')
    cols[2].metric('History Runs', len(st.session_state.history), 'Persistent')
    if st.session_state.automl:
        cols[3].metric('Best Model', st.session_state.automl['best_model_name'], 'Ready')
    else:
        cols[3].metric('Best Model', '—', 'Awaiting training', delta_color='off')
        cols[3].markdown('<span class="metric-empty-state">Empty State: no trained model yet</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    quick_actions = st.container()
    with quick_actions:
        st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3, gap='large')
        q1.markdown(
            '<div class="card-shell"><div class="card-icon">1</div><div class="card-title">Upload Dataset</div><div class="card-body">Import CSV or Excel files, choose your target, and instantly validate feature quality.</div></div>',
            unsafe_allow_html=True,
        )
        q2.markdown(
            '<div class="card-shell"><div class="card-icon">2</div><div class="card-title">Train & Rank</div><div class="card-body">Run automated model training, evaluate key metrics, and surface the strongest performer.</div></div>',
            unsafe_allow_html=True,
        )
        q3.markdown(
            '<div class="card-shell"><div class="card-icon">3</div><div class="card-title">Deploy Artifacts</div><div class="card-body">Export model bundles, reports, and deployment templates for production-ready delivery.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    workflow = st.container()
    with workflow:
        st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)
    steps = [
        ('Upload', '📤', 'Bring your data from CSV or Excel and define the prediction target in seconds.'),
        ('EDA', '📊', 'Explore distributions, missing values, correlations, and outliers through visual diagnostics.'),
        ('Train', '🤖', 'Launch the AutoML engine to train multiple candidate models and rank their performance.'),
        ('Compare', '🧪', 'Inspect model metrics side-by-side and analyze confusion, ROC, residual, and error plots.'),
        ('Improve', '💡', 'Apply actionable recommendations for feature engineering and quality-driven uplift.'),
        ('Track & Deploy', '🚀', 'Track progress across historical runs and export deployment-ready artifacts.'),
    ]
    for row in range(0, len(steps), 3):
        cards = st.columns(3, gap='large')
        for card, (title, icon, body) in zip(cards, steps[row:row + 3]):
            card.markdown(
                f'<div class="card-shell"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


def upload_page():
    st.header('Upload Dataset')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])

    if uploaded is None:
        st.info('Upload a dataset to unlock analysis and model training.')
        return

    try:
        df = read_tabular_file(uploaded)
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state.df = df
    st.subheader('Dataset Preview')
    preview = df.head(10).style.highlight_null(color='#fca5a5')
    st.dataframe(preview, use_container_width=True)

    dtype_df = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[col].dtype) for col in df.columns],
        'role': ['numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical' for col in df.columns],
        'missing': [int(df[col].isna().sum()) for col in df.columns],
    })
    with st.expander('Column Types and Missing Values', expanded=False):
        st.dataframe(dtype_df, use_container_width=True)

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
    st.session_state.profile = dataset_profile(df, target_col)

    cols = st.columns(5)
    profile = st.session_state.profile
    cols[0].metric('Rows', profile['rows'])
    cols[1].metric('Columns', profile['columns'])
    cols[2].metric('Missing Cells', profile['missing_total'])
    cols[3].metric('Numeric', profile['numeric_columns'])
    cols[4].metric('Categorical', profile['categorical_columns'])

    if st.button('Run AutoML Pipeline', type='primary', use_container_width=True):
        run_automl()


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
        result = train_models(df, target_col, task_type, progress_callback=on_progress)
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

    tab_dist, tab_missing, tab_corr, tab_outliers, tab_stats, tab_signal = st.tabs([
        'Distributions', 'Missing Values', 'Correlation', 'Outliers', 'Stats Infographics', 'Feature Signal'
    ])

    with tab_dist:
        if selected_features:
            selected = selected_features
            for col in selected:
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(filtered_df, x=col, color=target_col if task_type == 'classification' else None, marginal='box'), use_container_width=True, config=PLOT_CONFIG)
                c2.plotly_chart(px.box(filtered_df, y=col, x=target_col if task_type == 'classification' else None), use_container_width=True, config=PLOT_CONFIG)
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
        st.dataframe(profile['stats_table'], use_container_width=True)
        if not profile['stats_table'].empty:
            chart(px.bar(profile['stats_table'].head(25), x='feature', y=['mean', 'median', 'skew', 'kurt'], barmode='group'))

    with tab_signal:
        signal = feature_signal_scores(df, target_col, task_type)
        st.dataframe(signal.head(30), use_container_width=True)
        chart(px.bar(signal.head(20), x='score', y='feature', orientation='h', title='Mutual Information Feature Signal'))


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

    st.dataframe(metrics, use_container_width=True)
    if 'runtime_table' in result and not result['runtime_table'].empty:
        st.subheader('Parallel Training Runtime')
        st.dataframe(result['runtime_table'], use_container_width=True)
        chart(px.bar(result['runtime_table'], x='model', y='runtime_seconds', color='status', title='Model Runtime Estimates'))
    plot_cols = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc'] if st.session_state.task_type == 'classification' else ['rmse', 'mae', 'r2']
    plot_cols = [col for col in plot_cols if col in ranked.columns]
    chart(px.bar(ranked, x='model', y=plot_cols, barmode='group', title='Model Ranking'))

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
    st.dataframe(filtered, use_container_width=True)

    task_type = st.session_state.task_type
    st.subheader(f'Focused Analysis: {focus_model}')
    if st.session_state.task_type == 'classification':
        show_classification_visuals(result['artifacts'][focus_model])
    else:
        show_regression_visuals(result['artifacts'][focus_model])

    with st.expander('Side-by-side model visualizations', expanded=False):
        for model_name in selected:
            artifact = result['artifacts'][model_name]
            st.subheader(model_name)
            if task_type == 'classification':
                show_classification_visuals(artifact)
            else:
                show_regression_visuals(artifact)

    profile = st.session_state.profile or dataset_profile(st.session_state.df, st.session_state.target_col)
    suggestions = improvement_suggestions(st.session_state.df, st.session_state.target_col, st.session_state.task_type, result['metrics'])
    html_path = generate_html_report('analysis_results/automl_report.html', 'uploaded_dataset', st.session_state.target_col, st.session_state.task_type, profile, filtered, suggestions)
    pdf_path = generate_pdf_report('analysis_results/automl_report.pdf', 'uploaded_dataset', st.session_state.target_col, st.session_state.task_type, profile, filtered, suggestions)
    report_html = Path(html_path).read_text(encoding='utf-8')
    st.download_button('Download Comparison Report (HTML)', report_html, 'model_comparison_report.html', 'text/html')
    with open(pdf_path, 'rb') as pdf_file:
        st.download_button('Download Comparison Report (PDF)', pdf_file, 'model_comparison_report.pdf', 'application/pdf')


def show_classification_visuals(artifact):
    y_true = artifact['y_test']
    y_pred = artifact['y_pred']
    y_proba = artifact['y_proba']
    cm = confusion_matrix(y_true, y_pred)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.imshow(cm, text_auto=True, title='Confusion Matrix', labels={'x': 'Predicted', 'y': 'Actual'}), use_container_width=True, config=PLOT_CONFIG)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        c2.plotly_chart(px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}, title='ROC Curve'), use_container_width=True, config=PLOT_CONFIG)
        chart(px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'}, title='Precision-Recall Curve'))
    else:
        c2.info('ROC/PR curves require binary classification probabilities.')


def show_regression_visuals(artifact):
    y_true = np.asarray(artifact['y_test'])
    y_pred = np.asarray(artifact['y_pred'])
    residuals = y_true - y_pred
    df_plot = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Residual': residuals})
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(df_plot, x='Actual', y='Predicted', trendline='ols', title='Predicted vs Actual'), use_container_width=True, config=PLOT_CONFIG)
    c2.plotly_chart(px.scatter(df_plot, x='Predicted', y='Residual', title='Residual Plot'), use_container_width=True, config=PLOT_CONFIG)
    chart(px.histogram(df_plot, x='Residual', nbins=30, title='Error Distribution'))


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
    shap_df, shap_error = shap_summary(result['best_pipeline'], st.session_state.df.drop(columns=[target_col]))
    if shap_df is not None:
        chart(px.bar(shap_df.head(20), x='shap_importance', y='feature', orientation='h'))
    else:
        st.warning(shap_error)

    st.subheader('Explainable AI: LIME')
    lime_df, lime_error = lime_summary(result['best_pipeline'], result['prepared'])
    if lime_df is not None:
        st.dataframe(lime_df, use_container_width=True)
        chart(px.bar(lime_df, x='contribution', y='feature_rule', orientation='h'))
    else:
        st.warning(lime_error)

    st.subheader('Model Stability')
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
    model_path = 'analysis_results/deployment/best_automl_model.joblib'
    if st.button('Export Model + FastAPI + Docker Template', type='primary'):
        export_model(result['best_pipeline'], model_path)
        files = generate_deployment_files('analysis_results/deployment')
        st.success('Deployment package generated in analysis_results/deployment')
        st.json({'model': model_path, **files})

    deployment_dir = Path('analysis_results/deployment')
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

    st.dataframe(history, use_container_width=True)
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
                'EDA',
                'Train',
                'Compare',
                'Improve',
                'Track',
                'Deploy',
            ],
        )
        if st.session_state.df is not None:
            st.caption(f"Dataset: {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")

    apply_theme()

    pages = {
        'Home': landing_page,
        'Upload': upload_page,
        'EDA': analysis_page,
        'Train': training_page,
        'Compare': comparison_page,
        'Improve': suggestions_page,
        'Track': tracking_page,
        'Deploy': deployment_page,
    }
    pages[page]()


if __name__ == '__main__':
    main()
