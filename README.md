# AMLA - Adaptive Meta-Learning Architecture

## Overview

**AMLA** (Adaptive Meta-Learning Architecture) is a comprehensive intelligent framework designed to automate dataset characterization, predictive algorithm selection, and feature augmentation advice. This project was developed as a Major Project for the Bachelor of Technology degree in Computer Science and Engineering.

---

## 📚 Project Evaluation & Defense Documents
If you are evaluating this project, please refer to the following comprehensive guides:
- 📖 [End-to-End Project Explanation](PROJECT_EXPLANATION.md) - The high-level architecture and data flow.
- 🔬 [Technical Depth & Architecture](TECHNICAL_DEPTH.md) - A deep dive into the math, algorithms, and decoupled design.
- 🛡️ [Q&A Defense Guide](QNA_DEFENSE_GUIDE.md) - A cheat sheet for common panel questions.

---

## Key Features

### Core Components
- 🧬 **Dataset DNA Fingerprinting** - 21 meta-features across 4 layers
- 🎯 **Smart Algorithm Selection** - Random Forest meta-learner with 72% Precision@1
- 🔧 **Feature Health Advisor** - Detects 5 types of data quality issues
- 🧠 **Self-Improving System** - Meta-Knowledge Base with 50+ experiments
- 📊 **Explainable Recommendations** - Natural language justifications

### AutoML Web Platform
- **CSV and Excel upload** with target selection and automatic classification/regression detection
- **EDA dashboard** with histograms, boxplots, scatter matrices, correlation heatmaps, missing-value maps, outlier plots, and summary-stat infographics
- **Automatic model training and ranking** for classification and regression
- **Classification models:** Logistic Regression, Random Forest, SVM, Neural Network, XGBoost, LightGBM
- **Regression models:** Linear Regression, Random Forest, Gradient Boosting, SVR, Neural Network, XGBoost, LightGBM
- **Model visualizations** including confusion matrices, ROC/PR curves, residual plots, predicted-vs-actual plots, learning curves, validation curves, feature importance, and SHAP-style feature contribution summaries where supported
- **Improvement suggestions** for missing data, skewness, imbalance, low accuracy/R2, stability, and deployment readiness
- **Version tracking** for repeated uploads, with performance and dataset-change charts
- **Model export** to `analysis_results/best_automl_model.joblib`

### Advanced Analysis
- **20+ ML Metrics** - Comprehensive classification metrics
- **Publication-ready infographics** - Dataset overviews, model dashboards, confusion matrices, ROC/PR curves, calibration curves, learning curves, feature importance, per-class analysis, and comparison plots
- 🤖 **7 Algorithms Evaluated** - Compare multiple models
- 📈 **Learning Curves** - Bias-variance analysis
- 🎯 **Calibration Curves** - Probability quality assessment

### Interfaces
- 🌐 **Web Dashboard** - Modern React + Vite Web Application (Tailwind CSS)
- 📡 **REST API** - FastAPI backend
- 💻 **CLI Tool** - Command-line interface
- 🐍 **Python API** - Programmatic access

---

## 🎓 Project Information

**Institution:** GIFT Autonomous, Bhubaneswar  
**University:** Biju Patnaik University of Technology, Odisha  
**Batch:** 2022-2026

**Team:**
- Manohar Kumar Sah (2201298357)
- Prakash Sahoo (2201298366)

**Guide:** Mohapatra Girashree Sahu (Assistant Prof., Dept. of CSE)

---

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd FinalYr_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Start API and Web Dashboard

```bash
# Terminal 1: Backend API
python app.py

# Terminal 2: React Frontend
cd frontend
npm install
npm run dev
```

Open browser: **http://localhost:5173**.

The modern React dashboard is the primary full AutoML interface, communicating asynchronously with the FastAPI service for all heavy machine learning workloads.

### 3. Try Demo

```bash
python demo.py
```

### 4. Analyze Your Data

```bash
# Quick analysis
python main.py --mode analyze --file your_data.csv --target target

# Comprehensive analysis (95+ visualizations)
python main.py --mode comprehensive --file your_data.csv --target target

# Generate all available infographic types for every CSV in data/
python main.py --mode infographics --data-dir data --output-dir analysis_results
```

---

## 📂 Project Structure

```
FinalYr_project/
│
├── 📦 Core Package (amla/)
│   ├── characterizer.py         # 21 meta-features extraction
│   ├── metalearner.py          # Random Forest meta-learner
│   ├── feature_advisor.py      # Feature health analysis
│   ├── mkb.py                  # SQLite Meta-Knowledge Base
│   ├── pipeline.py             # Unified pipeline
│   ├── model_evaluator.py       # 20+ ML metrics
│   ├── visualizations.py        # 95+ visualizations
│   ├── comprehensive_analyzer.py # Complete analysis
│   └── utils.py                # Utilities
│
├── 🧪 Tests
│   └── test_amla.py            # Unit tests
│
├── 🌐 Web Interface
│   └── frontend/               # Modern React + Vite Application
│       ├── src/                # Components and Pages
│       └── package.json        # Node dependencies
│
├── 📡 API
│   └── app.py                 # FastAPI REST API
│
├── 💻 CLI Tools
│   ├── main.py                # CLI entry point
│   ├── demo.py                # Demonstration
│   └── seed_mkb.py            # Seed the meta-knowledge base
│
├── 🗄️ Data & Models
│   ├── data/                  # Sample datasets
│   └── models/                # Saved models
│
└── 📚 Documentation
    ├── README.md             # This file
    ├── LICENSE               # MIT License
    └── AGENTS.md             # Project instructions
```

---

## 🎯 Usage Examples

### 🌐 Web Dashboard (Recommended)

1. Open **http://localhost:8501**
2. Click **"Demo Dataset"** to try
3. Or upload your CSV file
4. View recommendations and visualizations

### 💻 Python API

```python
from amla import AMLAPipeline
import pandas as pd

# Load your data
df = pd.read_csv('your_dataset.csv')

# Create pipeline
pipeline = AMLAPipeline()

# Analyze
result = pipeline.run(df, target_col='target')

# Get recommendation
print(f"Best Algorithm: {result['algorithm_recommendation']['recommended_algorithm']}")
print(f"Confidence: {result['algorithm_recommendation']['confidence']:.1%}")
```

### 📡 REST API

```bash
# Start API
python app.py

# Analyze dataset
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@data.csv" \
  -F "target_column=target"
```

---

## 🏗️ Architecture

### System Flow

```
User Dataset
    ↓
Dataset Characterization Engine (DCE)
    ↓ [21 Meta-Features]
Meta-Learner (Random Forest)
    ↓
Algorithm Ranking + Confidence
    ↓
Feature Health Advisor
    ↓
Unified Recommendation Report
    ↓
Meta-Knowledge Base ← Feedback Loop
```

### Meta-Feature Layers

**Layer 1: Simple Statistics**
- Instances, Features, Missing values
- Classes, Class imbalance

**Layer 2: Distribution**
- Skewness, Kurtosis
- Variance, Correlation
- Outliers

**Layer 3: Information-Theoretic**
- Mutual information
- Feature redundancy
- Class entropy

**Layer 4: Landmarkers**
- Decision Stump accuracy
- Naive Bayes accuracy
- 1-NN accuracy
- Linear SVM accuracy

---

## Infographics for Research

For paper-ready visual evidence, run:

```bash
python main.py --mode infographics --data-dir data --output-dir analysis_results
```

The command scans every `.csv` file in `data/`, infers the target column by common names (`target`, `label`, `class`, `y`, `outcome`) or falls back to the last column, then writes per-dataset figures and an `infographic_summary.csv`.

Generated visual types include dataset overview, dataset statistics, confusion matrix, normalized confusion matrix, classification report, ROC curve, precision-recall curve, learning curve, feature importance when supported, prediction/error analysis, calibration, per-class analysis, model dashboard, and cross-model comparison plots.

For the full paper workflow, follow `RESEARCH_PROTOCOL.md`.

## Production Notes

The FastAPI backend reads `config.env` and environment variables for runtime settings:

```bash
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
MAX_UPLOAD_MB=25
```

The Streamlit UI reads:

```bash
AMLA_API_URL=http://localhost:8000
AMLA_API_TIMEOUT=120
```

For a public deployment, restrict `CORS_ORIGINS` to the exact dashboard origin instead of using `*`.

## Performance Metrics

The bundled meta-knowledge base and saved meta-learner support the reported AMLA workflow. Recompute project-specific figures with the benchmark and comprehensive commands before citing final paper numbers.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run demo
python demo.py
```

---

## 📈 Algorithms Evaluated

1. **Random Forest** - Ensemble of decision trees
2. **Gradient Boosting** - Sequential ensemble
3. **Logistic Regression** - Linear classifier
4. **SVM** - RBF kernel classifier
5. **K-Nearest Neighbors** - Instance-based
6. **Decision Tree** - Single tree
7. **Naive Bayes** - Probabilistic

---

## 🎓 Academic Value

This project demonstrates:
- ✅ Meta-learning concepts
- ✅ Algorithm selection theory
- ✅ Feature engineering
- ✅ Model evaluation
- ✅ AutoML principles
- ✅ Web development
- ✅ REST APIs
- ✅ Software engineering

---

## 🙏 Acknowledgments

- GIFT Autonomous, Bhubaneswar
- Biju Patnaik University of Technology
- OpenML Platform
- UCI Machine Learning Repository
- scikit-learn Community

---

## 📧 License

MIT License - See LICENSE file

Copyright (c) 2024 AMLA Project

---

## ⭐ Project Status

**Status:** Research-ready prototype

**Features Implemented:**
- ✅ Dataset DNA extraction
- ✅ Algorithm recommendation
- ✅ Feature health analysis
- ✅ Meta-knowledge base
- ✅ Web dashboard
- ✅ REST API
- ✅ CLI tool
- ✅ Comprehensive analysis
- ✅ 95+ visualizations
- ✅ 20+ metrics
- ✅ Documentation
- ✅ Unit tests

---

## 🚀 Future Enhancements

- Deep learning model selection
- Regression task support
- Time series analysis
- Automated hyperparameter tuning
- Graph neural network integration
- Uncertainty quantification

---

**Made with ❤️ for the ML Community**

*Version 1.0.0 | 2024*
