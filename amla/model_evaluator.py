import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, balanced_accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score,
    hinge_loss, brier_score_loss
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.figures = {}
        
    def evaluate_all(self, model, X, y, model_name, cv=5):
        self.results[model_name] = {}
        self.figures[model_name] = {}
        
        y_encoded = self._encode_labels(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        y_pred = cross_val_predict(model, X_scaled, y_encoded, cv=cv)
        y_pred_proba = self._get_proba_predictions(model, X_scaled, y_encoded, cv)
        
        self.results[model_name]['predictions'] = y_pred
        self.results[model_name]['y_true'] = y_encoded
        
        if y_pred_proba is not None:
            self.results[model_name]['proba'] = y_pred_proba
        
        self._compute_classification_metrics(model_name, y_encoded, y_pred, y_pred_proba)
        
        self._generate_confusion_matrix(model_name, y_encoded, y_pred)
        
        if y_pred_proba is not None and len(np.unique(y_encoded)) == 2:
            self._generate_roc_curve(model_name, y_encoded, y_pred_proba)
            self._generate_precision_recall_curve(model_name, y_encoded, y_pred_proba)
        
        self._generate_classification_report(model_name, y_encoded, y_pred)
        
        return self.results[model_name]
    
    def _encode_labels(self, y):
        le = LabelEncoder()
        return le.fit_transform(y)
    
    def _get_proba_predictions(self, model, X, y, cv):
        try:
            from sklearn.model_selection import cross_val_predict
            if hasattr(model, 'predict_proba'):
                proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
                return proba
            elif hasattr(model, 'decision_function'):
                decision = cross_val_predict(model, X, y, cv=cv, method='decision_function')
                if len(decision.shape) == 1:
                    proba = 1 / (1 + np.exp(-decision))
                else:
                    proba = decision
                return proba
        except:
            pass
        return None
    
    def _compute_classification_metrics(self, model_name, y_true, y_pred, y_proba):
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        n_classes = len(np.unique(y_true))
        
        if y_proba is not None:
            try:
                if n_classes == 2:
                    if y_proba.ndim == 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['brier_score'] = brier_score_loss(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                    metrics['log_loss'] = log_loss(y_true, y_proba)
                else:
                    metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        self.results[model_name]['metrics'] = metrics
    
    def _generate_confusion_matrix(self, model_name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        self.results[model_name]['confusion_matrix'] = cm
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        self.figures[model_name]['confusion_matrix'] = fig
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        ax2.set_title(f'{model_name} - Normalized Confusion Matrix')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        plt.tight_layout()
        self.figures[model_name]['confusion_matrix_normalized'] = fig2
        
        plt.close('all')
    
    def _generate_roc_curve(self, model_name, y_true, y_proba):
        if y_proba.ndim == 1:
            y_score = y_proba
        else:
            y_score = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        self.results[model_name]['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.figures[model_name]['roc_curve'] = fig
        
        plt.close('all')
    
    def _generate_precision_recall_curve(self, model_name, y_true, y_proba):
        if y_proba.ndim == 1:
            y_score = y_proba
        else:
            y_score = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        baseline = len(y_true[y_true==1]) / len(y_true)
        
        self.results[model_name]['pr_curve'] = {'precision': precision, 'recall': recall, 'auc': pr_auc}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.axhline(y=baseline, color='red', linestyle='--', lw=2, label=f'Baseline (Prevalence = {baseline:.3f})')
        ax.fill_between(recall, precision, alpha=0.3, color='blue')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - Precision-Recall Curve')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.figures[model_name]['precision_recall_curve'] = fig
        
        plt.close('all')
    
    def _generate_classification_report(self, model_name, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        self.results[model_name]['classification_report'] = report
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        table_data = []
        for cls in classes:
            table_data.append([
                cls,
                f"{report[cls]['precision']:.3f}",
                f"{report[cls]['recall']:.3f}",
                f"{report[cls]['f1-score']:.3f}",
                f"{int(report[cls]['support'])}"
            ])
        
        table_data.append(['', '', '', '', ''])
        table_data.append(['macro avg',
                          f"{report['macro avg']['precision']:.3f}",
                          f"{report['macro avg']['recall']:.3f}",
                          f"{report['macro avg']['f1-score']:.3f}",
                          f"{int(report['macro avg']['support'])}"])
        table_data.append(['weighted avg',
                          f"{report['weighted avg']['precision']:.3f}",
                          f"{report['weighted avg']['recall']:.3f}",
                          f"{report['weighted avg']['f1-score']:.3f}",
                          f"{int(report['weighted avg']['support'])}"])
        
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                        cellLoc='center',
                        loc='center',
                        colColours=['#4472C4']*5)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_text_props(color='white', fontweight='bold')
                elif i >= len(classes) + 1:
                    cell.set_facecolor('#E8E8E8')
                    cell.set_text_props(fontweight='bold')
        
        ax.set_title(f'{model_name} - Classification Report', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        self.figures[model_name]['classification_report'] = fig
        
        plt.close('all')
    
    def get_metrics_dataframe(self):
        records = []
        for model_name, result in self.results.items():
            if 'metrics' in result:
                record = {'Model': model_name}
                record.update(result['metrics'])
                records.append(record)
        
        return pd.DataFrame(records).sort_values('f1_weighted', ascending=False)
    
    def generate_comparison_plots(self):
        comparison_figures = {}
        
        metrics_df = self.get_metrics_dataframe()
        
        if len(metrics_df) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics_to_plot = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
            titles = ['Accuracy Comparison', 'F1-Score Comparison', 'Precision Comparison', 'Recall Comparison']
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            
            for idx, (metric, title, color) in enumerate(zip(metrics_to_plot, titles, colors)):
                ax = axes[idx // 2, idx % 2]
                bars = ax.barh(metrics_df['Model'], metrics_df[metric], color=color, alpha=0.8)
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_title(title)
                ax.set_xlim([0, 1.1])
                
                for bar, val in zip(bars, metrics_df[metric]):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)
                
                ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            comparison_figures['metrics_comparison'] = fig
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            models = metrics_df['Model'].values
            x = np.arange(len(models))
            width = 0.2
            
            metrics_bars = [
                ('precision_weighted', 'Precision'),
                ('recall_weighted', 'Recall'),
                ('f1_weighted', 'F1-Score')
            ]
            
            for i, (metric, label) in enumerate(metrics_bars):
                ax2.bar(x + i*width, metrics_df[metric], width, label=label, alpha=0.8)
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Score')
            ax2.set_title('Model Performance Comparison')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.legend()
            ax2.set_ylim([0, 1.1])
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            comparison_figures['bar_comparison'] = fig2
            
        plt.close('all')
        return comparison_figures
    
    def save_figure(self, model_name, figure_key, output_path):
        if model_name in self.figures and figure_key in self.figures[model_name]:
            self.figures[model_name][figure_key].savefig(output_path, dpi=300, bbox_inches='tight')
            return True
        return False
    
    def get_summary(self):
        summary = {
            'total_models': len(self.results),
            'best_model': None,
            'best_f1': 0,
            'all_metrics': self.get_metrics_dataframe()
        }
        
        for model_name, result in self.results.items():
            if 'metrics' in result:
                f1 = result['metrics'].get('f1_weighted', 0)
                if f1 > summary['best_f1']:
                    summary['best_f1'] = f1
                    summary['best_model'] = model_name
        
        return summary
