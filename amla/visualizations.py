import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class VisualizationGenerator:
    def __init__(self):
        self.figures = {}
        
    def generate_dataset_overview(self, df, target_col):
        figures = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(df[target_col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        class_counts = df[target_col].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Class Proportions')
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            sample_cols = numerical_cols[:min(5, len(numerical_cols))]
            df[sample_cols].hist(ax=axes[1, 0], bins=20, color='coral', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Feature Distributions (Sample)')
        
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
        if len(missing_counts) > 0:
            axes[1, 1].barh(missing_counts.index, missing_counts.values, color='indianred')
            axes[1, 1].set_xlabel('Missing Count')
            axes[1, 1].set_title('Missing Values by Column')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Missing Values')
        
        plt.tight_layout()
        figures['dataset_overview'] = fig
        
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        
        axes2[0, 0].bar(['Instances', 'Features'], [len(df), len(df.columns)-1],
                       color=['steelblue', 'coral'], edgecolor='black')
        axes2[0, 0].set_ylabel('Count')
        axes2[0, 0].set_title('Dataset Size')
        for i, v in enumerate([len(df), len(df.columns)-1]):
            axes2[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        axes2[0, 0].grid(axis='y', alpha=0.3)
        
        dtypes_counts = df.dtypes.value_counts()
        axes2[0, 1].bar(dtypes_counts.index.astype(str), dtypes_counts.values,
                        color='mediumseagreen', edgecolor='black')
        axes2[0, 1].set_xlabel('Data Type')
        axes2[0, 1].set_ylabel('Count')
        axes2[0, 1].set_title('Data Types Distribution')
        axes2[0, 1].grid(axis='y', alpha=0.3)
        
        corr_matrix = df[numerical_cols].corr() if len(numerical_cols) > 0 else pd.DataFrame()
        if len(corr_matrix) > 1:
            sns.heatmap(corr_matrix, ax=axes2[1, 0], cmap='coolwarm', center=0,
                       annot=False, square=True, cbar_kws={'shrink': 0.8})
            axes2[1, 0].set_title('Feature Correlation Matrix')
        else:
            axes2[1, 0].text(0.5, 0.5, 'Insufficient features for correlation',
                             ha='center', va='center')
            axes2[1, 0].set_title('Feature Correlation Matrix')
        
        variance = df[numerical_cols].var() if len(numerical_cols) > 0 else pd.Series()
        if len(variance) > 0:
            top_var = variance.sort_values(ascending=False)[:10]
            axes2[1, 1].barh(top_var.index, top_var.values, color='orchid')
            axes2[1, 1].set_xlabel('Variance')
            axes2[1, 1].set_title('Top 10 Feature Variances')
            axes2[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['dataset_statistics'] = fig2
        
        plt.close('all')
        return figures
    
    def generate_learning_curve(self, model, X, y, model_name, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='f1_weighted', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                       alpha=0.1, color='orange')
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.plot(train_sizes_abs, test_mean, 'o-', color='orange', label='Cross-validation Score')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{model_name} - Learning Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        result = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'test_mean': test_mean.tolist(),
            'test_std': test_std.tolist()
        }
        
        plt.close('all')
        return fig, result
    
    def generate_feature_importance(self, model, feature_names, model_name, top_n=15):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - Feature Importance (Top {top_n})')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_).flatten()
            if len(coefs) > 0:
                indices = np.argsort(coefs)[::-1][:top_n]
                ax.barh(range(len(indices)), coefs[indices], color='coral', alpha=0.8)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f'Feature {i}'
                                   for i in indices])
                ax.set_xlabel('Absolute Coefficient')
                ax.set_title(f'{model_name} - Feature Coefficients (Top {top_n})')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.close('all')
        return fig
    
    def generate_prediction_analysis(self, y_true, y_pred, y_proba, model_name):
        figures = {}
        n_classes = len(np.unique(y_true))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if y_proba is not None and y_proba.ndim > 1:
            proba_pred = y_proba.max(axis=1)
        else:
            proba_pred = y_proba if y_proba is not None else np.ones(len(y_true)) * 0.5
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, c='steelblue', s=50)
        axes[0].plot([0, n_classes-1], [0, n_classes-1], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Class')
        axes[0].set_ylabel('Predicted Class')
        axes[0].set_title(f'{model_name} - Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(proba_pred, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0.5, color='red', linestyle='--', lw=2, label='Decision Threshold')
        axes[1].set_xlabel('Prediction Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Prediction Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['prediction_analysis'] = fig
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        
        correct = y_true == y_pred
        axes2[0].hist(proba_pred[correct], bins=30, alpha=0.7, label='Correct', color='green')
        axes2[0].hist(proba_pred[~correct], bins=30, alpha=0.7, label='Incorrect', color='red')
        axes2[0].set_xlabel('Prediction Probability')
        axes2[0].set_ylabel('Frequency')
        axes2[0].set_title(f'{model_name} - Correct vs Incorrect Predictions')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        if n_classes == 2:
            thresholds = np.linspace(0, 1, 50)
            accuracies = []
            for t in thresholds:
                if y_proba is not None and y_proba.ndim > 1:
                    preds = (y_proba[:, 1] >= t).astype(int)
                else:
                    preds = (y_proba >= t).astype(int)
                accuracies.append((preds == y_true).mean())
            
            axes2[1].plot(thresholds, accuracies, 'b-', lw=2)
            axes2[1].axvline(x=0.5, color='red', linestyle='--', lw=2, label='Default Threshold')
            best_threshold = thresholds[np.argmax(accuracies)]
            axes2[1].axvline(x=best_threshold, color='green', linestyle='--', lw=2,
                           label=f'Best Threshold ({best_threshold:.2f})')
            axes2[1].set_xlabel('Classification Threshold')
            axes2[1].set_ylabel('Accuracy')
            axes2[1].set_title(f'{model_name} - Threshold vs Accuracy')
            axes2[1].legend()
            axes2[1].grid(True, alpha=0.3)
        else:
            axes2[1].text(0.5, 0.5, 'Multi-class Problem\nThreshold analysis not applicable',
                        ha='center', va='center', fontsize=12)
            axes2[1].set_title('Threshold Analysis')
        
        plt.tight_layout()
        figures['error_analysis'] = fig2
        
        plt.close('all')
        return figures
    
    def generate_multiclass_roc_curves(self, y_true, y_proba, model_name, classes):
        n_classes = len(classes)
        
        if y_proba is None or n_classes <= 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        y_true_binary = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            y_true_binary[i, label] = 1
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, (color, cls) in enumerate(zip(colors, classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'Class {cls} (AUC = {roc_auc[i]:.3f})')
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        ax.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
               label=f'Macro-average (AUC = {np.mean(list(roc_auc.values())):.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - One-vs-Rest ROC Curves')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.close('all')
        return fig
    
    def generate_calibration_curve(self, model, X, y, model_name, cv=5):
        from sklearn.model_selection import cross_val_predict
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if hasattr(model, 'predict_proba'):
            y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
            
            if y_proba.shape[1] == 2:
                prob_pos = y_proba[:, 1]
                fraction_positives, mean_predicted_value = calibration_curve(
                    y, prob_pos, n_bins=10
                )
                
                axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
                axes[0].plot(mean_predicted_value, fraction_positives, 's-', color='steelblue',
                           lw=2, markersize=8, label=model_name)
                axes[0].set_ylabel('Fraction of Positives')
                axes[0].set_xlabel('Mean Predicted Probability')
                axes[0].set_title(f'{model_name} - Calibration Curve')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].hist(prob_pos[y == 0], bins=20, alpha=0.5, label='Negative', color='red')
                axes[1].hist(prob_pos[y == 1], bins=20, alpha=0.5, label='Positive', color='green')
                axes[1].set_xlabel('Predicted Probability')
                axes[1].set_ylabel('Count')
                axes[1].set_title(f'{model_name} - Probability Distribution')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                for i in range(y_proba.shape[1]):
                    prob_pos = y_proba[:, i]
                    fraction_positives, mean_predicted_value = calibration_curve(
                        (y == i).astype(int), prob_pos, n_bins=10
                    )
                    axes[0].plot(mean_predicted_value, fraction_positives, 's-',
                               lw=2, markersize=8, label=f'Class {i}')
                
                axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
                axes[0].set_ylabel('Fraction of Positives')
                axes[0].set_xlabel('Mean Predicted Probability')
                axes[0].set_title(f'{model_name} - Calibration Curves (Multi-class)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].text(0.5, 0.5, 'Probability distributions\nshown per class',
                           ha='center', va='center', fontsize=12)
                axes[1].set_title('Probability Distribution')
        else:
            axes[0].text(0.5, 0.5, f'{model_name} does not support\nprobability predictions',
                        ha='center', va='center', fontsize=12)
            axes[0].set_title('Calibration Curve')
            axes[1].text(0.5, 0.5, 'Calibration not available',
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('Probability Distribution')
        
        plt.tight_layout()
        plt.close('all')
        return fig
    
    def generate_per_class_analysis(self, y_true, y_pred, model_name, classes):
        n_classes = len(classes)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        per_class_metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for cls in classes:
            mask = y_true == cls
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics['precision'].append(precision)
            per_class_metrics['recall'].append(recall)
            per_class_metrics['f1'].append(f1)
        
        x = np.arange(n_classes)
        width = 0.25
        
        axes[0].bar(x - width, per_class_metrics['precision'], width, label='Precision',
                   color='steelblue', alpha=0.8)
        axes[0].bar(x, per_class_metrics['recall'], width, label='Recall',
                   color='coral', alpha=0.8)
        axes[0].bar(x + width, per_class_metrics['f1'], width, label='F1-Score',
                   color='mediumseagreen', alpha=0.8)
        
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Score')
        axes[0].set_title(f'{model_name} - Per-Class Performance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(axis='y', alpha=0.3)
        
        support = [np.sum(y_true == cls) for cls in classes]
        axes[1].bar(classes, support, color='orchid', alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'{model_name} - Class Distribution')
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(support):
            axes[1].text(i, v + max(support)*0.01, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.close('all')
        return fig
    
    def generate_comprehensive_dashboard(self, model_results, model_name):
        figures = {}
        
        fig = plt.figure(figsize=(20, 16))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, :])
        
        metrics = model_results.get('metrics', {})
        metric_names = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        bars = ax1.bar(metric_labels, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                      alpha=0.8, edgecolor='black')
        ax1.set_ylim([0, 1.1])
        ax1.set_ylabel('Score')
        ax1.set_title('Key Metrics', fontweight='bold')
        for bar, val in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        if 'confusion_matrix' in model_results:
            cm = model_results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
            ax2.set_title('Confusion Matrix', fontweight='bold')
            ax2.set_ylabel('Actual')
            ax2.set_xlabel('Predicted')
        
        if 'roc_curve' in model_results:
            roc = model_results['roc_curve']
            ax3.plot(roc['fpr'], roc['tpr'], 'b-', lw=2,
                    label=f'AUC = {roc["auc"]:.3f}')
            ax3.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
            ax3.fill_between(roc['fpr'], roc['tpr'], alpha=0.3)
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
        
        if 'pr_curve' in model_results:
            pr = model_results['pr_curve']
            ax4.plot(pr['recall'], pr['precision'], 'g-', lw=2,
                    label=f'AUC = {pr["auc"]:.3f}')
            ax4.fill_between(pr['recall'], pr['precision'], alpha=0.3, color='green')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve', fontweight='bold')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        
        additional_metrics = ['mcc', 'cohen_kappa', 'balanced_accuracy']
        additional_labels = ['MCC', 'Cohen\'s Kappa', 'Balanced Acc']
        additional_values = [metrics.get(m, 0) for m in additional_metrics]
        
        bars2 = ax5.bar(additional_labels, additional_values,
                       color=['#636E72', '#B2BEC3', '#DFE6E9'], alpha=0.8, edgecolor='black')
        ax5.set_ylim([0, 1.1])
        ax5.set_ylabel('Score')
        ax5.set_title('Additional Metrics', fontweight='bold')
        for bar, val in zip(bars2, additional_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        if 'classification_report' in model_results:
            report = model_results['classification_report']
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            x = np.arange(len(classes))
            width = 0.25
            
            precisions = [report[c]['precision'] for c in classes]
            recalls = [report[c]['recall'] for c in classes]
            f1s = [report[c]['f1-score'] for c in classes]
            
            ax6.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.8)
            ax6.bar(x, recalls, width, label='Recall', color='coral', alpha=0.8)
            ax6.bar(x + width, f1s, width, label='F1', color='mediumseagreen', alpha=0.8)
            ax6.set_xticks(x)
            ax6.set_xticklabels(classes)
            ax6.set_xlabel('Class')
            ax6.set_ylabel('Score')
            ax6.set_title('Per-Class Metrics', fontweight='bold')
            ax6.legend()
            ax6.set_ylim([0, 1.1])
            ax6.grid(axis='y', alpha=0.3)
        
        summary_text = """
        ╔═══════════════════════════════════════════════════════════════════╗
        ║                    MODEL EVALUATION SUMMARY                       ║
        ╠═══════════════════════════════════════════════════════════════════╣
        ║  Primary Metrics:                                                    ║
        ║    • Accuracy:        {:.4f}                                        ║
        ║    • F1-Score (Wtd):  {:.4f}                                        ║
        ║    • Precision (Wtd):  {:.4f}                                        ║
        ║    • Recall (Wtd):    {:.4f}                                        ║
        ╠═══════════════════════════════════════════════════════════════════╣
        ║  Advanced Metrics:                                                    ║
        ║    • ROC-AUC:         {:.4f}                                        ║
        ║    • Log Loss:        {:.4f}                                        ║
        ║    • MCC:             {:.4f}                                        ║
        ║    • Cohen's Kappa:   {:.4f}                                        ║
        ╚═══════════════════════════════════════════════════════════════════╝
        """.format(
            metrics.get('accuracy', 0),
            metrics.get('f1_weighted', 0),
            metrics.get('precision_weighted', 0),
            metrics.get('recall_weighted', 0),
            metrics.get('roc_auc', 0),
            metrics.get('log_loss', 0),
            metrics.get('mcc', 0),
            metrics.get('cohen_kappa', 0)
        )
        
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax7.axis('off')
        ax7.set_title('Summary', fontweight='bold')
        
        plt.suptitle(f'Comprehensive Analysis: {model_name}', fontsize=16, fontweight='bold', y=1.02)
        
        figures['comprehensive_dashboard'] = fig
        
        plt.close('all')
        return figures
