from amla import AMLAPipeline
import pandas as pd
import argparse
import sys
from pathlib import Path


def infer_target_column(df):
    """Prefer common target names, otherwise use the last column."""
    common_names = ('target', 'label', 'class', 'y', 'outcome')
    lower_to_original = {col.lower(): col for col in df.columns}
    for name in common_names:
        if name in lower_to_original:
            return lower_to_original[name]
    return df.columns[-1]


def run_all_infographics(data_dir, output_dir, cv_folds):
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))

    if not csv_files:
        print(f"No CSV datasets found in {data_path.resolve()}")
        return 1

    pipeline = AMLAPipeline()
    completed = []

    for csv_file in csv_files:
        print("\n" + "=" * 80)
        print(f"Generating infographics for {csv_file}")
        print("=" * 80)

        df = pd.read_csv(csv_file)
        target_col = infer_target_column(df)
        dataset_output_dir = Path(output_dir) / csv_file.stem
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        results = pipeline.run_comprehensive_analysis(
            df=df,
            target_col=target_col,
            output_dir=str(dataset_output_dir),
            cv=cv_folds
        )

        completed.append({
            'dataset': str(csv_file),
            'target': target_col,
            'output_dir': str(dataset_output_dir),
            'best_model': results.get('best_model'),
            'best_score': results.get('best_score'),
            'visualization_count': sum(len(v) for v in results.get('all_figures', {}).values())
                + len(results.get('visualizations', {})),
            'report': results.get('report_path')
        })

    summary_path = Path(output_dir) / 'infographic_summary.csv'
    pd.DataFrame(completed).to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("ALL DATASET INFOGRAPHICS COMPLETE")
    print("=" * 80)
    print(f"Datasets processed: {len(completed)}")
    print(f"Summary saved to: {summary_path}")
    return 0


def write_json_output(path, payload):
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(payload, f, indent=2, default=str)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='AMLA - Adaptive Meta-Learning Architecture')
    parser.add_argument('--mode', choices=['analyze', 'benchmark', 'seed', 'comprehensive', 'infographics'], default='analyze',
                        help='Operation mode')
    parser.add_argument('--file', type=str, help='Input CSV file path')
    parser.add_argument('--target', type=str, help='Target column name')
    parser.add_argument('--name', type=str, default='cli_dataset', help='Dataset name')
    parser.add_argument('--domain', type=str, default='general', help='Dataset domain')
    parser.add_argument('--model-path', type=str, default='models/metalearner.pkl',
                        help='Path to save/load meta-learner model')
    parser.add_argument('--mkb-path', type=str, default='data/mkb.db',
                        help='Path to Meta-Knowledge Base')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds for benchmarking')
    parser.add_argument('--json-output', type=str, help='Output JSON file path')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Output directory for comprehensive analysis')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing CSV datasets for infographic generation')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        if not args.file or not args.target:
            print("Error: --file and --target are required for analyze mode")
            sys.exit(1)
        
        print(f"Loading dataset from {args.file}...")
        df = pd.read_csv(args.file)
        print(f"Dataset shape: {df.shape}")
        
        pipeline = AMLAPipeline(model_path=args.model_path, mkb_path=args.mkb_path)
        
        print(f"Running AMLA analysis on '{args.target}' column...")
        result = pipeline.run(
            df=df,
            target_col=args.target,
            dataset_name=args.name,
            domain=args.domain,
            return_details=True
        )
        
        if result['status'] == 'success':
            print("\n" + "="*60)
            print("AMLA ANALYSIS RESULTS")
            print("="*60)
            
            rec = result['algorithm_recommendation']
            print(f"\nRecommended Algorithm: {rec['recommended_algorithm']}")
            print(f"Confidence: {rec['confidence']:.1%}")
            print(f"Method: {rec['method']}")
            
            print("\nRanked Algorithms:")
            for i, algo in enumerate(rec['ranked_algorithms'], 1):
                print(f"  {i}. {algo['algorithm']}: score={algo['combined_score']:.4f}")
            
            health = result['feature_analysis']['health_summary']
            print(f"\nFeature Health: {health['status'].upper()}")
            print(f"Issues found: {health['total_issues']} "
                  f"(High: {health['high_severity']}, "
                  f"Medium: {health['medium_severity']}, "
                  f"Low: {health['low_severity']})")
            
            if args.json_output:
                output_path = write_json_output(args.json_output, result)
                print(f"\nResults saved to {output_path}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    elif args.mode == 'benchmark':
        if not args.file or not args.target:
            print("Error: --file and --target are required for benchmark mode")
            sys.exit(1)
        
        print(f"Loading dataset from {args.file}...")
        df = pd.read_csv(args.file)
        print(f"Dataset shape: {df.shape}")
        
        pipeline = AMLAPipeline(model_path=args.model_path, mkb_path=args.mkb_path)
        
        print(f"Running benchmark with {args.cv_folds}-fold cross-validation...")
        results = pipeline.run_benchmark_algorithms(
            df=df,
            target_col=args.target,
            cv_folds=args.cv_folds
        )
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        for algo_name in ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN']:
            if algo_name in results:
                r = results[algo_name]
                print(f"\n{algo_name}:")
                print(f"  F1 Score:   {r['f1']:.4f} ± {r['f1_std']:.4f}")
                print(f"  Accuracy:   {r['accuracy']:.4f}")
                print(f"  Precision:  {r['precision']:.4f}")
                print(f"  Recall:     {r['recall']:.4f}")
        
        if 'best_algorithm' in results:
            print(f"\nBest Algorithm: {results['best_algorithm']} (F1={results['winning_score']:.4f})")
        
        if args.json_output:
            output_path = write_json_output(args.json_output, results)
            print(f"\nResults saved to {output_path}")
    
    elif args.mode == 'seed':
        print("Seeding Meta-Knowledge Base...")
        from seed_mkb import seed_mkb
        mkb = seed_mkb(target_count=50)
        print(f"\nMKB seeded successfully. Total experiments: {mkb.get_experiment_count()}")
        
        print("\nTraining meta-learner...")
        pipeline = AMLAPipeline(model_path=args.model_path, mkb_path=args.mkb_path)
        retrain_result = pipeline.retrain()
        print(f"Meta-learner training: {retrain_result['status']}")
    
    elif args.mode == 'comprehensive':
        if not args.file or not args.target:
            print("Error: --file and --target are required for comprehensive mode")
            sys.exit(1)
        
        print(f"Loading dataset from {args.file}...")
        df = pd.read_csv(args.file)
        print(f"Dataset shape: {df.shape}")
        
        pipeline = AMLAPipeline(model_path=args.model_path, mkb_path=args.mkb_path)
        
        print(f"\nRunning comprehensive analysis...")
        print("This will train 7 algorithms and generate 50+ visualizations")
        print(f"Output directory: {args.output_dir}")
        print()
        
        results = pipeline.run_comprehensive_analysis(
            df=df,
            target_col=args.target,
            output_dir=args.output_dir,
            cv=args.cv_folds
        )
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nBest Model: {results['best_model']}")
        print(f"Best F1-Score: {results['best_score']:.4f}")
        print(f"\nGenerated {len(results['all_figures'])} models with multiple visualizations each")
        print(f"Total visualizations: {sum(len(v) for v in results['all_figures'].values())}")
        print(f"\nOutput directory: {args.output_dir}")
        print(f"Report: {results['report_path']}")
        
        if args.json_output:
            output_path = write_json_output(args.json_output, results)
            print(f"\nResults saved to {output_path}")

    elif args.mode == 'infographics':
        exit_code = run_all_infographics(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cv_folds=args.cv_folds
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
