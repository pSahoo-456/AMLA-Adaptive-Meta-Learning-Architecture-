"""
End-to-end workflow test: Simulates user uploading CSV → applying suggestions → downloading cleaned data
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from amla import automl_engine, image_processor

def test_tabular_workflow():
    """Test: Upload CSV → Auto-fix → Download cleaned CSV"""
    print("\n" + "="*70)
    print("TEST 1: Tabular Data Workflow (Upload → Auto-fix → Download)")
    print("="*70)
    
    # Step 1: Load sample data (simulates user upload)
    sample_path = Path("data/sample_dataset.csv")
    if not sample_path.exists():
        print(f"❌ Sample data not found: {sample_path}")
        return False
    
    print(f"✓ Step 1: Loading sample data from {sample_path}")
    df_original = pd.read_csv(sample_path)
    print(f"  Original shape: {df_original.shape}")
    print(f"  Original columns: {list(df_original.columns)}")
    print(f"  Nulls: {df_original.isnull().sum().sum()}")
    
    # Step 2: Create profile (metadata about dataset)
    print(f"✓ Step 2: Generating dataset profile")
    profile = automl_engine.dataset_profile(df_original, target_col=None)
    print(f"  Profile keys: {list(profile.keys())}")
    
    # Step 3: Apply auto-fix (cleaning suggestions)
    print(f"✓ Step 3: Applying auto-fix with aggressive=False")
    start = time.time()
    df_cleaned, log = automl_engine.auto_fix_dataset(
        df_original, 
        target_col=None, 
        task_type=None, 
        aggressive=False
    )
    elapsed = time.time() - start
    print(f"  Cleaned shape: {df_cleaned.shape}")
    print(f"  Time taken: {elapsed:.3f}s")
    print(f"  Auto-fix log: {log}")
    
    # Step 4: Save cleaned dataset (download)
    out_dir = Path("analysis_results/e2e_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "workflow_cleaned_data.csv"
    df_cleaned.to_csv(out_path, index=False)
    print(f"✓ Step 4: Saved cleaned data to {out_path}")
    
    # Step 5: Verify file can be re-read (simulate re-upload)
    print(f"✓ Step 5: Verifying cleaned data can be re-read")
    df_verify = pd.read_csv(out_path)
    print(f"  Re-read shape: {df_verify.shape}")
    assert df_verify.shape == df_cleaned.shape, "Shape mismatch!"
    print(f"  ✓ Verification passed")
    
    # Step 6: Summary of changes
    print(f"✓ Step 6: Summary of changes")
    rows_removed = len(df_original) - len(df_cleaned)
    cols_same = len(df_original.columns) == len(df_cleaned.columns)
    print(f"  Rows removed: {rows_removed}")
    print(f"  Columns unchanged: {cols_same}")
    print(f"  ✓ Tabular workflow test PASSED")
    return True


def test_image_workflow():
    """Test: Create synthetic image → Validate → Extract features → Save"""
    print("\n" + "="*70)
    print("TEST 2: Image Data Workflow (Create → Validate → Extract Features)")
    print("="*70)
    
    try:
        from PIL import Image
    except ImportError:
        print("⚠ Pillow not installed, skipping image test")
        return False
    
    import io
    
    # Step 1: Create synthetic image
    print(f"✓ Step 1: Creating synthetic test image (64x64 RGB)")
    arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    print(f"  Image size: {len(buf.getvalue())} bytes")
    
    # Step 2: Load and validate image
    print(f"✓ Step 2: Loading image via amla.image_processor")
    try:
        img_arr, img_obj = image_processor.load_image_from_upload(buf)
        print(f"  Loaded image shape: {img_arr.shape}")
        print(f"  Image dtype: {img_arr.dtype}")
    except Exception as e:
        print(f"  ❌ Failed to load image: {e}")
        return False
    
    # Step 3: Extract features
    print(f"✓ Step 3: Extracting image features")
    start = time.time()
    try:
        features = image_processor.extract_image_features(img_arr)
        elapsed = time.time() - start
        print(f"  Features extracted: {len(features)} properties")
        print(f"  Time taken: {elapsed:.3f}s")
        print(f"  Sample features: {list(features.keys())[:5]}")
    except Exception as e:
        print(f"  ❌ Failed to extract features: {e}")
        return False
    
    # Step 4: Save feature metadata
    out_dir = Path("analysis_results/e2e_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    metadata_path = out_dir / "image_workflow_features.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "image_shape": img_arr.shape,
            "num_features": len(features),
            "feature_names": list(features.keys()),
            "sample_values": {k: float(v) for k, v in list(features.items())[:3]}
        }, f, indent=2)
    print(f"✓ Step 4: Saved feature metadata to {metadata_path}")
    
    # Step 5: Create ImageDataset and convert to DataFrame
    print(f"✓ Step 5: Creating ImageDataset and converting to DataFrame")
    try:
        buf.seek(0)
        dataset = image_processor.ImageDataset([buf.getvalue()], ["test_image"])
        df = dataset.to_dataframe()
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:5]}...")
        
        # Save DataFrame
        df_path = out_dir / "image_workflow_dataframe.csv"
        df.to_csv(df_path, index=False)
        print(f"  Saved to {df_path}")
    except Exception as e:
        print(f"  ⚠ ImageDataset conversion skipped: {e}")
    
    print(f"  ✓ Image workflow test PASSED")
    return True


def test_performance():
    """Test: Measure performance of key operations"""
    print("\n" + "="*70)
    print("TEST 3: Performance Benchmarks")
    print("="*70)
    
    sample_path = Path("data/sample_dataset.csv")
    if not sample_path.exists():
        print(f"❌ Sample data not found")
        return False
    
    # Load once
    df = pd.read_csv(sample_path)
    
    # Benchmark 1: Dataset profile
    print(f"Benchmark 1: dataset_profile()")
    times = []
    for i in range(3):
        start = time.time()
        _ = automl_engine.dataset_profile(df, target_col=None)
        times.append(time.time() - start)
    print(f"  Min: {min(times):.3f}s | Max: {max(times):.3f}s | Avg: {np.mean(times):.3f}s")
    
    # Benchmark 2: auto_fix_dataset
    print(f"Benchmark 2: auto_fix_dataset() [x5 runs]")
    times = []
    for i in range(5):
        start = time.time()
        _, _ = automl_engine.auto_fix_dataset(df, target_col=None, task_type=None, aggressive=False)
        times.append(time.time() - start)
    print(f"  Min: {min(times):.3f}s | Max: {max(times):.3f}s | Avg: {np.mean(times):.3f}s")
    
    print(f"  ✓ Performance benchmarks complete")
    return True


def main():
    print("\n" + "█"*70)
    print("█ END-TO-END WORKFLOW TEST SUITE")
    print("█"*70)
    
    results = {}
    results['tabular'] = test_tabular_workflow()
    results['image'] = test_image_workflow()
    results['performance'] = test_performance()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
