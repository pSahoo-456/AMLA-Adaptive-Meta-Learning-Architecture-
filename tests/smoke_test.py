import io
import os
import csv
from pathlib import Path

from amla import automl_engine
from amla import image_processor


def tabular_smoke_test():
    sample_csv = Path("data/sample_dataset.csv")
    out_dir = Path("analysis_results/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sample_csv.exists():
        print("SKIP: sample CSV not found:", sample_csv)
        return False

    print("Running tabular auto_fix_dataset on", sample_csv)
    import pandas as pd; df = pd.read_csv(str(sample_csv))
    cleaned, log = automl_engine.auto_fix_dataset(df, target_col=None, task_type=None, aggressive=False)
    out_path = out_dir / "cleaned_sample.csv"
    cleaned.to_csv(out_path, index=False)
    print("Wrote cleaned CSV to", out_path)
    return True


def image_smoke_test():
    out_dir = Path("analysis_results/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create a synthetic RGB image in-memory
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        print("SKIP: Pillow or numpy not available:", e)
        return False

    arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    print("Validating synthetic image via amla.image_processor")
    try:
        img_arr, img_obj = image_processor.load_image_from_upload(buf); img_meta = {"width": img_obj.width, "height": img_obj.height}
        print("Image metadata:", img_meta)
    except Exception as e:
        print("Image validation failed:", e)
        return False

    # Try feature extraction (may lazy-import cv/tf inside)
    try:
        feats = image_processor.extract_image_features(img_arr)
        out_path = out_dir / "image_features.csv"
        # feats assumed to be dict-like
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for k, v in feats.items():
                writer.writerow([k, v])
        print("Wrote image features to", out_path)
    except Exception as e:
        print("Image feature extraction skipped/failed:", e)
        return False

    return True


if __name__ == '__main__':
    ok_tab = tabular_smoke_test()
    ok_img = image_smoke_test()
    print("Tabular smoke test:", "OK" if ok_tab else "FAIL")
    print("Image smoke test:", "OK" if ok_img else "FAIL")
