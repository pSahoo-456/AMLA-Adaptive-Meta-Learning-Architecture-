"""
Image dataset characterization and analysis module
Provides image-based meta-features and CNN-based feature extraction
"""

import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import cv2
except ImportError:
    Image = None
    cv2 = None

# Defer heavy TensorFlow/Keras imports until CNN feature extraction is requested
TF_AVAILABLE = False


def validate_image_dataset(uploaded_files):
    """Validate uploaded image files and return metadata."""
    if not uploaded_files:
        return None, "No files uploaded"
    
    valid_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    
    image_metadata = []
    errors = []
    
    for file_obj in uploaded_files:
        file_path = Path(file_obj.name)
        
        if file_path.suffix.lower() not in valid_formats:
            errors.append(f"❌ {file_obj.name} - Unsupported format (use PNG, JPG, BMP, GIF, WebP)")
            continue
        
        try:
            img = Image.open(file_obj)
            img_array = np.array(img)
            
            metadata = {
                'filename': file_obj.name,
                'format': img.format,
                'width': img.width,
                'height': img.height,
                'size_bytes': len(file_obj.getvalue()),
                'channels': img_array.shape[2] if len(img_array.shape) > 2 else 1,
                'dtype': str(img_array.dtype),
                'status': 'ok'
            }
            image_metadata.append(metadata)
            file_obj.seek(0)  # Reset file pointer
        except Exception as e:
            errors.append(f"❌ {file_obj.name} - Error: {str(e)}")
    
    return image_metadata, errors


def extract_image_features(image_array, use_cnn=True, model_name='resnet50'):
    """Extract features from image using CNN or traditional methods."""
    
    if image_array is None:
        return None
    
    features = {}
    
    # Traditional hand-crafted features
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    features['brightness_mean'] = float(np.mean(gray))
    features['brightness_std'] = float(np.std(gray))
    features['contrast'] = float(np.max(gray) - np.min(gray))
    features['entropy'] = float(calculate_entropy(gray))
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    features['edge_density'] = float(np.sum(edges > 0) / edges.size)
    
    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    features['histogram_peaks'] = int(np.sum(hist > np.percentile(hist, 90)))
    
    # Texture features (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['texture_variance'] = float(laplacian_var)
    
    # CNN features (if available and requested)
    if use_cnn and TF_AVAILABLE:
        try:
            cnn_features = extract_cnn_features(image_array, model_name)
            features.update(cnn_features)
        except Exception as e:
            features['cnn_error'] = str(e)
    
    return features


def extract_cnn_features(image_array, model_name='resnet50'):
    """Extract deep features using pre-trained CNN."""
    try:
        # Lazy-import TensorFlow/Keras only when needed
        try:
            from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
        except Exception as e:
            return {'cnn_extraction_error': f'TensorFlow/Keras not available: {e}'}

        # Resize to model input size
        target_size = 224
        img_resized = cv2.resize(image_array, (target_size, target_size))
        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0

        # Select model
        if model_name == 'resnet50':
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif model_name == 'vgg16':
            model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        elif model_name == 'inception':
            model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        else:
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Extract features
        x = np.expand_dims(img_normalized, axis=0)
        features_vector = model.predict(x, verbose=0)[0]

        # Aggregate features to reduce dimensionality
        return {
            f'cnn_feat_{i}': float(features_vector[i]) 
            for i in [0, len(features_vector)//4, len(features_vector)//2, 3*len(features_vector)//4, -1]
        }
    except Exception as e:
        return {'cnn_extraction_error': str(e)}


def calculate_entropy(image_array):
    """Calculate image entropy as texture measure."""
    hist, _ = np.histogram(image_array, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy


def image_dataset_fingerprint(image_metadata_list):
    """Generate dataset-level fingerprint from multiple images."""
    
    if not image_metadata_list:
        return None
    
    df = pd.DataFrame(image_metadata_list)
    
    fingerprint = {
        'n_images': len(df),
        'avg_width': float(df['width'].mean()),
        'avg_height': float(df['height'].mean()),
        'avg_size_kb': float(df['size_bytes'].mean() / 1024),
        'width_std': float(df['width'].std()),
        'height_std': float(df['height'].std()),
        'aspect_ratio_mean': float((df['width'] / df['height']).mean()),
        'channels_mode': int(df['channels'].mode()[0]) if len(df['channels'].mode()) > 0 else 3,
        'format_distribution': df['format'].value_counts().to_dict(),
        'has_grayscale': int((df['channels'] == 1).sum()) > 0,
        'has_color': int((df['channels'] >= 3).sum()) > 0,
        'size_variability': float(df['size_bytes'].std() / df['size_bytes'].mean()) if df['size_bytes'].mean() > 0 else 0,
    }
    
    return fingerprint


def extract_dataset_features(image_metadata_list, feature_extraction_fn=None):
    """Extract features from all images in dataset."""
    
    all_features = []
    
    for metadata in image_metadata_list:
        try:
            # For now, use metadata-based features
            features = {
                'width': metadata['width'],
                'height': metadata['height'],
                'channels': metadata['channels'],
                'size_kb': metadata['size_bytes'] / 1024,
                'aspect_ratio': metadata['width'] / metadata['height'],
            }
            all_features.append(features)
        except Exception as e:
            print(f"Error extracting features from {metadata['filename']}: {e}")
    
    return pd.DataFrame(all_features) if all_features else None


def load_image_from_upload(uploaded_file):
    """Load PIL Image from Streamlit uploaded file."""
    try:
        img = Image.open(uploaded_file)
        return np.array(img), img
    except Exception as e:
        return None, str(e)


class ImageDataset:
    """Container for image dataset with metadata and features."""
    
    def __init__(self, image_files, labels=None):
        self.image_files = image_files
        self.labels = labels or []
        self.metadata = []
        self.features = None
        self.fingerprint = None
    
    def load_and_validate(self):
        """Load and validate all images."""
        self.metadata, errors = validate_image_dataset(self.image_files)
        return errors
    
    def extract_features(self, use_cnn=False):
        """Extract features from all images."""
        if not self.metadata:
            return None
        
        self.features = extract_dataset_features(self.metadata)
        self.fingerprint = image_dataset_fingerprint(self.metadata)
        return self.features
    
    def to_dataframe(self):
        """Convert to DataFrame format for compatibility."""
        if self.features is None:
            return None
        
        df = self.features.copy()
        if self.labels and len(self.labels) == len(df):
            df['label'] = self.labels
        
        return df
