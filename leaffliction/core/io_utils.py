"""I/O utilities for caching and file operations."""

import json
import hashlib
import os

CACHE_FILE = "accuracy_cache.json"


def get_cache_key(model_path: str, data_dir: str) -> str:
    """Generate a unique cache key based on model and data directory."""
    # Get modification times
    model_mtime = os.path.getmtime(model_path)
    data_mtime = max(
        os.path.getmtime(os.path.join(root, f))
        for root, _, files in os.walk(data_dir)
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ) if os.path.exists(data_dir) else 0
    
    # Create hash from paths and modification times
    cache_str = f"{model_path}:{model_mtime}:{data_dir}:{data_mtime}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def load_cached_results(cache_key: str):
    """Load cached accuracy results if they exist."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        if cache_key in cache:
            print("📦 Loading cached accuracy results...")
            return cache[cache_key]
    except (json.JSONDecodeError, KeyError):
        pass
    
    return None


def save_cached_results(cache_key: str, overall_acc: float, class_accs: dict, class_names: list, conf_matrix: list = None):
    """Save accuracy results to cache."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            pass
    
    cache[cache_key] = {
        'overall_accuracy': overall_acc,
        'class_accuracies': class_accs,
        'class_names': class_names,
        'confusion_matrix': conf_matrix
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print("💾 Results cached for future use")
