"""
Fungsi-fungsi pembantu (utility).
"""
import time
import json
from typing import Dict
from functools import wraps
from pathlib import Path

def timer(func):
    """Decorator untuk mengukur waktu eksekusi sebuah fungsi."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Selesai: '{func.__name__}' dalam {run_time:.2f} detik")
        return result
    return wrapper

def save_metrics(metrics: Dict[str, float], path: Path):
    """Menyimpan metrik dalam format JSON."""
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrik berhasil disimpan -> {path}")
    except Exception as e:
        print(f"Gagal menyimpan metrik: {e}")