import time
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("⚡ Starting Week 3 Baseline Model Optimization Analysis on FULL Dataset...")

# 1. Load Processed Test Split
processed_test_path = "data/processed/test.csv"
if not os.path.exists(processed_test_path):
    print(f"❌ Error: Processed test data not found at {processed_test_path}. Run your ingestion pipeline first.")
    exit(1)

test_df = pd.read_csv(processed_test_path)

# CHANGED: Using the FULL dataset instead of sampling or slicing
texts = test_df["cleaned_text"].astype(str).tolist()
labels = test_df["sentiment"].tolist()

print(f"📊 Loaded {len(texts)} samples for full evaluation.")

# 2. Setup/Load Local Trained Model
MODEL_PATH = "models/baseline_model.pkl"

if os.path.exists(MODEL_PATH):
    print(f"📦 Loading locally trained baseline model from {MODEL_PATH}...")
    loaded_artifact = joblib.load(MODEL_PATH)
    
    # Robust unwrapping logic
    if isinstance(loaded_artifact, dict):
        if 'model' in loaded_artifact:
            baseline_pipeline = loaded_artifact['model']
        elif 'pipeline' in loaded_artifact:
            baseline_pipeline = loaded_artifact['pipeline']
        else:
            candidates = [v for v in loaded_artifact.values() if hasattr(v, 'predict')]
            if candidates:
                baseline_pipeline = candidates[0]
            else:
                raise AttributeError(f"Loaded dict does not contain a model. Keys found: {list(loaded_artifact.keys())}")
    else:
        baseline_pipeline = loaded_artifact
else:
    print(f"⚠️ Local model not found. Mocking an unoptimized pipeline...")
    mock_texts = ["great service", "terrible food", "loved the atmosphere", "bad experience"] * 250
    mock_labels = [1, 0, 1, 0] * 250
    baseline_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42))
    ])
    baseline_pipeline.fit(mock_texts, mock_labels)
    os.makedirs("models", exist_ok=True)
    joblib.dump(baseline_pipeline, MODEL_PATH)

# Calculate base file size
def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

baseline_size_mb = get_file_size_mb(MODEL_PATH)

# ==========================================
# EXPERIMENT 1: Baseline Sequential Inference
# ==========================================
print("\n--- Running Experiment 1: Unoptimized Sequential Inference (Baseline) ---")
seq_preds = []
start_time = time.perf_counter()

for text in texts:
    pred = baseline_pipeline.predict([text])[0]
    seq_preds.append(pred)

seq_total_time = time.perf_counter() - start_time
avg_seq_latency = (seq_total_time / len(texts)) * 1000  # ms per sample
seq_acc = accuracy_score(labels, seq_preds)

print(f"Total Time for {len(texts)} samples: {seq_total_time:.4f} seconds")
print(f"Average Latency: {avg_seq_latency:.2f} ms/sample")
print(f"Baseline Accuracy: {seq_acc:.4f}")

# ==========================================
# EXPERIMENT 2: Level 1 Software Optimization - Vectorized Batching
# ==========================================
print("\n--- Running Experiment 2: Vectorized Batched Inference (Level 1) ---")
start_time = time.perf_counter()

batch_preds = baseline_pipeline.predict(texts)

batch_total_time = time.perf_counter() - start_time
avg_batch_latency = (batch_total_time / len(texts)) * 1000
batch_acc = accuracy_score(labels, batch_preds)

print(f"Total Time for {len(texts)} samples: {batch_total_time:.4f} seconds")
print(f"Average Latency: {avg_batch_latency:.2f} ms/sample")
print(f"Throughput Speedup Factor: {seq_total_time / batch_total_time:.2f}x faster")
print(f"Batched Accuracy: {batch_acc:.4f} (Must be identical to baseline)")

# ==========================================
# EXPERIMENT 3: Level 2 Core Optimization - Vocabulary Pruning (Sparsity)
# ==========================================
print("\n--- Running Experiment 3: Feature Sparsity & Vocabulary Pruning (Level 2) ---")

pruned_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
    ('clf', LogisticRegression(random_state=42))
])

# Re-fitting a pruned version on the full dataset
pruned_pipeline.fit(texts, labels)

pruned_model_path = "models/baseline_pruned.pkl"
joblib.dump(pruned_pipeline, pruned_model_path)
pruned_size_mb = get_file_size_mb(pruned_model_path)

start_time = time.perf_counter()
pruned_preds = pruned_pipeline.predict(texts)
pruned_total_time = time.perf_counter() - start_time
avg_pruned_latency = (pruned_total_time / len(texts)) * 1000
pruned_acc = accuracy_score(labels, pruned_preds)

print(f"Total Time for {len(texts)} samples: {pruned_total_time:.4f} seconds")
print(f"Average Latency: {avg_pruned_latency:.2f} ms/sample")
print(f"Pruned Accuracy: {pruned_acc:.4f}")
print(f"Accuracy Variance vs Baseline: {pruned_acc - seq_acc:.4f}")

# ==========================================
# 4. GENERATE COMPARISON TABLE
# ==========================================
print("\n" + "="*70)
print("              WEEK 3 PERFORMANCE ANALYSIS MATRIX (BASELINE)")
print("="*70)
summary_data = {
    "Optimization State": ["Sequential Baseline", "Batched Inference (L1)", "Vocabulary Pruned (L2)"],
    "Model Size (MB)": [f"{baseline_size_mb:.3f} MB", f"{baseline_size_mb:.3f} MB", f"{pruned_size_mb:.3f} MB"],
    "Avg Latency / Sample": [f"{avg_seq_latency:.3f} ms", f"{avg_batch_latency:.3f} ms", f"{avg_pruned_latency:.3f} ms"],
    "Validation Accuracy": [f"{seq_acc:.4f}", f"{batch_acc:.4f}", f"{pruned_acc:.4f}"],
    "Size Reduction": ["0% (Baseline)", "0% (Baseline)", f"-{((baseline_size_mb - pruned_size_mb)/baseline_size_mb)*100:.1f}%"]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("="*70)