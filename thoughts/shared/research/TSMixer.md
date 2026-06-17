# PatchTSMixer Research Documentation
## Comprehensive Guide for Time Series Forecasting Implementation

**Research Date:** 2026-01-12
**Model:** PatchTSMixer (HuggingFace Transformers)
**Target Framework:** PyTorch
**Project Context:** DREAM-ML Time Series Forecasting Integration

---

## Executive Summary

PatchTSMixer is a lightweight MLP-Mixer-based model for time series forecasting that achieves state-of-the-art performance while being 2-3X more efficient than Transformer models. It uses a patching mechanism to segment time series data and employs multi-layer perceptrons for mixing across patches, channels, and features. The model supports both univariate and multivariate forecasting, transfer learning, and provides excellent reproducibility guarantees.

**Key Strengths:**
- 8-60% improvement over MLP models
- 1-2% improvement over Patch-Transformer with 2-3X less memory/compute
- Lightweight architecture (1-5MB model size)
- Compatible with HuggingFace Trainer API
- Supports CPU training (critical for DREAM-ML deployment)
- Strong reproducibility features
- Transfer learning capabilities

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [How PatchTSMixer Works](#how-patchtsmixer-works)
3. [Dependencies & Installation](#dependencies--installation)
4. [Core API & Configuration](#core-api--configuration)
5. [Data Input/Output Formats](#data-inputoutput-formats)
6. [Training Implementation](#training-implementation)
7. [Reproducibility Guide](#reproducibility-guide)
8. [Hyperparameter Reference](#hyperparameter-reference)
9. [Best Practices](#best-practices)
10. [Integration with DREAM-ML](#integration-with-dream-ml)

---

## Architecture Overview

### Model Structure

```
Input Time Series (batch, context_length, num_channels)
    ↓
[Patching Layer] → Splits into patches
    ↓
[Patch Embedding] → Linear projection to d_model dimensions
    ↓
[MLP-Mixer Blocks] × num_layers
    ├─ [Patch Mixing] → Mix across time patches
    ├─ [Channel Mixing] → Mix across variables
    ├─ [Feature Mixing] → Mix hidden features
    └─ [Gated Attention] → Prioritize important features
    ↓
[Prediction Head] → Linear layer
    ↓
Output Forecast (batch, prediction_length, num_channels)
```

### Key Components

1. **Patching Mechanism**
   - Divides time series into fixed-size patches
   - Reduces sequence length → faster training
   - Captures local temporal patterns
   - Non-overlapping or overlapping patches (controlled by `patch_stride`)

2. **MLP-Mixer Backbone**
   - Patch mixer: Learns inter-patch dependencies (temporal patterns)
   - Feature mixer: Learns hidden feature representations
   - Channel mixer: Learns inter-channel correlations (multivariate)
   - Residual connections for gradient flow

3. **Gated Attention**
   - Lightweight attention mechanism
   - Prioritizes important features
   - Adds minimal computational overhead
   - Can be disabled if not needed

4. **Channel Handling Modes**
   - **common_channel**: Channel-independent with implicit mixing (recommended for pretraining)
   - **mix_channel**: Explicit channel mixing (use when channel correlations are critical)

---

## How PatchTSMixer Works

### Mathematical Flow

Given a time series **X** of shape `(context_length, num_channels)`:

1. **Patching:**
   ```
   num_patches = (context_length - patch_length) / patch_stride + 1
   patches = reshape(X, (num_patches, patch_length, num_channels))
   ```

2. **Embedding:**
   ```
   embeddings = Linear(patches)  # Shape: (num_patches, d_model, num_channels)
   ```

3. **Mixing (for each layer):**
   ```
   # Patch mixing (along time dimension)
   x = LayerNorm(embeddings)
   x = MLP(x, expansion_factor) + embeddings

   # Feature mixing (along hidden dimension)
   x = LayerNorm(x)
   x = MLP(x, expansion_factor) + x

   # Gated attention (optional)
   x = GatedAttention(x)
   ```

4. **Forecasting Head:**
   ```
   # Aggregate patches
   aggregated = mean_pool(x)  # or use last patch

   # Project to forecast horizon
   forecast = Linear(aggregated, prediction_length * num_channels)
   forecast = reshape(forecast, (prediction_length, num_channels))
   ```

### Why Patching Works

- **Reduces computational complexity:** O(N²) → O((N/P)²) where P = patch_length
- **Captures local patterns:** Each patch contains local temporal structure
- **Better inductive bias:** Temporal locality is preserved
- **Enables longer context:** Can process longer sequences efficiently

---

## Dependencies & Installation

### Required Packages

```bash
# Core dependencies (add to requirements-base.txt)
torch>=2.0.0
transformers>=4.35.0
```

### Optional But Recommended

```bash
# For advanced data preprocessing (IBM TSFM toolkit)
git+https://github.com/IBM/tsfm.git

# For visualization and metrics
matplotlib>=3.5.0
scikit-learn>=1.3.0
```

### Version Compatibility

- **Python:** 3.11+ (DREAM-ML requirement)
- **PyTorch:** 2.0.0+ (for deterministic ops support)
- **Transformers:** 4.35.0+ (PatchTSMixer added in v4.35)
- **CUDA:** Not required (CPU training supported)

### Installation Commands

```bash
# Basic installation
pip install torch transformers

# With IBM toolkit for preprocessing
pip install git+https://github.com/IBM/tsfm.git

# Verify installation
python -c "from transformers import PatchTSMixerConfig; print('PatchTSMixer available')"
```

---

## Core API & Configuration

### Classes Overview

| Class | Purpose | Use Case |
|-------|---------|----------|
| `PatchTSMixerConfig` | Model configuration | Define architecture hyperparameters |
| `PatchTSMixerForPrediction` | Forecasting model | Direct multi-step ahead forecasting |
| `PatchTSMixerForTimeSeriesClassification` | Classification | Time series classification tasks |
| `PatchTSMixerForRegression` | Regression | Predict scalar values from time series |
| `PatchTSMixerForPretraining` | Masked pretraining | Self-supervised learning |
| `PatchTSMixerModel` | Base model | Custom head implementations |

### PatchTSMixerConfig Parameters

**Complete parameter reference with defaults and descriptions:**

#### Essential Parameters (Tier 1 - Must Expose to Users)

```python
from transformers import PatchTSMixerConfig

config = PatchTSMixerConfig(
    # --- Core Architecture ---
    context_length=512,              # Historical window size (REQUIRED)
    prediction_length=96,            # Forecast horizon (REQUIRED)
    num_input_channels=1,            # Number of variables (1=univariate, >1=multivariate)
    patch_length=8,                  # Size of each patch
    patch_stride=8,                  # Stride between patches (=patch_length for non-overlapping)

    # --- Model Capacity ---
    d_model=16,                      # Hidden dimension (16-64 typical)
    num_layers=8,                    # Number of mixer layers (3-15 typical)
    expansion_factor=2,              # MLP expansion factor (2-5 typical)

    # --- Regularization ---
    dropout=0.2,                     # Dropout rate (0.0-0.5)
    head_dropout=0.2,                # Dropout in prediction head (0.0-0.5)

    # --- Channel Handling ---
    mode="common_channel",           # "common_channel" or "mix_channel"

    # --- Preprocessing ---
    scaling="std",                   # "std", "mean", or None
)
```

#### Advanced Parameters (Tier 2 - Optional Advanced Toggle)

```python
config = PatchTSMixerConfig(
    # --- Attention Mechanism ---
    gated_attn=True,                 # Enable gated attention
    self_attn=False,                 # Enable self-attention across patches (expensive)
    self_attn_heads=1,               # Number of attention heads (if self_attn=True)
    use_positional_encoding=False,   # Positional encoding for attention
    positional_encoding_type="sincos", # "sincos" or "random"

    # --- Normalization ---
    norm_mlp="LayerNorm",            # "LayerNorm" or "BatchNorm"
    norm_eps=1e-5,                   # Epsilon for numerical stability

    # --- Distribution Head (for probabilistic forecasting) ---
    loss="mse",                      # "mse" or "nll" (negative log likelihood)
    distribution_output="student_t", # "student_t", "normal", "negative_binomial"
    num_parallel_samples=100,        # Samples for probabilistic forecast

    # --- Initialization ---
    init_std=0.02,                   # Weight initialization std dev
    post_init=False,                 # Use HuggingFace vs PyTorch init

    # --- Advanced Features ---
    prediction_channel_indices=None, # List of channels to forecast (None=all)
)
```

#### Masking Parameters (for Pretraining - Not Needed for Forecasting)

```python
config = PatchTSMixerConfig(
    mask_type="random",              # "random" or "forecast"
    random_mask_ratio=0.5,           # Ratio of patches to mask
    num_forecast_mask_patches=[2],   # Patches to mask at end
    mask_value=0.0,                  # Value for masked patches
    masked_loss=True,                # Compute loss only on masked patches
    channel_consistent_masking=True, # Same mask across channels
    unmasked_channel_indices=None,   # Channels never masked
)
```

---

## Data Input/Output Formats

### Input Format

PatchTSMixer expects data in the following format:

```python
# Shape: (batch_size, context_length, num_input_channels)
past_values = torch.FloatTensor([
    [[v1_t1, v2_t1, v3_t1],  # time step 1
     [v1_t2, v2_t2, v3_t2],  # time step 2
     ...
     [v1_tn, v2_tn, v3_tn]], # time step n
    # ... more samples in batch
])
```

**Key Requirements:**
- **Data type:** `torch.FloatTensor` or `torch.float32`
- **Shape:** `(batch_size, seq_length, num_channels)`
- **Univariate:** Set `num_input_channels=1`, data shape `(batch, context_length, 1)`
- **Multivariate:** Set `num_input_channels=N`, data shape `(batch, context_length, N)`

### Handling Missing Values

```python
# Create observed_mask (1=observed, 0=missing)
observed_mask = torch.ones_like(past_values)
observed_mask[past_values.isnan()] = 0

# Replace NaN with zeros (model ignores where observed_mask=0)
past_values[past_values.isnan()] = 0.0
```

### Output Format

```python
# Forecasting output shape: (batch_size, prediction_length, num_input_channels)
outputs = model(past_values=past_values)
predictions = outputs.prediction_outputs

# predictions shape: (batch, 96, num_channels) for prediction_length=96
```

### Data Preprocessing Requirements

1. **Normalization:** Built-in via `scaling` parameter
   - `scaling="std"`: Per-window standard scaling (recommended)
   - `scaling="mean"`: Per-window mean scaling
   - `scaling=None`: No scaling (not recommended)

2. **Missing Values:** Use `observed_mask` tensor

3. **Sequence Length:** Must equal `context_length` exactly

4. **Channel Count:** Must match `num_input_channels`

### Converting Pandas DataFrame to Tensors

```python
import pandas as pd
import torch

# Example DataFrame
df = pd.read_csv("data.csv", parse_dates=["date"])
# Columns: date, target, feature1, feature2

# Extract time series columns
ts_columns = ["target", "feature1", "feature2"]
context_length = 512

# Create sequences
sequences = []
for i in range(len(df) - context_length):
    seq = df[ts_columns].iloc[i:i+context_length].values
    sequences.append(seq)

# Convert to tensor
past_values = torch.FloatTensor(sequences)
# Shape: (num_sequences, context_length, num_channels)
```

---

## Training Implementation

### Basic Training Pipeline

```python
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
)
import torch

# 1. Configure model
config = PatchTSMixerConfig(
    context_length=512,
    prediction_length=96,
    num_input_channels=3,  # e.g., 3 features
    patch_length=8,
    patch_stride=8,
    d_model=32,
    num_layers=8,
    expansion_factor=2,
    dropout=0.2,
    head_dropout=0.2,
    mode="common_channel",
    scaling="std",
    loss="mse",
)

# 2. Initialize model
model = PatchTSMixerForPrediction(config)

# 3. Define training arguments
training_args = TrainingArguments(
    output_dir="./patchtsmixer_checkpoints",
    overwrite_output_dir=True,

    # Training config
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=0.001,

    # Evaluation
    do_eval=True,
    evaluation_strategy="epoch",

    # Checkpointing
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Reproducibility
    seed=42,
    data_seed=42,

    # Performance
    dataloader_num_workers=4,  # Adjust based on CPU cores

    # Early stopping (via callback)
    # See EarlyStoppingCallback below
)

# 4. Early stopping callback
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.001,
)

# 5. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping],
)

# 6. Train
trainer.train()

# 7. Evaluate
test_results = trainer.evaluate(test_dataset)
print(f"Test Loss: {test_results['eval_loss']}")

# 8. Save model
trainer.save_model("./final_model")
```

### Creating PyTorch Datasets

```python
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, past_values, future_values, observed_mask=None):
        """
        Args:
            past_values: Tensor of shape (num_samples, context_length, num_channels)
            future_values: Tensor of shape (num_samples, prediction_length, num_channels)
            observed_mask: Optional tensor of shape (num_samples, context_length, num_channels)
        """
        self.past_values = past_values
        self.future_values = future_values
        self.observed_mask = observed_mask if observed_mask is not None else torch.ones_like(past_values)

    def __len__(self):
        return len(self.past_values)

    def __getitem__(self, idx):
        item = {
            "past_values": self.past_values[idx],
            "future_values": self.future_values[idx],
            "observed_mask": self.observed_mask[idx],
        }
        return item

# Usage
train_dataset = TimeSeriesDataset(train_past, train_future)
valid_dataset = TimeSeriesDataset(valid_past, valid_future)
test_dataset = TimeSeriesDataset(test_past, test_future)
```

### Training Without Trainer API (Manual Loop)

```python
import torch
from torch.optim import AdamW
from torch.nn import MSELoss

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatchTSMixerForPrediction(config).to(device)
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = MSELoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in train_dataloader:
        past_values = batch["past_values"].to(device)
        future_values = batch["future_values"].to(device)
        observed_mask = batch["observed_mask"].to(device)

        # Forward pass
        outputs = model(
            past_values=past_values,
            future_values=future_values,
            observed_mask=observed_mask,
            return_loss=True,
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader):.4f}")
```

---

## Reproducibility Guide

### Complete Reproducibility Setup

To ensure 100% deterministic training across runs and platforms:

```python
import random
import numpy as np
import torch
import os
from transformers import set_seed

def set_global_reproducibility(seed=42):
    """
    Set all random seeds for full reproducibility.
    Based on PyTorch and HuggingFace best practices.
    """
    # 1. Python random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 5. CuDNN determinism (critical for GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. PyTorch deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # 7. Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic ops

    # 8. HuggingFace set_seed (handles Trainer internals)
    set_seed(seed)

    print(f"✓ Reproducibility configured with seed={seed}")

# Call before any model/data initialization
set_global_reproducibility(42)
```

### HuggingFace Trainer Reproducibility

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    # Seed settings
    seed=42,                    # Global seed
    data_seed=42,               # DataLoader seed

    # Deterministic DataLoader
    use_seedable_sampler=True,  # Ensures reproducible sampling
    dataloader_num_workers=0,   # Single-threaded (most deterministic)

    # Other settings...
)
```

### Model Initialization Reproducibility

```python
def model_init():
    """
    Model initialization function for Trainer.
    Ensures model is initialized with the same seed each time.
    """
    # Seed is already set globally
    config = PatchTSMixerConfig(...)
    model = PatchTSMixerForPrediction(config)
    return model

# Use with Trainer
trainer = Trainer(
    model_init=model_init,  # Instead of passing model directly
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
```

### Cross-Platform Considerations

**CPU vs GPU:**
- Results may differ between CPU and GPU due to floating-point arithmetic
- For true reproducibility, **always use the same device** (CPU for DREAM-ML)

**Different OS/Architectures:**
- Some atomic operations are non-deterministic across platforms
- PyTorch deterministic mode handles most cases
- Test on target deployment platform

**Known Limitations:**
- Some operations (e.g., scatter_add, index_put) don't have deterministic implementations
- If encounter errors, set `torch.use_deterministic_algorithms(False)` and accept minor variance

### Verification Script

```python
def verify_reproducibility(model, dataloader, device, num_runs=3):
    """
    Verify that model produces identical outputs across runs.
    """
    outputs_list = []

    for run in range(num_runs):
        # Reset seed
        set_global_reproducibility(42)

        # Initialize model
        model_test = model.__class__(model.config).to(device)
        model_test.load_state_dict(model.state_dict())
        model_test.eval()

        # Forward pass
        with torch.no_grad():
            batch = next(iter(dataloader))
            past_values = batch["past_values"].to(device)
            outputs = model_test(past_values=past_values)
            predictions = outputs.prediction_outputs

        outputs_list.append(predictions.cpu().numpy())

    # Check if all outputs are identical
    for i in range(1, num_runs):
        if not np.allclose(outputs_list[0], outputs_list[i], rtol=1e-5, atol=1e-8):
            print("❌ Reproducibility check FAILED")
            return False

    print(f"✓ Reproducibility verified across {num_runs} runs")
    return True
```

---

## Hyperparameter Reference

### Impact Analysis

| Hyperparameter | Impact on Performance | Impact on Memory | Impact on Training Time | Recommended Range |
|----------------|----------------------|------------------|-------------------------|-------------------|
| `context_length` | High (longer=more context) | High | High | 128-1024 |
| `prediction_length` | Medium (task-specific) | Low | Low | 24-192 |
| `num_input_channels` | High (data-dependent) | Medium | Medium | 1-50 |
| `patch_length` | Medium | Medium | Medium | 8-32 |
| `d_model` | High | High | High | 16-128 |
| `num_layers` | Very High | Medium | High | 3-15 |
| `expansion_factor` | Medium | High | Medium | 2-5 |
| `dropout` | Medium (regularization) | None | None | 0.0-0.5 |
| `batch_size` | Low | Very High | Medium | 16-128 |
| `learning_rate` | Very High | None | None | 0.0001-0.01 |

### Hyperparameter Tuning Strategy

**For CPU-constrained environments (DREAM-ML):**

1. **Start Conservative:**
   ```python
   context_length=512
   prediction_length=96
   patch_length=16        # Larger = fewer patches = faster
   d_model=16            # Small model
   num_layers=6          # Shallow network
   batch_size=32         # Balance speed vs memory
   learning_rate=0.001
   ```

2. **If Underfitting:**
   - Increase `d_model` (16 → 32 → 64)
   - Increase `num_layers` (6 → 8 → 12)
   - Increase `context_length` (if data available)
   - Decrease `dropout`

3. **If Overfitting:**
   - Increase `dropout` (0.2 → 0.3 → 0.4)
   - Decrease `d_model` or `num_layers`
   - Use early stopping
   - Reduce model capacity

4. **Memory Issues:**
   - Decrease `batch_size`
   - Decrease `context_length`
   - Increase `patch_length` (reduces num_patches)
   - Decrease `d_model`

### Sensible Defaults by Use Case

**Short-term forecasting (prediction_length ≤ 96):**
```python
context_length=512
patch_length=8
d_model=32
num_layers=8
```

**Long-term forecasting (prediction_length > 96):**
```python
context_length=1024
patch_length=16
d_model=64
num_layers=12
```

**Univariate (num_input_channels=1):**
```python
d_model=16
num_layers=6
mode="common_channel"
```

**High-dimensional multivariate (num_input_channels > 20):**
```python
d_model=64
num_layers=10
mode="mix_channel"  # Explicit channel mixing
```

---

## Best Practices

### 1. Data Preprocessing

**Do:**
- ✓ Always enable scaling (`scaling="std"`)
- ✓ Handle missing values with `observed_mask`
- ✓ Ensure `context_length` divides evenly by `patch_length`
- ✓ Use temporal train/val/test split (no shuffling)
- ✓ Normalize per-window (not global normalization)

**Don't:**
- ✗ Shuffle time series data
- ✗ Use different scaling on train/test
- ✗ Ignore missing values
- ✗ Use global standardization (breaks temporal structure)

### 2. Model Configuration

**Do:**
- ✓ Set `patch_stride = patch_length` (non-overlapping)
- ✓ Use `mode="common_channel"` for general use
- ✓ Start with smaller models and scale up
- ✓ Use `gated_attn=True` (default)
- ✓ Set `loss="mse"` for point forecasting

**Don't:**
- ✗ Enable `self_attn=True` unless necessary (expensive)
- ✗ Use very small `patch_length` (<4)
- ✗ Use `patch_length` > `context_length // 8`

### 3. Training

**Do:**
- ✓ Use early stopping (patience=10-20)
- ✓ Monitor validation loss every epoch
- ✓ Save best model checkpoint
- ✓ Log learning rate, loss curves
- ✓ Use gradient clipping if training unstable

**Don't:**
- ✗ Train without validation set
- ✗ Use too high learning rate (>0.01)
- ✗ Train for fixed epochs without early stopping
- ✗ Ignore validation metrics

### 4. CPU Training Optimization

**For DREAM-ML deployment:**

```python
# Optimize DataLoader
dataloader_num_workers=4  # Use 4-8 CPU cores
pin_memory=False          # CPU-only training
persistent_workers=True   # Reuse workers

# Batch size tuning
# Test with: 16, 32, 64
# Monitor: training speed, memory usage

# Mixed precision (CPU doesn't support, but document for future)
# fp16=False  # Not supported on CPU
# bf16=False  # Not supported on CPU
```

**Expected training times (CPU):**
- Small model (d_model=16, num_layers=6): ~2-5 min/epoch
- Medium model (d_model=32, num_layers=8): ~5-15 min/epoch
- Large model (d_model=64, num_layers=12): ~15-30 min/epoch

*(Times vary based on dataset size, context_length, and CPU)*

### 5. Reproducibility

**Critical checklist:**
- ✓ Call `set_global_reproducibility(seed)` before any imports
- ✓ Set `seed` and `data_seed` in TrainingArguments
- ✓ Use `model_init` function with Trainer
- ✓ Set `dataloader_num_workers=0` for maximum determinism
- ✓ Document exact versions (torch, transformers, Python)
- ✓ Test reproducibility on target platform

### 6. Model Saving & Loading

**Save model:**
```python
# Method 1: Via Trainer
trainer.save_model("./my_model")

# Method 2: Manual
model.save_pretrained("./my_model")
config.save_pretrained("./my_model")
```

**Load model:**
```python
# Load from saved directory
model = PatchTSMixerForPrediction.from_pretrained("./my_model")
config = PatchTSMixerConfig.from_pretrained("./my_model")
```

**MLflow Integration:**
```python
import mlflow.pytorch

# Log model
with mlflow.start_run():
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="patchtsmixer",
        registered_model_name="patchtsmixer_v1",
    )

    # Log config
    mlflow.log_dict(config.to_dict(), "config.json")

    # Log metrics
    mlflow.log_metrics({
        "test_mse": test_mse,
        "test_mae": test_mae,
    })
```

### 7. Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor performance | Model too small | Increase `d_model`, `num_layers` |
| Overfitting | Too complex for data | Increase `dropout`, use early stopping |
| OOM (Out of Memory) | Batch/model too large | Reduce `batch_size`, `d_model`, or `context_length` |
| Slow training | Too many patches | Increase `patch_length`, reduce `context_length` |
| Non-reproducible | Missing seed setup | Follow reproducibility guide completely |
| NaN loss | Learning rate too high | Reduce `learning_rate` (try 0.0001) |
| Unstable training | No gradient clipping | Add `max_grad_norm=1.0` to TrainingArguments |

---

## Integration with DREAM-ML

### File Structure

```
DREAM-ML-backend/GEML/apiTimeSeries/
├── train.py                    # Add train_patchtsmixer_model()
├── services.py                 # Update TrainModelService
├── data_encoding_utils.py      # (no changes needed)
├── data_cleaning_utils.py      # (no changes needed)
└── requirements-base.txt       # Add: torch, transformers

DREAM-ML-frontend/frontend/src/components/
└── TSTrainCard.jsx            # Add PatchTSMixer UI
```

### Backend Implementation Template

```python
# In train.py

def train_patchtsmixer_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Train PatchTSMixer model for time series forecasting.

    Args:
        dataset_path: Path to encoded CSV file
        data: Configuration dictionary with hyperparameters
        experiment_dir: Directory for saving outputs

    Returns:
        Dictionary with val_metrics, test_metrics, model_path
    """
    import torch
    from transformers import (
        PatchTSMixerConfig,
        PatchTSMixerForPrediction,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    # 1. Set reproducibility
    set_global_seeds()  # Use existing DREAM-ML function

    # 2. Load and prepare data
    df = pd.read_csv(dataset_path)
    date_col = data.get("date_col_name")
    target_col = data.get("target_variable")
    input_features = data.get("input_features", [])

    # Determine univariate vs multivariate
    if len(input_features) == 0:
        # Univariate
        ts_columns = [target_col]
        num_input_channels = 1
    else:
        # Multivariate
        ts_columns = input_features + [target_col]
        num_input_channels = len(ts_columns)

    # 3. Extract parameters
    manual_params = data.get("manual_params", {})
    context_length = manual_params.get("context_length", 512)
    prediction_length = data.get("forecast_horizon", 96)
    patch_length = manual_params.get("patch_length", 8)
    d_model = manual_params.get("d_model", 32)
    num_layers = manual_params.get("num_layers", 8)
    dropout = manual_params.get("dropout", 0.2)
    learning_rate = manual_params.get("learning_rate", 0.001)
    batch_size = manual_params.get("batch_size", 32)
    epochs = manual_params.get("epochs", 100)
    early_stopping_patience = manual_params.get("early_stopping_patience", 10)

    # 4. Create sequences
    train_past, train_future = create_sequences(
        df[ts_columns],
        context_length,
        prediction_length,
        split="train"
    )
    valid_past, valid_future = create_sequences(
        df[ts_columns],
        context_length,
        prediction_length,
        split="val"
    )
    test_past, test_future = create_sequences(
        df[ts_columns],
        context_length,
        prediction_length,
        split="test"
    )

    # 5. Create PyTorch datasets
    train_dataset = TimeSeriesDataset(train_past, train_future)
    valid_dataset = TimeSeriesDataset(valid_past, valid_future)
    test_dataset = TimeSeriesDataset(test_past, test_future)

    # 6. Configure model
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        num_input_channels=num_input_channels,
        patch_length=patch_length,
        patch_stride=patch_length,
        d_model=d_model,
        num_layers=num_layers,
        expansion_factor=2,
        dropout=dropout,
        head_dropout=dropout,
        mode="common_channel",
        scaling="std",
        loss="mse",
    )

    # 7. Initialize model
    model = PatchTSMixerForPrediction(config)

    # 8. Training arguments
    output_dir = os.path.join(experiment_dir, "patchtsmixer_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        data_seed=42,
        dataloader_num_workers=4,
    )

    # 9. Callbacks
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
    )

    # 10. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping],
    )

    # 11. Train with MLflow logging
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params({
            "context_length": context_length,
            "prediction_length": prediction_length,
            "num_input_channels": num_input_channels,
            "patch_length": patch_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        })

        # Train
        trainer.train()

        # Evaluate
        val_results = trainer.evaluate(valid_dataset)
        test_results = trainer.evaluate(test_dataset)

        # Log metrics
        mlflow.log_metrics({
            "val_loss": val_results["eval_loss"],
            "test_loss": test_results["eval_loss"],
        })

        # Save model
        model_path = os.path.join(experiment_dir, "patchtsmixer_model")
        trainer.save_model(model_path)

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "patchtsmixer")

    # 12. Calculate metrics (MSE, MAE, MAPE)
    predictions = trainer.predict(test_dataset).predictions
    test_targets = test_future.numpy()

    val_metrics = compute_metrics(trainer.predict(valid_dataset).predictions, valid_future.numpy())
    test_metrics = compute_metrics(predictions, test_targets)

    # 13. Return results
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
    }
```

### Frontend Configuration

**Essential Hyperparameters (Manual Mode):**

```javascript
// In TSTrainCard.jsx

const [patchTSMixerParams, setPatchTSMixerParams] = useState({
  context_length: 512,
  patch_length: 8,
  d_model: 32,
  num_layers: 8,
  dropout: 0.2,
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 100,
  early_stopping_patience: 10,
});
```

**Advanced Hyperparameters (Advanced Toggle):**

```javascript
const [patchTSMixerAdvanced, setPatchTSMixerAdvanced] = useState({
  expansion_factor: 2,
  head_dropout: 0.2,
  mode: "common_channel",  // or "mix_channel"
  gated_attn: true,
  self_attn: false,
  scaling: "std",
});
```

### Leveraging Existing Pipeline

**Good news:** Most of DREAM-ML's existing pipeline works as-is:

1. **Data cleaning:** No changes needed
2. **EDA:** No changes needed
3. **Feature engineering:** Lag features compatible (but not required)
4. **Train/val/test split:** Use existing `ts_train_val_test_split()`
5. **MLflow logging:** Use existing infrastructure
6. **DVC versioning:** Use existing `dvc add` workflow
7. **Metrics computation:** Use existing `compute_metrics()` function

**Only new requirement:** Convert DataFrame to PyTorch tensors (see code above)

---

## Additional Information

### Transfer Learning & Pretrained Models

PatchTSMixer supports transfer learning via HuggingFace Hub:

```python
# Load pretrained model
model = PatchTSMixerForPrediction.from_pretrained(
    "ibm/patchtsmixer-etth1-forecasting"
)

# Fine-tune on your dataset
trainer = Trainer(model=model, ...)
trainer.train()
```

**Available pretrained models:**
- `ibm/patchtsmixer-etth1-forecasting`
- `ibm/patchtsmixer-etth1-pretrain`
- `ibm-granite/granite-timeseries-patchtsmixer`

**Transfer learning strategies:**
1. **Zero-shot:** Evaluate pretrained model directly (no training)
2. **Linear probing:** Freeze backbone, train only head
3. **Full fine-tuning:** Train all parameters

### Probabilistic Forecasting

For uncertainty quantification:

```python
config = PatchTSMixerConfig(
    ...
    loss="nll",                      # Negative log likelihood
    distribution_output="student_t",  # Heavy-tailed distribution
    num_parallel_samples=100,        # Monte Carlo samples
)

# Predictions will be sampled from distribution
predictions = model.generate(past_values, num_samples=100)
# Shape: (batch, num_samples, prediction_length, num_channels)

# Compute quantiles
mean_forecast = predictions.mean(dim=1)
lower_quantile = predictions.quantile(0.1, dim=1)
upper_quantile = predictions.quantile(0.9, dim=1)
```

### Performance Benchmarks

**Electricity Dataset (univariate, prediction_length=96):**
- MSE: 0.128 (SOTA)
- Training time: ~30 min (GPU), ~4 hours (CPU)

**ETTh2 Dataset (multivariate, prediction_length=96):**
- Zero-shot MSE: 0.304
- Fine-tuned MSE: 0.273

**vs Other Models:**
- 8-60% better than MLP models
- 1-2% better than PatchTransformer
- 2-3X faster training and inference

### Known Issues & Limitations

1. **Patch length constraint:** `context_length` should be divisible by `patch_length`
2. **Fixed context:** Cannot change `context_length` after training
3. **CPU limitations:** No mixed precision training on CPU
4. **Memory usage:** Grows linearly with `num_input_channels` and `d_model`
5. **Self-attention:** Very expensive; use sparingly

---

## References & Sources

### Official Documentation
- [PatchTSMixer HuggingFace Docs](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)
- [PatchTSMixer Blog Post](https://huggingface.co/blog/patchtsmixer)
- [IBM Granite TSFM Tutorial](https://deepwiki.com/ibm-granite/granite-tsfm/4.2-patchtst-and-patchtsmixer-tutorial)

### Research Papers
- TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting ([arXiv:2306.09364](https://arxiv.org/abs/2306.09364))

### Reproducibility Resources
- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html)
- [HuggingFace Trainer Reproducibility](https://huggingface.co/docs/transformers/main_classes/trainer)

### Additional Resources
- [PyTorch Model Saving Guide](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [HuggingFace Training Tips](https://huggingface.co/docs/transformers/en/training)

---

## Implementation Checklist

- [ ] Install PyTorch (`pip install torch`)
- [ ] Install Transformers (`pip install transformers`)
- [ ] Add dependencies to `requirements-base.txt`
- [ ] Create `train_patchtsmixer_model()` in `train.py`
- [ ] Add TimeSeriesDataset class for PyTorch datasets
- [ ] Implement sequence creation function (DataFrame → Tensors)
- [ ] Update `TrainModelService.train_model_logic()` in `services.py`
- [ ] Add "patchtsmixer" case in backend views
- [ ] Create frontend UI in `TSTrainCard.jsx`
- [ ] Add essential hyperparameter inputs (10 params)
- [ ] Add advanced hyperparameter toggle (10 params)
- [ ] Test univariate forecasting
- [ ] Test multivariate forecasting
- [ ] Verify reproducibility (3+ runs with same seed)
- [ ] Create unit tests (similar to `test_lstm_phase2a.py`)
- [ ] Document in user guide
- [ ] Update API documentation

---

## Summary

PatchTSMixer is an excellent choice for DREAM-ML's time series forecasting needs:

**Strengths:**
- Lightweight and efficient (2-3X faster than Transformers)
- State-of-the-art performance (8-60% better than MLPs)
- CPU-friendly architecture
- Strong reproducibility guarantees
- Compatible with existing DREAM-ML pipeline
- HuggingFace Trainer integration (familiar API)

**Implementation Effort:**
- **Low:** Leverages existing pipeline (data cleaning, splits, MLflow, DVC)
- **Medium:** Need to add PyTorch tensor conversion
- **High:** Frontend UI matches existing LSTM complexity

**Recommendation:**
✅ **Proceed with implementation**. PatchTSMixer aligns perfectly with DREAM-ML requirements: CPU training, reproducibility, user customization, and integration with existing infrastructure.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-12
**Researched By:** Claude Sonnet 4.5
**Review Status:** Ready for Implementation
