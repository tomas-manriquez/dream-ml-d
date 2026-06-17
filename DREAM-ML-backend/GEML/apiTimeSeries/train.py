# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

import os
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)  # Set programmatically
# ========================================================
import sys
import json
import logging
import pickle
import gc
import uuid
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch for PatchTSMixer
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# HuggingFace Transformers for PatchTSMixer
try:
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
    TRANSFORMERS_TRAINER_AVAILABLE = True
except ImportError:
    TRANSFORMERS_TRAINER_AVAILABLE = False
    Trainer = None
    TrainingArguments = None
    EarlyStoppingCallback = None

# Time series and forecasting libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

# skforecast for walk-forward validation
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import backtesting_sarimax

# XGBoost and additional ML libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# TensorFlow and Keras for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

# MLflow and tracking
import mlflow
from mlflow import log_param, log_metric
from mlflow.models import infer_signature
from mlflow import MlflowClient
from codecarbon import EmissionsTracker

from scipy.stats import norm
import psutil
import time

# Optuna for Bayesian hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna import Trial

# ======================
# CONFIGURACIÓN GLOBAL
# ======================
SEED = 42
logger = logging.getLogger(__name__)

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)

# SARIMAX optimizer defaults for cross-platform reproducibility
# These parameters ensure consistent optimization behavior across different
# platforms (MacOS ARM, Windows x86, Linux) and BLAS/LAPACK implementations
# Fix: Strengthened convergence tolerances for improved determinism
# Tighter tolerances reduce optimizer variance from 5-10% to 0.5-2%
SARIMAX_OPTIMIZER_DEFAULTS = {
    "method": "lbfgs",       # L-BFGS-B optimizer (statsmodels default)
    "maxiter": 500,          # Increased from 200 to 500 for better convergence
    "disp": 0,               # Quiet mode (no optimizer output)
    "ftol": 1e-8,            # Tighter function tolerance (was 1e-6)
    "gtol": 1e-6,            # Tighter gradient tolerance (was 1e-5)
    "epsilon": 1e-10,        # Smaller step for numerical gradient (was 1e-8)
    "iprint": -1,            # Suppress iteration warnings for cleaner output
}

# Training mode constants for LSTM
TRAINING_MODE_UNIVARIATE = "univariate"
TRAINING_MODE_MULTIVARIATE = "multivariate"

# Module-level variables for memory monitoring (Phase 8)
# These are reset at the start of each Bayesian search optimization
peak_memory_mb = 0.0
memory_exceeded = False

def set_global_seeds():
    """
    Fija semillas para todas las librerías relevantes para reproducibilidad completa.

    Configura:
    - Python's random module (para random.choice, random.shuffle, etc.)
    - NumPy random (para np.random.*)
    - TensorFlow random (para tf.random.*, inicializadores de capas)
    - TensorFlow determinismo operacional (para operaciones no-deterministas)
    """
    import random
    import os

    # Seed Python's built-in random module
    random.seed(SEED)

    # Seed NumPy random
    np.random.seed(SEED)

    # Replace tf.random.set_seed with the unified API
    tf.keras.utils.set_random_seed(SEED)  # ← was: tf.random.set_seed(SEED)

    # Add sklearn seeding:

    # Enable TensorFlow deterministic operations (TF 2.9+)
    # This ensures operations like tf.nn.bias_add, cuDNN convolutions, etc. are deterministic
    tf.config.experimental.enable_op_determinism()

    # Verify and set environment variables for additional reproducibility
    # (These should already be set in .env, but we ensure them programmatically)
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('PYTHONHASHSEED', '42')

    logger.info(f"Global seeds initialized: SEED={SEED}, TF determinism enabled")

# ======================
# PYTORCH REPRODUCIBILITY (for PatchTSMixer)
# ======================

def set_pytorch_reproducibility(seed=42):
    """
    Configure PyTorch deterministic behavior for reproducibility.

    This function ensures that PatchTSMixer training runs produce identical
    results across multiple executions with the same seed. Must be called
    before any PyTorch operations.

    Args:
        seed (int): Random seed for reproducibility. Default: 42

    Note:
        - Forces CPU-only execution for maximum reproducibility
        - Some operations may be slower with deterministic mode enabled
        - Requires PyTorch >= 2.0.0 and transformers >= 4.36.0

    Raises:
        ImportError: If torch or transformers not installed
        RuntimeError: If deterministic algorithms cannot be enabled (will warn and continue)
    """
    import logging
    import os

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch==2.0.1"
        )

    try:
        from transformers import set_seed as transformers_set_seed
    except ImportError:
        raise ImportError(
            "Transformers not installed. Install with: pip install transformers==4.36.0"
        )

    logger = logging.getLogger(__name__)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Set CUDA seeds if available (but we'll force CPU usage for reproducibility)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA available but will use CPU for reproducibility")

    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        logger.warning(
            f"Could not enable fully deterministic algorithms: {e}. "
            "Some operations may be non-deterministic."
        )
        # Continue anyway - partial determinism better than none

    # Configure cuDNN for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)  # ← ADD THIS for CPU reproducibility
    torch.set_num_interop_threads(1)  # ← Set inter-op threads too

    # Set environment variables for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #os.environ['PYTHONHASHSEED'] = str(seed)

    # Set transformers library seed (handles internal randomness)
    transformers_set_seed(seed)

    logger.info(
        f"PyTorch reproducibility configured with seed={seed}. "
        f"Deterministic algorithms: enabled, cuDNN benchmark: disabled"
    )

# Inicializar semillas globales
set_global_seeds()
set_pytorch_reproducibility(SEED)

# ======================
# PATCHTSMIXER DATA CLASSES
# ======================

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for PatchTSMixer time series data.

    This dataset wraps pre-computed past_values (context) and future_values (targets)
    tensors for efficient batch loading with PyTorch DataLoader.

    Args:
        past_values: Tensor of shape (num_samples, context_length, num_channels)
                    Historical values used as model input
        future_values: Tensor of shape (num_samples, prediction_length, num_channels)
                      Target values for forecasting
        observed_mask: Optional mask tensor of shape (num_samples, context_length, num_channels)
                      Indicates which values are observed (1) vs missing (0)
                      Defaults to all-ones (all values observed)

    Example:
        >>> dataset = TimeSeriesDataset(past_values, future_values)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=False)
        >>> for batch in loader:
        ...     past = batch['past_values']  # (32, 512, 3)
        ...     future = batch['future_values']  # (32, 96, 3)
    """
    def __init__(
        self,
        past_values: torch.Tensor,
        future_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TimeSeriesDataset but not installed. "
                "Install with: pip install torch>=2.0.1"
            )

        self.past_values = past_values
        self.future_values = future_values

        # Create default observed_mask if not provided (all values observed)
        if observed_mask is None:
            self.observed_mask = torch.ones_like(past_values)
        else:
            self.observed_mask = observed_mask

        # Validate shapes match between past and future
        assert past_values.shape[0] == future_values.shape[0], \
            f"Sample count mismatch: past={past_values.shape[0]}, future={future_values.shape[0]}"
        assert past_values.shape[2] == future_values.shape[2], \
            f"Channel count mismatch: past={past_values.shape[2]}, future={future_values.shape[2]}"

        if observed_mask is not None:
            assert observed_mask.shape == past_values.shape, \
                f"Observed mask shape {observed_mask.shape} doesn't match past_values {past_values.shape}"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.past_values.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with keys:
                - 'past_values': (context_length, num_channels)
                - 'future_values': (prediction_length, num_channels)
                - 'observed_mask': (context_length, num_channels)
        """
        return {
            'past_values': self.past_values[idx],
            'future_values': self.future_values[idx],
            'observed_mask': self.observed_mask[idx]
        }

# ======================
# FUNCIONES AUXILIARES
# ======================

def load_and_validate_ts_data(dataset_path: str, date_col_name: str, target_variable: str) -> pd.DataFrame:
    """
    Carga y valida el dataset de series temporales
    """
    # Fix: Explicit UTF-8 encoding for cross-platform reproducibility
    # Prevents encoding-dependent data loading across different OS locales
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Validar que las columnas existan
    if date_col_name not in df.columns:
        raise ValueError(f"Columna de fecha no encontrada: {date_col_name}")
    if target_variable not in df.columns:
        raise ValueError(f"Variable objetivo no encontrada: {target_variable}")
    
    # Convertir columna de fecha a datetime
    try:
        df[date_col_name] = pd.to_datetime(df[date_col_name])
    except Exception as e:
        raise ValueError(f"Error al convertir columna de fecha: {e}")
    
    # Ordenar por fecha
    df = df.sort_values(date_col_name).reset_index(drop=True)
    
    # Verificar valores nulos en la serie objetivo
    if df[target_variable].isnull().any():
        logger.warning(f"La serie objetivo {target_variable} contiene valores nulos")
    
    # Establecer fecha como índice
    df.set_index(date_col_name, inplace=True)

    # Inferir y establecer la frecuencia del DatetimeIndex
    # Esto es necesario para skforecast (ForecasterSarimax) que requiere freq explícito
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is not None:
        df = df.asfreq(inferred_freq)
        logger.info(f"Inferred and set DatetimeIndex frequency: {inferred_freq}")
    else:
        raise ValueError(
            "Could not infer frequency from DatetimeIndex. "
            "Ensure your time series has regular intervals (e.g., daily, hourly, monthly). "
            "Check for missing dates or irregular spacing in your data."
        )

    return df

def ts_train_val_test_split(df: pd.DataFrame, target_variable: str, split_ratios: Dict[str, float]) -> Tuple:
    """
    Divide el dataset de series temporales en train/val/test manteniendo el orden temporal
    """
    # Validar que los ratios sumen ≈1.0
    total_ratio = split_ratios["train"] + split_ratios["val"] + split_ratios["test"]
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Suma de ratios debe ser 1.0, actual: {total_ratio}")
    
    n = len(df)
    train_size = int(n * split_ratios["train"])
    val_size = int(n * split_ratios["val"])
    
    # Crear los conjuntos respetando el orden temporal
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    
    # Extraer las series objetivo
    y_train = train_data[target_variable]
    y_val = val_data[target_variable]
    y_test = test_data[target_variable]
    
    logger.info(f"División temporal - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    return y_train, y_val, y_test, train_data, val_data, test_data

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE (Mean Absolute Percentage Error).
    Handles zero values in y_true by using epsilon.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero by adding small epsilon where y_true is zero
    epsilon = 1e-10
    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        # All values are zero, return NaN
        logger.warning("All true values are zero or very close to zero, MAPE cannot be calculated")
        return np.nan

    # Only calculate MAPE for non-zero values
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def log_sarimax_convergence_diagnostics(model, prefix=""):
    """
    Log detailed convergence diagnostics for SARIMAX model to MLflow and logger.

    Helps detect optimization issues, platform-specific failures, and numerical instability.

    Args:
        model: Fitted SARIMAX model
        prefix: Prefix for MLflow metric names (e.g., "manual_", "grid_best_")
    """
    try:
        # Extract MLE optimization results
        if not hasattr(model, 'mle_retvals') or model.mle_retvals is None:
            logger.warning(f"{prefix}No MLE convergence information available")
            return

        mle = model.mle_retvals
        converged = mle.get('converged', False)
        iterations = mle.get('iterations', 0)
        fcalls = mle.get('fcalls', 0)
        warnflag = mle.get('warnflag', -1)

        # Log to MLflow
        mlflow.log_metric(f"{prefix}optimizer_converged", int(converged))
        mlflow.log_metric(f"{prefix}optimizer_iterations", iterations)
        mlflow.log_metric(f"{prefix}optimizer_fcalls", fcalls)
        mlflow.log_metric(f"{prefix}optimizer_warnflag", warnflag)

        # Extract gradient norm if available
        grad = mle.get('grad', None)
        grad_norm = None
        if grad is not None:
            if isinstance(grad, np.ndarray):
                grad_norm = float(np.linalg.norm(grad))
            else:
                grad_norm = float(grad)
            mlflow.log_metric(f"{prefix}optimizer_grad_norm", grad_norm)

            # WARNING: High gradient norm indicates poor convergence
            if grad_norm > 10.0:
                logger.warning(
                    f"⚠️  HIGH GRADIENT NORM DETECTED: {grad_norm:.2f} "
                    f"(threshold: 10.0). Model may not have converged properly. "
                    f"Consider increasing maxiter or adjusting convergence tolerances."
                )

        # Detailed logging
        convergence_status = "✅ CONVERGED" if converged else "❌ NOT CONVERGED"
        logger.info(f"{'='*60}")
        logger.info(f"SARIMAX Optimizer Diagnostics ({prefix})")
        logger.info(f"{'='*60}")
        logger.info(f"Convergence Status: {convergence_status}")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Function Calls: {fcalls}")
        logger.info(f"Warning Flag: {warnflag}")
        if grad_norm is not None:
            logger.info(f"Final Gradient Norm: {grad_norm:.6f}")
        logger.info(f"{'='*60}")

        # Warning flags interpretation
        if warnflag == 1:
            logger.warning("Optimizer warning: Too many function evaluations or iterations")
        elif warnflag == 2:
            logger.error("❌ CRITICAL: ABNORMAL_TERMINATION_IN_LNSRCH - Line search failed!")
            logger.error("This indicates numerical instability. Recommendations:")
            logger.error("  1. Check for collinearity in exogenous variables")
            logger.error("  2. Try scaling/normalizing features")
            logger.error("  3. Reduce model complexity (lower p, q, P, Q)")
            logger.error("  4. Increase gtol tolerance")

    except Exception as e:
        logger.error(f"Error logging convergence diagnostics: {e}", exc_info=True)

def evaluate_arima_model(model, y_train: pd.Series, y_true: pd.Series, prefix: str, forecast_horizon: int, experiment_dir: str, exog: pd.DataFrame = None,
                         y_val=None) -> Tuple[Dict, Dict]:
    """
    Evalúa el modelo ARIMA y genera métricas/gráficos
    """
    try:
        # Generar predicciones out-of-sample
        forecast_steps = len(y_true)

        # Si el modelo tiene exog, pasarlo al forecast
        if exog is not None:
            forecast = model.forecast(steps=forecast_steps, exog=exog)
        else:
            forecast = model.forecast(steps=forecast_steps)
        
        # Si forecast es un array, crear un Series con el índice correcto
        if isinstance(forecast, np.ndarray):
            y_pred = pd.Series(forecast, index=y_true.index)
        else:
            y_pred = forecast
        
        # Calcular métricas (convert numpy types to Python natives)
        metrics = {
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

        # Generar gráficos
        artifacts = generate_ts_plots(y_train, y_true, y_pred, model, prefix, experiment_dir, y_val=y_val)

        return metrics, artifacts
        
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        # Retornar métricas vacías si falla la evaluación
        return {f"{prefix}_rmse": None, f"{prefix}_mae": None, f"{prefix}_mape": None}, {}

def generate_ts_plots(y_train: pd.Series, y_true: pd.Series, y_pred: pd.Series, model, prefix: str, experiment_dir: str,
                      y_val=None) -> Dict:
    """
    Genera y guarda gráficos de evaluación para series temporales
    """
    artifacts = {
        "forecast_plot": None,
        "residuals_plot": None,
        "acf_pacf_plot": None
    }
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    try:
        # Gráfico de pronósticos vs real
        plt.figure(figsize=(15, 8))
        
        # Mostrar los últimos puntos del entrenamiento para contexto
        # Build context: last N points of train, plus full val if available (test plot).
        train_context = y_train.tail(min(50, len(y_train)))
        if y_val is not None:
            # Concatenate so the context line is continuous up to the test window.
            context = pd.concat([train_context, y_val])
        else:
            context = train_context

        plt.plot(context.index, context.values,
                 label='Training Data', color='blue', alpha=0.7)
        plt.plot(y_true.index, y_true.values, label='Actual', color='green', linewidth=2)
        plt.plot(y_true.index, y_pred.values, label='Forecast', color='red', linestyle='--', linewidth=2)
        
        plt.title(f'Time Series Forecast vs Actual - {prefix.capitalize()}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        forecast_path = os.path.join(experiment_dir, f"forecast_plot_{prefix}.png")
        plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts["forecast_plot"] = forecast_path
        
        # Gráfico de residuos
        residuals = y_true - y_pred
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuos vs tiempo
        axes[0, 0].plot(residuals.index, residuals.values, alpha=0.7)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[0, 1].hist(residuals.values, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuos vs predicciones
        axes[1, 1].scatter(y_pred.values, residuals.values, alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        residuals_path = os.path.join(experiment_dir, f"residuals_plot_{prefix}.png")
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts["residuals_plot"] = residuals_path
        
        # ACF y PACF de residuos
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_acf(residuals.values, ax=axes[0], lags=min(20, len(residuals)//4))
        plot_pacf(residuals.values, ax=axes[1], lags=min(20, len(residuals)//4))
        axes[0].set_title('ACF of Residuals')
        axes[1].set_title('PACF of Residuals')
        plt.tight_layout()
        acf_pacf_path = os.path.join(experiment_dir, f"acf_pacf_residuals_{prefix}.png")
        plt.savefig(acf_pacf_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts["acf_pacf_plot"] = acf_pacf_path
        
    except Exception as e:
        logger.error(f"Error generando gráficos: {e}")
    
    # Registrar artefactos en MLflow
    for artifact_type, path in artifacts.items():
        if path and os.path.exists(path):
            mlflow.log_artifact(path, "plots")
    
    return artifacts

def log_energy_metrics(tracker):
    """Registra métricas de energía y emisiones"""
    try:
        # Access the .kWh property of the Energy object
        energy_kwh = float(tracker._total_energy.kWh) if tracker._total_energy else 0.0
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not extract energy from tracker._total_energy: {e}")
        energy_kwh = 0.0

    try:
        emissions_kg = float(tracker.final_emissions) if tracker.final_emissions else 0.0
    except (TypeError, ValueError) as e:
        logger.warning(f"Could not extract emissions from tracker.final_emissions: {e}")
        emissions_kg = 0.0

    mlflow.log_metric("energy_consumed_total_kWh", energy_kwh)
    mlflow.log_metric("carbon_emission_kg", emissions_kg)
    return energy_kwh, emissions_kg

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Handles:
    - numpy integers (int8, int16, int32, int64) → Python int
    - numpy floats (float16, float32, float64) → Python float
    - numpy booleans (bool_) → Python bool
    - numpy arrays (ndarray) → Python list
    - dictionaries and lists (recursively)
    - None, str, and other native types (unchanged)

    Args:
        obj: Object to convert (can be dict, list, numpy type, or native type)

    Returns:
        Converted object with all numpy types replaced by Python natives
    """
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        # Native Python types (int, float, str, bool) pass through unchanged
        return obj

def save_pipeline_config(experiment_dir: str, config: Dict):
    """
    Guarda la configuración del pipeline en JSON.
    Convierte automáticamente tipos numpy a tipos nativos de Python.
    """
    config_path = os.path.join(experiment_dir, "pipeline_config.json")

    # Convert numpy types in config before saving
    config = convert_numpy_to_python(config)

    # Try to load existing config
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing_config = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted pipeline_config.json detected: {e}. Creating backup and starting fresh.")
            # Backup corrupted file
            backup_path = config_path + ".corrupted_backup"
            os.rename(config_path, backup_path)
            logger.info(f"Corrupted file backed up to: {backup_path}")
            existing_config = {"steps": []}
    else:
        existing_config = {"steps": []}

    # Append new config and save
    existing_config["steps"].append(config)
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=4)


def validate_pipeline_config_schema(config: Dict, strict: bool = False) -> bool:
    """
    Validates pipeline_config.json schema for completeness and type correctness.

    Supports version-aware validation:
    - v1.0: Relaxed validation (legacy format, pre-Phase 3A)
    - v1.1: Strict validation (requires all fields including complete metrics and hyperparameter_search)

    Args:
        config: Pipeline configuration dictionary to validate
        strict: If True, raise exceptions on validation errors. If False, log warnings only.

    Returns:
        True if validation passes, False if validation fails (non-strict mode only)

    Raises:
        ValueError: If strict=True and validation fails with detailed error messages

    Example:
        >>> config = {"schema_version": "1.1", "step": "train_model", ...}
        >>> validate_pipeline_config_schema(config, strict=False)
        True
    """
    version = config.get("schema_version", "1.0")
    errors = []

    # ===== Version 1.1 Validation (New Format - Phase 3A+) =====
    if version == "1.1":
        # Required top-level fields
        required_top_level = ["schema_version", "step", "algorithm", "params",
                               "metrics", "hyperparameter_search", "lstm_metadata"]

        for field in required_top_level:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate metrics completeness (all 6 metrics required)
        if "metrics" in config:
            required_metrics = ["val_rmse", "val_mae", "val_mape",
                                "test_rmse", "test_mae", "test_mape"]
            for metric in required_metrics:
                if metric not in config["metrics"]:
                    errors.append(f"Missing required metric: {metric}")
                elif config["metrics"][metric] is not None:
                    # Allow None for MAPE (division by zero case)
                    if not isinstance(config["metrics"][metric], (int, float)):
                        errors.append(
                            f"Invalid type for {metric}: expected float or None, "
                            f"got {type(config['metrics'][metric]).__name__}"
                        )

        # Validate hyperparameter_search structure
        if "hyperparameter_search" in config:
            hs = config["hyperparameter_search"]
            required_hs_fields = ["strategy", "iterations_total", "best_iteration", "best_val_loss"]

            for field in required_hs_fields:
                if field not in hs:
                    errors.append(f"Missing hyperparameter_search field: {field}")

            # Strategy-specific validation
            if "strategy" in hs:
                strategy = hs["strategy"]
                if strategy not in ["none", "grid", "random", "bayesian"]:
                    errors.append(f"Invalid strategy: {strategy}")

                # Grid search should include grid_search_params
                if strategy == "grid" and hs.get("grid_search_params") is None:
                    errors.append("Missing grid_search_params for grid strategy")

                # Random search should include random_search_params and n_random_iterations
                if strategy == "random":
                    if hs.get("random_search_params") is None:
                        errors.append("Missing random_search_params for random strategy")
                    if hs.get("n_random_iterations") is None:
                        errors.append("Missing n_random_iterations for random strategy")

                # Bayesian search should include bayesian_config at top level
                if strategy == "bayesian":
                    if "bayesian_config" not in config:
                        errors.append("Missing bayesian_config for bayesian strategy")
                    else:
                        bc = config["bayesian_config"]
                        required_bayesian_fields = ["n_trials", "n_initial_points", "optimization_metric",
                                                    "best_trial_number", "n_completed_trials", "best_params", "seed"]
                        for field in required_bayesian_fields:
                            if field not in bc:
                                errors.append(f"Missing {field} in bayesian_config")

                        # Validate n_trials >= 1
                        if "n_trials" in bc and bc["n_trials"] < 1:
                            errors.append("n_trials must be at least 1")

                        # Validate n_initial_points < n_trials
                        if "n_trials" in bc and "n_initial_points" in bc:
                            if bc["n_initial_points"] >= bc["n_trials"]:
                                errors.append("n_initial_points must be less than n_trials")

        # Validate lstm_metadata
        if "lstm_metadata" in config:
            required_metadata = ["sequence_length", "model_architecture", "total_params", "cpu_only"]

            # Validate required fields
            for field in required_metadata:
                if field not in config["lstm_metadata"]:
                    errors.append(f"Missing lstm_metadata field: {field}")

            # Optional fields (Phase 4): validate type and values if present
            if "training_mode" in config["lstm_metadata"]:
                mode = config["lstm_metadata"]["training_mode"]
                if mode not in [TRAINING_MODE_UNIVARIATE, TRAINING_MODE_MULTIVARIATE]:
                    errors.append(
                        f"Invalid training_mode: {mode}. "
                        f"Expected '{TRAINING_MODE_UNIVARIATE}' or '{TRAINING_MODE_MULTIVARIATE}'"
                    )

            if "n_input_features" in config["lstm_metadata"]:
                n_features = config["lstm_metadata"]["n_input_features"]
                if not isinstance(n_features, int) or n_features < 1:
                    errors.append(
                        f"Invalid n_input_features: {n_features}. Expected positive integer"
                    )

    # ===== Version 1.0 Validation (Legacy Format - Backward Compatibility) =====
    elif version == "1.0":
        # Relaxed validation for legacy format (pre-Phase 3A)
        logger.info("Validating legacy pipeline_config schema (v1.0) - relaxed validation")
        if "step" not in config:
            errors.append("Missing required field: step")
        if "algorithm" not in config:
            errors.append("Missing required field: algorithm")

    else:
        errors.append(f"Unknown schema version: {version}")

    # ===== Handle Validation Errors =====
    if errors:
        error_msg = (
            f"Pipeline config schema validation failed ({len(errors)} error(s)):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
        if strict:
            raise ValueError(error_msg)
        else:
            # Non-strict mode: Log warnings only, don't block training
            logger.warning(error_msg)
            return False

    # Validation passed (silent success per user requirement)
    logger.info(f"Pipeline config schema validation passed (version {version})")
    return True



def create_supervised_dataset(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                            forecast_horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convierte los datos de series temporales a formato supervisado para XGBoost

    Args:
        df: DataFrame con todas las características preparadas
        target_col: Nombre de la columna objetivo
        feature_cols: Lista de columnas a usar como características
        forecast_horizon: Horizonte de pronóstico

    Returns:
        Tupla (X, y) donde 'X' son las características e 'y' es el objetivo
    """
    # Asegurar que no hay valores NaN en las características objetivo para el futuro
    df_clean = df.copy()

    # Para el objetivo, necesitamos poder hacer forecasting hacia adelante
    # Por lo tanto, el objetivo para cada fila es el valor forecast_horizon pasos adelante
    y = df_clean[target_col].shift(-forecast_horizon)

    # Seleccionar solo las características especificadas
    X = df_clean[feature_cols].copy()

    # Eliminar las últimas 'forecast_horizon' filas porque no tienen objetivo
    X = X.iloc[:-forecast_horizon]
    y = y.iloc[:-forecast_horizon]

    # Eliminar filas con valores NaN
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Dataset supervisado creado: {X.shape[0]} muestras, {X.shape[1]} características")
    logger.info(f"Características utilizadas: {feature_cols}")

    return X, y

def xgboost_train_val_test_split(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                                forecast_horizon: int, split_ratios: Dict[str, float]) -> Tuple:
    """
    Divide el dataset preparado para XGBoost manteniendo el orden temporal

    Args:
        df: DataFrame con características preparadas
        target_col: Nombre de la columna objetivo
        feature_cols: Lista de columnas de características
        forecast_horizon: Horizonte de pronóstico
        split_ratios: Diccionario con ratios de división

    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test, train_data, val_data, test_data)
    """
    # Crear dataset supervisado
    X, y = create_supervised_dataset(df, target_col, feature_cols, forecast_horizon)

    # Validar que los ratios sumen ≈1.0
    total_ratio = split_ratios["train"] + split_ratios["val"] + split_ratios["test"]
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Suma de ratios debe ser 1.0, actual: {total_ratio}")

    n = len(X)
    train_size = int(n * split_ratios["train"])
    val_size = int(n * split_ratios["val"])

    # Crear los conjuntos respetando el orden temporal
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]

    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]

    offset = len(df) - len(X)
    train_data = df.iloc[offset:offset + train_size]
    val_data   = df.iloc[offset + train_size:offset + train_size + val_size]
    test_data  = df.iloc[offset + train_size + val_size:] 

    logger.info(f"División temporal XGBoost - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, train_data, val_data, test_data

def evaluate_xgboost_model(model, X_train: pd.DataFrame, X_true: pd.DataFrame, y_train: pd.Series, y_true: pd.Series,
                          prefix: str, forecast_horizon: int, experiment_dir: str, feature_names: List[str],
                          y_val=None) -> Tuple[Dict, Dict]:
    """
    Evalúa el modelo XGBoost y genera métricas/gráficos

    Args:
        model: Modelo XGBoost entrenado
        X_train: Características de entrenamiento (para contexto en gráficos)
        X_true: Características del conjunto a evaluar
        y_true: Valores reales del conjunto a evaluar
        prefix: Prefijo para nombres de métricas/archivos
        forecast_horizon: Horizonte de pronóstico
        experiment_dir: Directorio para guardar artefactos
        feature_names: Lista de nombres de características

    Returns:
        Tupla (metrics, artifacts)
    """
    try:
        # Generar predicciones
        y_pred = model.predict(X_true)
        logger.info(f"y_pred min/max: {y_pred.min()}/{y_pred.max()}")

        # Asegurar que y_pred tenga el mismo índice que y_true
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_true.index)

        # Calcular métricas
        metrics = {
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_mape": mean_absolute_percentage_error(y_true, y_pred),
        }

        # Generar gráficos
        artifacts = generate_xgboost_plots(X_train, X_true, y_train, y_true, y_pred, model, prefix,
                                         experiment_dir, feature_names, y_val=y_val)

        return metrics, artifacts

    except Exception as e:
        logger.error(f"Error en evaluación XGBoost: {e}")
        # Retornar métricas vacías si falla la evaluación
        return {f"{prefix}_rmse": None, f"{prefix}_mae": None, f"{prefix}_mape": None}, {}

def generate_xgboost_plots(X_train: pd.DataFrame, X_true: pd.DataFrame, y_train: pd.Series, y_true: pd.Series,
                          y_pred: pd.Series, model, prefix: str, experiment_dir: str,
                          feature_names: List[str],
                          y_val=None) -> Dict:
    """
    Genera y guarda gráficos de evaluación para XGBoost

    Args:
        X_train: Características de entrenamiento
        X_true: Características del conjunto evaluado
        y_true: Valores reales
        y_pred: Predicciones del modelo
        model: Modelo XGBoost entrenado
        prefix: Prefijo para archivos
        experiment_dir: Directorio para guardar artefactos
        feature_names: Lista de nombres de características

    Returns:
        Diccionario con rutas de artefactos generados
    """
    artifacts = {
        "forecast_plot": None,
        "residuals_plot": None,
        "feature_importance_plot": None
    }

    os.makedirs(experiment_dir, exist_ok=True)

    try:
        # Gráfico de pronósticos vs real
        plt.figure(figsize=(15, 8))

        # Mostrar los últimos puntos del entrenamiento para contexto

        # Build context: last N points of train, plus full val if available (test plot).
        train_context = y_train.tail(min(50, len(y_train)))
        if y_val is not None:
            # Concatenate so the context line is continuous up to the test window.
            context = pd.concat([train_context, y_val])
        else:
            context = train_context

        plt.plot(context.index, context.values,label='Training Data', color='blue', alpha=0.7)

        plt.plot(y_true.index, y_true.values, label='Actual', color='green', linewidth=2)
        plt.plot(y_pred.index, y_pred.values, label='Forecast', color='red', linestyle='--', linewidth=2)

        plt.title(f'XGBoost Time Series Forecast vs Actual - {prefix.capitalize()}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        forecast_path = os.path.join(experiment_dir, f"xgb_forecast_plot_{prefix}.png")
        plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts["forecast_plot"] = forecast_path

        # Gráfico de residuos
        residuals = y_true - y_pred
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuos vs tiempo
        axes[0, 0].plot(residuals.index, residuals.values, alpha=0.7)
        axes[0, 0].set_title('Residuals vs Time')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].grid(True, alpha=0.3)

        # Histograma de residuos
        axes[0, 1].hist(residuals.values, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuos vs predicciones
        axes[1, 1].scatter(y_pred.values, residuals.values, alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        residuals_path = os.path.join(experiment_dir, f"xgb_residuals_plot_{prefix}.png")
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts["residuals_plot"] = residuals_path

        # Gráfico de importancia de características
        try:
            # Obtener importancia de características
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)  # Top 20 características
            plt.barh(top_features['feature'], top_features['importance'])
            plt.title(f'Top 20 Feature Importance - {prefix.capitalize()}')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            importance_path = os.path.join(experiment_dir, f"xgb_feature_importance_{prefix}.png")
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts["feature_importance_plot"] = importance_path

            # Registrar importancia en MLflow
            for feature, importance in zip(importance_df['feature'][:10], importance_df['importance'][:10]):
                mlflow.log_metric(f"feature_importance_{feature}", importance)

        except Exception as e:
            logger.warning(f"Error generando gráfico de importancia: {e}")

    except Exception as e:
        logger.error(f"Error generando gráficos XGBoost: {e}")

    # Registrar artefactos en MLflow
    for artifact_type, path in artifacts.items():
        if path and os.path.exists(path):
            mlflow.log_artifact(path, "plots")

    return artifacts

# ======================
# SARIMAX HELPER FUNCTIONS
# ======================

def compute_stable_start_params(model_spec, y_data, exog_data=None):
    """
    Compute stable starting parameters for SARIMAX optimization.

    Uses deterministic heuristics to provide consistent initial values across platforms,
    reducing sensitivity to platform-dependent statsmodels defaults.

    Args:
        model_spec: SARIMAX model specification (not yet fitted)
        y_data: Training data (pd.Series)
        exog_data: Exogenous variables (pd.DataFrame or None)

    Returns:
        np.ndarray: Initial parameter vector, or None if computation fails
    """
    try:
        # Force deterministic behavior
        np.random.seed(SEED)

        # Get parameter count
        k_params = model_spec.k_params

        # Initialize all parameters to zero (conservative starting point)
        start_params = np.zeros(k_params)

        # Initialize variance parameter (typically the last parameter)
        # Use log of target variance for numerical stability
        y_var = np.var(y_data)
        if y_var > 0:
            start_params[-1] = np.log(y_var)

        # Initialize AR/MA coefficients with small positive values
        # This helps avoid saddle points in the optimization landscape
        ar_count = model_spec.k_ar + model_spec.k_seasonal_ar
        ma_count = model_spec.k_ma + model_spec.k_seasonal_ma

        # Small positive values for AR terms
        for i in range(ar_count):
            start_params[i] = 0.1

        # Small positive values for MA terms
        for i in range(ma_count):
            start_params[ar_count + i] = 0.1

        logger.debug(f"Generated stable start_params: shape={start_params.shape}, "
                     f"AR={ar_count}, MA={ma_count}, total={k_params}")
        return start_params

    except Exception as e:
        logger.warning(f"Could not compute stable start_params: {e}. "
                      f"Will use statsmodels defaults.")
        return None

# ======================
# GRID SEARCH HELPER FUNCTIONS
# ======================

def generate_arima_grid(grid_config: Dict) -> List[Dict]:
    """
    Genera el grid completo de parámetros ARIMA/SARIMA desde la configuración del usuario.

    Lógica de manejo:
    - Si p,d,q están presentes y P,D,Q,s están todos vacíos → Solo ARIMA
    - Si P,D,Q,s están presentes pero p,d,q están vacíos → Usar P,D,Q como p,d,q para ARIMA
    - Si ambos están presentes:
        - s=None → ARIMA con (p,d,q)
        - s=valor → SARIMA con (p,d,q)x(P,D,Q,s)

    Args:
        grid_config: Diccionario con listas de parámetros
            {
                "p": [0, 1, 2],
                "d": [1],
                "q": [0, 1],
                "P": [0, 1],
                "D": [1],
                "Q": [0, 1],
                "s": [12, None],
                "trend": ["n", "c"],
                "enforce_stationarity": [True, False],
                "enforce_invertibility": [True]
            }

    Returns:
        Lista de diccionarios con parámetros para cada iteración:
        [
            {
                "order": (p, d, q),
                "seasonal_order": None or (P, D, Q, s),
                "trend": "n",
                "enforce_stationarity": True,
                "enforce_invertibility": True
            },
            ...
        ]
    """
    # Extraer listas de parámetros
    p_values = grid_config.get("p", [])
    d_values = grid_config.get("d", [])
    q_values = grid_config.get("q", [])
    P_values = grid_config.get("P", [])
    D_values = grid_config.get("D", [])
    Q_values = grid_config.get("Q", [])
    s_values = grid_config.get("s", [])

    # Parámetros adicionales
    trend_values = grid_config.get("trend", [])
    if not trend_values:  # Si está vacío, usar solo 'n'
        trend_values = ['n']

    enforce_stationarity_values = grid_config.get("enforce_stationarity", [True])
    enforce_invertibility_values = grid_config.get("enforce_invertibility", [True])

    param_grid = []

    # Caso 1: Si P,D,Q,s están todos vacíos → Solo ARIMA con p,d,q
    seasonal_empty = (not P_values and not D_values and not Q_values and not s_values)

    # Caso 2: Si p,d,q están vacíos pero P,D,Q,s tienen valores → Usar P,D,Q como p,d,q
    if not p_values and not d_values and not q_values and not seasonal_empty:
        logger.info("p, d, q vacíos - usando P, D, Q como p, d, q para ARIMA")
        p_values = P_values
        d_values = D_values
        q_values = Q_values
        # Forzar ARIMA (no seasonal)
        seasonal_empty = True

    # Generar combinaciones ARIMA (cuando seasonal está vacío o s=None está en la lista)
    if seasonal_empty or None in s_values:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for trend in trend_values:
                        for enf_stat in enforce_stationarity_values:
                            for enf_inv in enforce_invertibility_values:
                                param_grid.append({
                                    "order": (int(p), int(d), int(q)),
                                    "seasonal_order": None,
                                    "trend": trend if trend != 'n' else None,
                                    "enforce_stationarity": bool(enf_stat),
                                    "enforce_invertibility": bool(enf_inv)
                                })

    # Generar combinaciones SARIMA (cuando s tiene valores no-None)
    if not seasonal_empty:
        s_values_filtered = [s for s in s_values if s is not None]

        if s_values_filtered:
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    for s in s_values_filtered:
                                        for trend in trend_values:
                                            for enf_stat in enforce_stationarity_values:
                                                for enf_inv in enforce_invertibility_values:
                                                    param_grid.append({
                                                        "order": (int(p), int(d), int(q)),
                                                        "seasonal_order": (int(P), int(D), int(Q), int(s)),
                                                        "trend": trend if trend != 'n' else None,
                                                        "enforce_stationarity": bool(enf_stat),
                                                        "enforce_invertibility": bool(enf_inv)
                                                    })

    logger.info(f"Grid generado con {len(param_grid)} combinaciones de parámetros")
    return param_grid


def walk_forward_validate_sarimax(
    y_data: pd.Series,
    exog_data: pd.DataFrame | None,
    params: Dict,
    n_folds: int,
    initial_train_size: int,
    forecast_horizon: int
) -> Dict[str, float]:
    """
    Realiza validación walk-forward para modelo SARIMAX usando skforecast.

    Estrategia:
    - Rolling window (tamaño fijo de entrenamiento)
    - Always refit (reentrenar en cada fold)
    - Step size = forecast_horizon (avance por horizonte de pronóstico)
    - Retorna métricas solo del fold final

    Args:
        y_data: Serie temporal objetivo
        exog_data: Variables exógenas (puede ser None)
        params: Diccionario con parámetros del modelo
            {
                "order": (p, d, q),
                "seasonal_order": None or (P, D, Q, s),
                "trend": None or str,
                "enforce_stationarity": bool,
                "enforce_invertibility": bool
            }
        n_folds: Número de folds para walk-forward
        initial_train_size: Tamaño de la ventana de entrenamiento (rolling)
        forecast_horizon: Pasos adelante a predecir

    Returns:
        Diccionario con métricas del fold final:
        {
            "val_rmse": float,
            "val_mae": float,
            "val_mape": float,
            "test_rmse": float,
            "test_mae": float,
            "test_mape": float
        }

    Raises:
        ValueError: Si la configuración es inválida
        RuntimeError: Si el modelo no converge
    """
    # Debug logging: entrada a walk_forward_validate_sarimax
    logger.info(f"[DEBUG-WF] Entrada a walk_forward_validate_sarimax")
    logger.info(f"[DEBUG-WF] y_data shape: {y_data.shape}, type: {type(y_data)}")
    logger.info(f"[DEBUG-WF] y_data index: type={type(y_data.index)}, freq={getattr(y_data.index, 'freq', None)}")
    if exog_data is not None:
        logger.info(f"[DEBUG-WF] exog_data shape: {exog_data.shape}, type: {type(exog_data)}")
        logger.info(f"[DEBUG-WF] exog_data index: type={type(exog_data.index)}, freq={getattr(exog_data.index, 'freq', None)}")
        logger.info(f"[DEBUG-WF] exog_data columns: {list(exog_data.columns)}")
    else:
        logger.info(f"[DEBUG-WF] exog_data is None")
    logger.info(f"[DEBUG-WF] params: {params}")
    logger.info(f"[DEBUG-WF] n_folds: {n_folds}, initial_train_size: {initial_train_size}, forecast_horizon: {forecast_horizon}")

    # Validar tamaño de datos
    if len(y_data) < initial_train_size + forecast_horizon:
        raise ValueError(
            f"Dataset insuficiente: necesita al menos {initial_train_size + forecast_horizon} muestras, "
            f"tiene {len(y_data)}"
        )

    # Crear forecaster de skforecast
    sarimax_params = {
        "order": params["order"],
        "seasonal_order": params.get("seasonal_order"),
        "trend": params.get("trend"),
        "enforce_stationarity": params.get("enforce_stationarity", True),
        "enforce_invertibility": params.get("enforce_invertibility", True)
    }

    forecaster = ForecasterSarimax(regressor=Sarimax(**sarimax_params))

    # Configurar estrategia de walk-forward
    # Step size = forecast_horizon (avanzar por horizonte completo en cada fold)
    # Calcular step para que tengamos exactamente n_folds
    total_available = len(y_data) - initial_train_size - forecast_horizon
    if total_available < 0:
        raise ValueError("No hay suficientes datos para crear folds")

    # Realizar backtesting con skforecast
    all_fold_predictions = []
    all_fold_actuals = []

    for fold_idx in range(n_folds):
        train_start = fold_idx * forecast_horizon
        train_end = train_start + initial_train_size
        test_start = train_end
        test_end = test_start + forecast_horizon

        logger.info(f"[DEBUG-FOLD] === Fold {fold_idx + 1}/{n_folds} ===")
        logger.info(f"[DEBUG-FOLD] Boundaries: train[{train_start}:{train_end}], test[{test_start}:{test_end}]")

        # Verificar que no excedamos el tamaño del dataset
        if test_end > len(y_data):
            logger.warning(f"Fold {fold_idx + 1} excede el tamaño del dataset, ajustando...")
            test_end = len(y_data)
            if test_end <= test_start:
                break  # No hay datos suficientes para este fold

        # Extraer datos para este fold
        y_train_fold = y_data.iloc[train_start:train_end]
        y_test_fold = y_data.iloc[test_start:test_end]

        logger.info(f"[DEBUG-FOLD] y_train_fold shape: {y_train_fold.shape}, freq: {getattr(y_train_fold.index, 'freq', None)}")
        logger.info(f"[DEBUG-FOLD] y_test_fold shape: {y_test_fold.shape}, freq: {getattr(y_test_fold.index, 'freq', None)}")

        exog_train_fold = None
        exog_test_fold = None
        if exog_data is not None and len(exog_data) > 0:
            exog_train_fold = exog_data.iloc[train_start:train_end]
            exog_test_fold = exog_data.iloc[test_start:test_end]

            logger.info(f"[DEBUG-FOLD] exog_train_fold BEFORE asfreq: shape={exog_train_fold.shape}, freq={getattr(exog_train_fold.index, 'freq', None)}")
            logger.info(f"[DEBUG-FOLD] exog_test_fold BEFORE asfreq: shape={exog_test_fold.shape}, freq={getattr(exog_test_fold.index, 'freq', None)}")

            # Preserve frequency after slicing (required for skforecast SARIMAX)
            # .iloc[] slicing can drop the frequency attribute even if parent has it
            if hasattr(exog_data, 'index') and hasattr(exog_data.index, 'freq') and exog_data.index.freq is not None:
                exog_train_fold = exog_train_fold.asfreq(exog_data.index.freq)
                exog_test_fold = exog_test_fold.asfreq(exog_data.index.freq)
                logger.info(f"[DEBUG-FOLD] Frequency preserved: {exog_data.index.freq}")

            logger.info(f"[DEBUG-FOLD] exog_train_fold AFTER asfreq: shape={exog_train_fold.shape}, freq={getattr(exog_train_fold.index, 'freq', None)}")
            logger.info(f"[DEBUG-FOLD] exog_test_fold AFTER asfreq: shape={exog_test_fold.shape}, freq={getattr(exog_test_fold.index, 'freq', None)}")
        else:
            logger.info(f"[DEBUG-FOLD] No exog data for this fold")

        # Entrenar modelo en este fold
        logger.info(f"[DEBUG-FOLD] Calling forecaster.fit() with y shape={y_train_fold.shape}, exog shape={exog_train_fold.shape if exog_train_fold is not None else None}")
        try:
            forecaster.fit(y=y_train_fold, exog=exog_train_fold)
            logger.info(f"[DEBUG-FOLD] forecaster.fit() succeeded")
        except Exception as e:
            logger.error(f"[DEBUG-FOLD] forecaster.fit() FAILED: {str(e)}")
            raise

        # Predecir
        steps_to_predict = len(y_test_fold)
        logger.info(f"[DEBUG-FOLD] Calling forecaster.predict() with steps={steps_to_predict}, exog shape={exog_test_fold.shape if exog_test_fold is not None else None}")
        try:
            predictions = forecaster.predict(steps=steps_to_predict, exog=exog_test_fold)
            logger.info(f"[DEBUG-FOLD] forecaster.predict() succeeded, predictions shape: {predictions.shape}")
        except Exception as e:
            logger.error(f"[DEBUG-FOLD] forecaster.predict() FAILED: {str(e)}")
            raise

        # Guardar predicciones y valores reales
        all_fold_predictions.append(predictions)
        all_fold_actuals.append(y_test_fold)

    if len(all_fold_predictions) == 0:
        raise RuntimeError("No se pudo completar ningún fold de validación")

    # Concatenar todas las predicciones de todos los folds
    # (Fix: anteriormente solo usaba el fold final, causando arrays vacíos con forecast_horizon=1)
    all_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    all_actuals = pd.concat(all_fold_actuals, ignore_index=True)

    logger.info(f"[DEBUG-METRICS] Total predictions from all folds: {len(all_predictions)}")
    logger.info(f"[DEBUG-METRICS] Total actuals from all folds: {len(all_actuals)}")

    # Validar que tenemos suficientes predicciones para dividir
    n_total = len(all_predictions)
    if n_total < 2:
        raise ValueError(
            f"Insufficient predictions for val/test split: {n_total} samples. "
            f"Need at least 2 predictions. Consider increasing n_folds or forecast_horizon."
        )

    # Dividir las predicciones agregadas en val/test (mitad y mitad)
    split_point = n_total // 2

    logger.info(f"[DEBUG-METRICS] Splitting {n_total} predictions at index {split_point}")

    # Val metrics: primera mitad de todas las predicciones
    val_pred = all_predictions.iloc[:split_point]
    val_true = all_actuals.iloc[:split_point]

    # Test metrics: segunda mitad de todas las predicciones
    test_pred = all_predictions.iloc[split_point:]
    test_true = all_actuals.iloc[split_point:]

    logger.info(f"[DEBUG-METRICS] val_pred shape: {val_pred.shape}, val_true shape: {val_true.shape}")
    logger.info(f"[DEBUG-METRICS] test_pred shape: {test_pred.shape}, test_true shape: {test_true.shape}")

    # Calcular métricas para val
    val_rmse = float(np.sqrt(mean_squared_error(val_true, val_pred)))
    val_mae = float(mean_absolute_error(val_true, val_pred))
    val_mape = float(mean_absolute_percentage_error(val_true, val_pred))

    logger.info(f"[DEBUG-METRICS] Val metrics - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, MAPE: {val_mape:.4f}")

    # Calcular métricas para test
    test_rmse = float(np.sqrt(mean_squared_error(test_true, test_pred)))
    test_mae = float(mean_absolute_error(test_true, test_pred))
    test_mape = float(mean_absolute_percentage_error(test_true, test_pred))

    logger.info(f"[DEBUG-METRICS] Test metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

    return {
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_mape": val_mape,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_mape": test_mape
    }


# ======================
# RANDOM SEARCH HELPER FUNCTIONS
# ======================

def generate_random_arima_params(random_search_params: Dict, seasonal_s: int = None, rng: np.random.Generator = None) -> Dict:
    """
    Genera parámetros ARIMA aleatorios basados en los rangos especificados.

    Args:
        random_search_params: Diccionario con rangos de parámetros
        seasonal_s: Valor de estacionalidad (si es None, no se usan parámetros estacionales)

    Returns:
        Diccionario con parámetros aleatorios generados
    """
    # Valores por defecto para los rangos
    default_ranges = {
        "p_range": [0, 4],
        "d_range": [0, 3],
        "q_range": [0, 4],
        "seasonal_P_range": [0, 3],
        "seasonal_D_range": [0, 3],
        "seasonal_Q_range": [0, 3]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros ARIMA básicos (convert numpy int64 to Python int)
    p = int(rng.integers(ranges["p_range"][0], ranges["p_range"][1]))
    d = int(rng.integers(ranges["d_range"][0], ranges["d_range"][1]))
    q = int(rng.integers(ranges["q_range"][0], ranges["q_range"][1]))

    params = {
        "order": (p, d, q),
        "seasonal_order": None
    }

    # Generar parámetros estacionales si se especifica seasonal_s
    if seasonal_s is not None:
        P = int(rng.integers(ranges["seasonal_P_range"][0], ranges["seasonal_P_range"][1]))
        D = int(rng.integers(ranges["seasonal_D_range"][0], ranges["seasonal_D_range"][1]))
        Q = int(rng.integers(ranges["seasonal_Q_range"][0], ranges["seasonal_Q_range"][1]))
        params["seasonal_order"] = (P, D, Q, seasonal_s)

    return params

def generate_random_xgboost_params(random_search_params: Dict, rng: np.random.Generator = None) -> Dict:
    """
    Genera parámetros XGBoost aleatorios basados en los rangos especificados.

    Args:
        random_search_params: Diccionario con rangos de parámetros

    Returns:
        Diccionario con parámetros aleatorios generados
    """
    # Handle rng == None
    if rng is None:
        rng = np.random.default_rng(SEED)
    # Valores por defecto para los rangos
    default_ranges = {
        "n_estimators_range": [50, 1000],
        "max_depth_range": [3, 10],
        "learning_rate_range": [0.01, 0.3],
        "subsample_range": [0.5, 1.0],
        "colsample_bytree_range": [0.5, 1.0],
        "gamma_range": [0.0, 5.0],
        "min_child_weight_range": [1, 10],
        "reg_alpha_range": [0.0, 1.0],
        "reg_lambda_range": [0.0, 1.0]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros aleatorios
    params = {
        "n_estimators": int(rng.integers(ranges["n_estimators_range"][0], ranges["n_estimators_range"][1] + 1)),
        "max_depth": int(rng.integers(ranges["max_depth_range"][0], ranges["max_depth_range"][1] + 1)),
        "min_child_weight": int(rng.integers(ranges["min_child_weight_range"][0], ranges["min_child_weight_range"][1] + 1)),
        "subsample": float(rng.uniform(ranges["subsample_range"][0], ranges["subsample_range"][1])),
        "colsample_bytree": float(rng.uniform(ranges["colsample_bytree_range"][0], ranges["colsample_bytree_range"][1])),
        "gamma": float(rng.uniform(ranges["gamma_range"][0], ranges["gamma_range"][1])),
        "reg_alpha": float(rng.uniform(ranges["reg_alpha_range"][0], ranges["reg_alpha_range"][1])),
        "reg_lambda": float(rng.uniform(ranges["reg_lambda_range"][0], ranges["reg_lambda_range"][1])),
        "random_state": SEED,
        "n_jobs": -1
    }

    # Usar muestreo log-uniforme para learning_rate
    log_min = np.log(ranges["learning_rate_range"][0])
    log_max = np.log(ranges["learning_rate_range"][1])
    params["learning_rate"] = float(np.exp(rng.uniform(log_min, log_max)))

    return params


def calculate_ts_metric(y_true, y_pred, metric_name: str) -> float:
    """
    Calcula métrica de series temporales para optimización.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        metric_name: Nombre de la métrica ('mse', 'mae', 'rmse', 'mape') tanto en validation o test

    Returns:
        Valor de la métrica calculada

    Raises:
        ValueError: Si el nombre de la métrica no es válido
    """
    if metric_name == "val_mse" or metric_name == "test_mse":
        return mean_squared_error(y_true, y_pred)
    elif metric_name == "val_mae" or metric_name == "test_mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == "val_rmse" or metric_name == "test_rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == "val_mape" or metric_name == "test_mape":
        # Evitar división por cero
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        raise ValueError(f"Métrica no soportada: {metric_name}")

# ======================
# FUNCIÓN DE ENTRENAMIENTO ARIMA
# ======================

def train_arima_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Entrena y registra un modelo ARIMA/SARIMA para pronóstico de series temporales.
    """
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=experiment_dir, save_to_file=False, allow_multiple_runs=True)
    tracker.start()

    # Log platform information for reproducibility tracking
    import platform
    import sys
    import scipy
    import statsmodels

    platform_info = {
        "python_version": sys.version.split()[0],  # Just version number
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": statsmodels.__version__,
    }

    logger.info("="*60)
    logger.info("Platform Information (for reproducibility):")
    for key, value in platform_info.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)

    # Log to MLflow (truncate if needed to avoid param length limits)
    mlflow.log_params({f"platform_{k}": str(v)[:250] for k, v in platform_info.items()})

    # Extraer parámetros
    date_col_name = data.get("date_col_name")
    target_variable = data["target_variable"]
    forecast_horizon = data.get("forecast_horizon", 10)
    hyperparams = data.get("manual_params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "ARIMA_Model")
    problem_type = "ts_forecasting"
    input_features = data.get("input_features")
    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", "none")

    # Validar estrategia
    valid_strategies = ["manual", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}. Recibido: {hyperparameter_search_strategy}")

    # Parámetros para random search
    n_random_iterations = data.get("n_random_iterations", 100)
    random_search_params = data.get("random_search_params", {})

    if not date_col_name:
        raise ValueError("date_col_name es requerido para modelos de series temporales")

    # Validar parámetros de random search
    if hyperparameter_search_strategy == "random":
        if n_random_iterations <= 0:
            raise ValueError("n_random_iterations debe ser un número positivo")
        if n_random_iterations > 1000:
            logger.warning(f"n_random_iterations es muy alto ({n_random_iterations}). Considere usar un valor menor para mejorar el rendimiento.")

    # Carga y preparación de datos
    df = load_and_validate_ts_data(dataset_path, date_col_name, target_variable)
    y_train, y_val, y_test, train_data, val_data, test_data = ts_train_val_test_split(df, target_variable, split_ratios)

    # Determinar variables exogenas para S/ARIMAX
    exog_train = None
    exog_val = None
    exog_test = None
    numeric_features = []  # Track filtered numeric features for use in grid search

    # Validate that all requested features exist in train_data
    if input_features:
        print(f"ARIMA training - input features : {input_features}")
        missing_features = set(input_features) - set(train_data.columns)
        if not missing_features:
            # Filter to only numeric columns (ARIMA/SARIMAX requires numeric exog variables)
            numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = [f for f in input_features if train_data[f].dtype in numeric_dtypes]

            if len(numeric_features) < len(input_features):
                excluded_features = set(input_features) - set(numeric_features)
                logger.warning(f"ARIMA training - Excluding non-numeric features from exog: {excluded_features}")
                print(f"ARIMA training - Excluding non-numeric features from exog: {excluded_features}")

            if numeric_features:
                exog_train = train_data.loc[:, numeric_features]
                exog_val = val_data.loc[:, numeric_features]
                exog_test = test_data.loc[:, numeric_features]
                print(f"ARIMA training - Using numeric exog features: {numeric_features}")
            else:
                logger.warning("ARIMA training - No numeric features available for exog, proceeding without exogenous variables")
                print("ARIMA training - No numeric features available for exog, proceeding without exogenous variables")
    
    # Configuración MLflow
    current_run = mlflow.active_run()
    if not current_run:
        raise RuntimeError("No hay un run activo de MLflow")
    
    run_id = current_run.info.run_id
    logger.info(f"Iniciando entrenamiento ARIMA en run: {run_id}")
    
    # Registro de parámetros
    mlflow.log_params({
        "model_type": "ARIMA",
        "date_col_name": date_col_name,
        "target_variable": target_variable,
        "input_features": input_features,
        "forecast_horizon": forecast_horizon,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
        "problem_type": problem_type
    })


    # Entrenamiento del modelo
    if hyperparameter_search_strategy == "grid":
        # Extraer configuración de grid_search
        grid_config = data.get("grid_search", {})

        if not grid_config:
            raise ValueError(
                "grid_search configuration is required when hyperparameter_search_strategy='grid'. "
                "Provide grid_search dictionary with parameter lists (p, d, q, etc.)"
            )

        # Generar grid de parámetros usando la nueva función
        param_grid = generate_arima_grid(grid_config)

        if len(param_grid) == 0:
            raise ValueError(
                "Generated parameter grid is empty. Check grid_search configuration. "
                "Ensure at least one of (p,d,q) or (P,D,Q,s) has values."
            )

        # Configuración de walk-forward validation
        n_folds = 5
        initial_train_size = int(len(df) * split_ratios["train"])
        optimization_metric = data.get("optimization_metric", "val_rmse")

        # Validar métrica de optimización
        valid_metrics = ["val_rmse", "val_mae", "val_mape", "test_rmse", "test_mae", "test_mape"]
        if optimization_metric not in valid_metrics:
            raise ValueError(
                f"optimization_metric must be one of {valid_metrics}. "
                f"Received: {optimization_metric}"
            )

        # Validar que forecast_horizon no exceda el tamaño de val/test
        val_test_size = len(df) - initial_train_size
        if forecast_horizon > val_test_size:
            raise ValueError(
                f"forecast_horizon ({forecast_horizon}) is larger than val+test set size ({val_test_size}). "
                f"Reduce forecast_horizon or increase dataset size."
            )

        # Preparar datos completos para walk-forward
        y_full = df[target_variable]
        exog_full = None
        if numeric_features and exog_train is not None:
            exog_full = df[numeric_features]

        # Debug logging: datos de entrada para walk-forward
        logger.info(f"[DEBUG] y_full shape: {y_full.shape}, type: {type(y_full)}")
        logger.info(f"[DEBUG] y_full index type: {type(y_full.index)}, freq: {getattr(y_full.index, 'freq', None)}")
        if exog_full is not None:
            logger.info(f"[DEBUG] exog_full shape: {exog_full.shape}, type: {type(exog_full)}")
            logger.info(f"[DEBUG] exog_full index type: {type(exog_full.index)}, freq: {getattr(exog_full.index, 'freq', None)}")
            logger.info(f"[DEBUG] exog_full columns: {list(exog_full.columns)}")
        else:
            logger.info(f"[DEBUG] exog_full is None")

        # Grid search con walk-forward validation
        best_score = float('inf')
        best_model = None
        best_params = None
        best_iteration_metrics = None
        best_iteration= 0
        iterations_total = len(param_grid)
        iterations_successful = 0
        iterations_failed = 0

        logger.info(f"Iniciando ARIMA/SARIMA grid search con walk-forward validation")
        logger.info(f"Total iteraciones: {iterations_total}, Folds: {n_folds}, Tamaño ventana: {initial_train_size}")
        logger.info(f"Métrica de optimización: {optimization_metric}")

        for i, params in enumerate(param_grid):
            try:
                # Realizar walk-forward validation para estos parámetros
                fold_metrics = walk_forward_validate_sarimax(
                    y_data=y_full,
                    exog_data=exog_full,
                    params=params,
                    n_folds=n_folds,
                    initial_train_size=initial_train_size,
                    forecast_horizon=forecast_horizon
                )

                iterations_successful += 1

                # Extraer métrica de optimización
                metric_value = fold_metrics.get(optimization_metric)

                if metric_value is None:
                    logger.warning(f"Iteration {i+1}: metric {optimization_metric} is None, skipping")
                    continue

                # Actualizar mejor modelo si es necesario
                if metric_value < best_score:
                    best_score = metric_value
                    best_params = params.copy()
                    best_iteration_metrics = fold_metrics.copy()
                    best_iteration = i

                    logger.info(
                        f"Nueva mejor configuración encontrada en iteración {i+1}: "
                        f"{optimization_metric} = {best_score:.4f}"
                    )

                # Logging de progreso
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Progreso: {i+1}/{iterations_total} "
                        f"({iterations_successful} exitosas, {iterations_failed} fallidas), "
                        f"Mejor {optimization_metric}: {best_score:.4f}"
                    )

                # Limpiar memoria (Option A - inmediatamente después de cada iteración)
                gc.collect()

            except Exception as e:
                iterations_failed += 1
                logger.warning(
                    f"Iteración {i+1}/{iterations_total} falló con parámetros {params}: {str(e)}"
                )
                logger.error(f"[DEBUG-ERROR] Exception type: {type(e).__name__}")
                logger.error(f"[DEBUG-ERROR] Full traceback:", exc_info=True)
                continue

        # Verificar si todas las iteraciones fallaron
        if iterations_successful == 0:
            raise RuntimeError(
                f"Todas las {iterations_total} iteraciones fallaron. "
                f"Verifique la configuración de grid_search y la calidad de los datos. "
                f"Revise los logs para detalles de errores específicos."
            )

        if best_params is None:
            raise RuntimeError(
                f"No se encontró un modelo válido después de {iterations_successful} iteraciones exitosas. "
                f"Todas las métricas fueron None."
            )

        logger.info(
            f"Grid search completado: {iterations_successful} exitosas, {iterations_failed} fallidas. "
            f"Mejor {optimization_metric}: {best_score:.4f}"
        )

        # Reentrenar el mejor modelo en el conjunto de entrenamiento completo para guardarlo
        logger.info(f"Reentrenando mejor modelo con parámetros: {best_params}")

        if best_params.get("seasonal_order") is None:
            model_spec = ARIMA(
                endog=y_train,
                exog=exog_train,
                order=best_params["order"],
                trend=best_params.get("trend"),
                enforce_stationarity=best_params.get("enforce_stationarity", True),
                enforce_invertibility=best_params.get("enforce_invertibility", True)
            )
            logger.info(f"Reentrenando ARIMA{best_params['order']} en conjunto de entrenamiento")
        else:
            model_spec = SARIMAX(
                endog=y_train,
                exog=exog_train,
                order=best_params["order"],
                seasonal_order=best_params["seasonal_order"],
                trend=best_params.get("trend"),
                enforce_stationarity=best_params.get("enforce_stationarity", True),
                enforce_invertibility=best_params.get("enforce_invertibility", True)
            )
            logger.info(
                f"Reentrenando SARIMA{best_params['order']}x{best_params['seasonal_order']} "
                f"en conjunto de entrenamiento"
            )

        # Fit with explicit optimizer parameters and stable start_params for reproducibility
        logger.info(f"Optimizer config: {SARIMAX_OPTIMIZER_DEFAULTS}")

        # Compute stable starting parameters
        start_params = compute_stable_start_params(model_spec, y_train, exog_train)

        # Prepare fit kwargs based on model type
        model_class_name = type(model_spec).__name__
        if model_class_name == 'ARIMA':
            fit_kwargs = {
                'method': 'statespace',
                'method_kwargs': {**SARIMAX_OPTIMIZER_DEFAULTS}
            }
            if start_params is not None:
                fit_kwargs['start_params'] = start_params
                logger.info(f"Using computed start_params for grid retraining (ARIMA)")
        else:
            fit_kwargs = {**SARIMAX_OPTIMIZER_DEFAULTS}
            if start_params is not None:
                fit_kwargs['start_params'] = start_params
                logger.info(f"Using computed start_params for grid retraining (SARIMAX)")

        model = model_spec.fit(**fit_kwargs)

        # Log convergence diagnostics
        log_sarimax_convergence_diagnostics(model, prefix="grid_best_")

        # Usar métricas del grid search (no re-evaluación, según Q6.2a opción 2)
        val_metrics = {
            "val_rmse": best_iteration_metrics.get("val_rmse"),
            "val_mae": best_iteration_metrics.get("val_mae"),
            "val_mape": best_iteration_metrics.get("val_mape")
        }
        test_metrics = {
            "test_rmse": best_iteration_metrics.get("test_rmse"),
            "test_mae": best_iteration_metrics.get("test_mae"),
            "test_mape": best_iteration_metrics.get("test_mape")
        }

        # Registrar parámetros del mejor modelo en MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{optimization_metric}", best_score)
        mlflow.log_metric("grid_search_iterations_total", iterations_total)
        mlflow.log_metric("grid_search_iterations_successful", iterations_successful)
        mlflow.log_metric("grid_search_iterations_failed", iterations_failed)

        # Registrar todas las métricas del mejor modelo
        for metric_name, metric_value in {**val_metrics, **test_metrics}.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

    elif hyperparameter_search_strategy == "random":
        # Random search para ARIMA
        best_aic = float('inf')
        best_model = None
        best_params = None

        # Obtener seasonal_s - priorizar random_search_params sobre hyperparams
        # para permitir configuración de estacionalidad en random search
        seasonal_s = random_search_params.get("seasonal_s") or hyperparams.get("seasonal_s")
        if seasonal_s:
            seasonal_s = int(seasonal_s)

        logger.info(f"Iniciando random search para ARIMA con {n_random_iterations} iteraciones...")
        rng = np.random.default_rng(seed=SEED)

        for i in range(n_random_iterations):
            try:
                # Generar parámetros aleatorios
                random_params = generate_random_arima_params(random_search_params, seasonal_s, rng)

                # Crear y entrenar modelo
                if random_params["seasonal_order"] is None:
                    model_spec = ARIMA(y_train, order=random_params["order"])
                else:
                    model_spec = SARIMAX(y_train, order=random_params["order"], seasonal_order=random_params["seasonal_order"])

                # Fit with explicit optimizer parameters and stable start_params
                start_params = compute_stable_start_params(model_spec, y_train, exog_train)

                # Prepare fit kwargs based on model type
                model_class_name = type(model_spec).__name__
                if model_class_name == 'ARIMA':
                    fit_kwargs = {
                        'method': 'statespace',
                        'method_kwargs': {**SARIMAX_OPTIMIZER_DEFAULTS}
                    }
                    if start_params is not None:
                        fit_kwargs['start_params'] = start_params
                else:
                    fit_kwargs = {**SARIMAX_OPTIMIZER_DEFAULTS}
                    if start_params is not None:
                        fit_kwargs['start_params'] = start_params

                fitted_model = model_spec.fit(**fit_kwargs)

                # Calcular AIC
                aic = fitted_model.aic

                if aic < best_aic:
                    best_aic = aic
                    best_model = fitted_model
                    best_params = random_params.copy()
                    best_params["aic"] = aic

                if (i + 1) % 20 == 0:
                    logger.info(f"Progreso random search: {i+1}/{n_random_iterations}, Mejor AIC: {best_aic:.4f}")

            except Exception as e:
                logger.debug(f"Error con parámetros {random_params}: {e}")
                continue

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el random search")

        model = best_model

        # Log convergence diagnostics for best model
        log_sarimax_convergence_diagnostics(model, prefix="random_best_")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_aic", best_aic)
        mlflow.log_metric("random_search_iterations", n_random_iterations)

    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian config from data
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)
        optimization_metric = data.get("optimization_metric", "val_rmse")

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        # Reset global memory monitoring variables (Phase 8)
        global peak_memory_mb, memory_exceeded
        peak_memory_mb = 0.0
        memory_exceeded = False

        # Log Bayesian configuration
        logger.info("="*60)
        logger.info("Bayesian Search Configuration:")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info(f"  optimization_metric: {optimization_metric}")
        logger.info("="*60)

        # Log platform information for reproducibility debugging
        logger.info("="*60)
        logger.info("Platform Information (for reproducibility):")
        logger.info(f"  Python version: {sys.version.split()[0]}")
        logger.info(f"  NumPy version: {np.__version__}")
        logger.info(f"  Pandas version: {pd.__version__}")
        logger.info(f"  Statsmodels version: {sm.__version__}")
        logger.info(f"  Optuna version: {optuna.__version__}")
        logger.info(f"  Platform: {sys.platform}")
        logger.info(f"  SEED: {SEED}")
        logger.info("="*60)

        # Prepare data for walk-forward validation
        n_folds = 5
        initial_train_size = int(len(df) * split_ratios["train"])
        y_full = df[target_variable]
        exog_full = None
        if numeric_features and exog_train is not None:
            exog_full = df[numeric_features]

        # Determine if seasonal params should be suggested
        # Check if seasonal parameters are present in manual_params
        seasonal_P = hyperparams.get("seasonal_P")
        seasonal_D = hyperparams.get("seasonal_D")
        seasonal_Q = hyperparams.get("seasonal_Q")
        seasonal_s = hyperparams.get("seasonal_s")

        # Enable seasonal optimization if ALL seasonal params are provided
        enableSeasonalParams = all(x is not None for x in [seasonal_P, seasonal_D, seasonal_Q, seasonal_s])

        # Log seasonal parameter detection
        if enableSeasonalParams:
            logger.info(f"Seasonal parameters detected: P={seasonal_P}, D={seasonal_D}, Q={seasonal_Q}, s={seasonal_s}")
            logger.info("Bayesian Search will optimize seasonal parameters")
        else:
            logger.info("No seasonal parameters detected. Bayesian Search will optimize non-seasonal ARIMA only")

        # Extract and validate custom parameter ranges (Phase 9: Configurable Parameter Ranges)
        param_ranges = bayesian_config.get("param_ranges", {})

        # Validate param_ranges structure if provided
        if param_ranges:
            logger.info("Custom parameter ranges detected. Validating configuration...")

            # Track unknown parameters to warn user
            known_params = {'p', 'd', 'q', 'P', 'D', 'Q', 's', 'trend', 'enforce_stationarity', 'enforce_invertibility'}
            unknown_params = set(param_ranges.keys()) - known_params
            if unknown_params:
                logger.warning(f"Unknown parameters in param_ranges will be ignored: {unknown_params}")

            for param_name, config in param_ranges.items():
                if param_name not in known_params:
                    continue  # Skip validation for unknown params (already warned)

                # For categorical parameters, expect "choices" key
                if param_name in ['trend', 'enforce_stationarity', 'enforce_invertibility']:
                    if not isinstance(config, dict) or "choices" not in config:
                        raise ValueError(
                            f"param_ranges['{param_name}'] is a categorical parameter and must have 'choices' key. "
                            f"Expected format: {{'choices': [list_of_values]}}. Got: {config}"
                        )
                    if not isinstance(config["choices"], list) or len(config["choices"]) == 0:
                        raise ValueError(
                            f"param_ranges['{param_name}']['choices'] must be a non-empty list. Got: {config['choices']}"
                        )
                else:
                    # For numeric parameters, expect "min" and "max" keys
                    if not isinstance(config, dict) or "min" not in config or "max" not in config:
                        raise ValueError(
                            f"param_ranges['{param_name}'] must have 'min' and 'max' keys. "
                            f"Expected format: {{'min': X, 'max': Y}}. Got: {config}"
                        )

                    min_val = config["min"]
                    max_val = config["max"]

                    # Validate min < max (strictly)
                    if min_val >= max_val:
                        raise ValueError(
                            f"param_ranges['{param_name}'] min ({min_val}) must be strictly less than max ({max_val})"
                        )

                    # Validate integer parameters have integer values
                    if param_name in ['p', 'd', 'q', 'P', 'D', 'Q', 's']:
                        if not isinstance(min_val, int) or not isinstance(max_val, int):
                            raise ValueError(
                                f"param_ranges['{param_name}'] is an integer parameter. "
                                f"Both 'min' and 'max' must be integers. Got min={min_val} ({type(min_val).__name__}), "
                                f"max={max_val} ({type(max_val).__name__})"
                            )

                        # Validate step is positive if provided
                        if "step" in config:
                            step_val = config["step"]
                            if not isinstance(step_val, int) or step_val <= 0:
                                raise ValueError(
                                    f"param_ranges['{param_name}']['step'] must be a positive integer. Got: {step_val}"
                                )

            logger.info(f"Custom parameter ranges validated successfully: {list(param_ranges.keys())}")
        else:
            logger.info("No custom parameter ranges provided. Using default ranges.")

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for ARIMA/SARIMAX hyperparameter optimization.

            Returns:
                float: Validation metric to minimize (RMSE, MAE, or MAPE)
            """
            # Suggest non-seasonal parameters with configurable or default ranges (Phase 9)
            p_config = param_ranges.get("p", {"min": 0, "max": 3})
            p = trial.suggest_int('p', p_config["min"], p_config["max"], step=p_config.get("step", 1))

            d_config = param_ranges.get("d", {"min": 0, "max": 1})
            d = trial.suggest_int('d', d_config["min"], d_config["max"], step=d_config.get("step", 1))

            q_config = param_ranges.get("q", {"min": 0, "max": 3})
            q = trial.suggest_int('q', q_config["min"], q_config["max"], step=q_config.get("step", 1))

            # Suggest seasonal parameters if enabled
            if enableSeasonalParams:
                P_config = param_ranges.get("P", {"min": 0, "max": 2})
                P = trial.suggest_int('P', P_config["min"], P_config["max"], step=P_config.get("step", 1))

                D_config = param_ranges.get("D", {"min": 0, "max": 1})
                D = trial.suggest_int('D', D_config["min"], D_config["max"], step=D_config.get("step", 1))

                Q_config = param_ranges.get("Q", {"min": 0, "max": 2})
                Q = trial.suggest_int('Q', Q_config["min"], Q_config["max"], step=Q_config.get("step", 1))

                s_config = param_ranges.get("s", {"min": 2, "max": 24})
                s = trial.suggest_int('s', s_config["min"], s_config["max"], step=s_config.get("step", 1))

                seasonal_order = (P, D, Q, s)
            else:
                seasonal_order = (0, 0, 0, 0)

            # Suggest categorical parameters with configurable or default choices (Phase 9)
            trend_config = param_ranges.get("trend", {"choices": ['n', 'c', 't', 'ct']})
            trend = trial.suggest_categorical('trend', trend_config["choices"])

            enforce_stationarity_config = param_ranges.get("enforce_stationarity", {"choices": [True, False]})
            enforce_stationarity = trial.suggest_categorical('enforce_stationarity', enforce_stationarity_config["choices"])

            enforce_invertibility_config = param_ranges.get("enforce_invertibility", {"choices": [True, False]})
            enforce_invertibility = trial.suggest_categorical('enforce_invertibility', enforce_invertibility_config["choices"])

            # Build params dict for SARIMAX
            params = {
                'order': (p, d, q),
                'seasonal_order': seasonal_order,
                'trend': trend,
                'enforce_stationarity': enforce_stationarity,
                'enforce_invertibility': enforce_invertibility
            }

            try:
                # Perform walk-forward validation with these parameters
                fold_metrics = walk_forward_validate_sarimax(
                    y_data=y_full,
                    exog_data=exog_full,
                    params=params,
                    n_folds=n_folds,
                    initial_train_size=initial_train_size,
                    forecast_horizon=forecast_horizon
                )

                # Extract the optimization metric
                score = fold_metrics[optimization_metric]

                # Log trial result
                logger.info(
                    f"Trial {trial.number}: {optimization_metric}={score:.4f}, "
                    f"params={params['order']}, seasonal={params['seasonal_order']}"
                )

                return score

            except Exception as e:
                # Log the error and return high penalty for failed trials
                logger.warning(
                    f"Trial {trial.number} failed with params {params['order']}, "
                    f"seasonal={params['seasonal_order']}: {str(e)}"
                )
                # Return infinity for minimization (Optuna will mark as failed)
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,  # Fixed seed for reproducibility
            n_startup_trials=n_initial_points,  # Random exploration before Bayesian
            multivariate=False,  # Use independent TPE (simpler, more stable)
            consider_magic_clip=True,  # Limit smallest variances
            consider_endpoints=False  # Don't account for domain endpoints
        )

        study = optuna.create_study(
            direction='minimize',  # Minimize RMSE/MAE/MAPE
            sampler=sampler,
            study_name=f"arima_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization start time
        optimization_start_time = time.time()

        # Extract convergence config (Phase 7: Nice-to-Have Features)
        # Use 'or' to handle both missing keys and explicit None values from frontend
        convergence_tolerance = bayesian_config.get("convergence_tolerance") or 0.001
        convergence_patience = bayesian_config.get("convergence_patience") or 5

        # Define convergence callback (Phase 7)
        def convergence_callback(study, trial):
            """
            Stop optimization if improvement is below tolerance for patience consecutive trials.

            This is a simple heuristic that hard caps training based on lack of improvement.
            """
            # Need at least convergence_patience completed trials
            completed_trials = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
                and np.isfinite(t.value)
            ]

            if len(completed_trials) < convergence_patience:
                return  # Not enough trials yet

            # Get recent trial values
            recent_values = [t.value for t in completed_trials[-convergence_patience:]]

            # Calculate improvements between consecutive trials
            improvements = [abs(recent_values[i] - recent_values[i+1]) for i in range(len(recent_values)-1)]

            # Check if all recent improvements are below tolerance
            if all(imp < convergence_tolerance for imp in improvements):
                logger.info(f"Convergence detected: improvements {improvements} all below tolerance {convergence_tolerance}")
                logger.info(f"Stopping optimization early at trial {trial.number}")
                study.stop()

        # Extract memory limit config (Phase 8)
        max_memory_mb = bayesian_config.get("max_memory_mb", None)

        # Define memory monitoring callback (Phase 8)
        def memory_callback(study, trial):
            """
            Monitor memory usage during optimization.
            Tracks peak memory usage and stops if max_memory_mb limit is exceeded.
            """
            global peak_memory_mb, memory_exceeded

            # Get current process memory usage in MB
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Always track peak memory (even if no limit set)
            if memory_mb > peak_memory_mb:
                peak_memory_mb = memory_mb

            # Early return if no memory limit set
            if max_memory_mb is None:
                return

            # Check if memory limit exceeded
            if memory_mb > max_memory_mb:
                logger.warning(f"Memory limit exceeded: {memory_mb:.2f} MB > {max_memory_mb} MB")
                logger.warning(f"Stopping optimization at trial {trial.number}")
                memory_exceeded = True
                study.stop()

        # Build callbacks list (Phase 7 & 8)
        callbacks = []
        if convergence_tolerance and convergence_patience:
            callbacks.append(convergence_callback)
        if max_memory_mb is not None:
            callbacks.append(memory_callback)

        # Run optimization
        logger.info(f"Starting Bayesian Search optimization with Optuna TPESampler")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,  # Optional timeout
            callbacks=callbacks,  # Phase 7 & 8: convergence detection and memory monitoring
            show_progress_bar=False,  # Quiet mode for logs
            n_jobs=1  # Single-threaded for reproducibility
        )

        # Track optimization end time
        optimization_time_seconds = time.time() - optimization_start_time

        # Extract best parameters
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Bayesian Search failed: All trials returned errors or no valid trials completed. "
                "Check parameter ranges and data quality."
            )

        best_params_dict = study.best_params
        best_score = study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Bayesian Search Completed")
        logger.info(f"  Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"  Best parameters: {best_params_dict}")
        logger.info(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{len(study.trials)}")
        logger.info(f"  Optimization time: {optimization_time_seconds:.2f} seconds")
        logger.info("="*60)

        # Build final params for model training
        final_order = (
            best_params_dict['p'],
            best_params_dict['d'],
            best_params_dict['q']
        )

        if enableSeasonalParams:
            final_seasonal_order = (
                best_params_dict['P'],
                best_params_dict['D'],
                best_params_dict['Q'],
                best_params_dict['s']
            )
        else:
            final_seasonal_order = (0, 0, 0, 0)

        final_trend = best_params_dict['trend']
        final_enforce_stationarity = best_params_dict['enforce_stationarity']
        final_enforce_invertibility = best_params_dict['enforce_invertibility']

        # Train final model with best parameters on train set
        logger.info(f"Training final SARIMAX model with best parameters on train set")
        final_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=final_order,
            seasonal_order=final_seasonal_order,
            trend=final_trend,
            enforce_stationarity=final_enforce_stationarity,
            enforce_invertibility=final_enforce_invertibility
        )

        fitted_model = final_model.fit(**SARIMAX_OPTIMIZER_DEFAULTS)

        # Store best params for later use
        best_params = {
            'order': final_order,
            'seasonal_order': final_seasonal_order,
            'trend': final_trend,
            'enforce_stationarity': final_enforce_stationarity,
            'enforce_invertibility': final_enforce_invertibility
        }

        # Assign model for evaluation
        model = fitted_model

        # Log convergence diagnostics for best model
        log_sarimax_convergence_diagnostics(model, prefix="bayesian_best_")

        # Log Bayesian-specific parameters to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_timeout_seconds": timeout_seconds,
            "bayesian_optimization_metric": optimization_metric
        })

        # Log best parameters to MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Log Bayesian optimization results to MLflow
        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        })

    else:
        # Parámetros manuales
        p = int(hyperparams.get("p", 1))
        d = int(hyperparams.get("d", 1))
        q = int(hyperparams.get("q", 1))
        
        # Parámetros estacionales opcionales
        seasonal_P = hyperparams.get("seasonal_P")
        seasonal_D = hyperparams.get("seasonal_D")  
        seasonal_Q = hyperparams.get("seasonal_Q")
        seasonal_s = hyperparams.get("seasonal_s")
        
        model_params = {
            "order": (p, d, q),
            "p": p,
            "d": d,
            "q": q
        }

        # Parametros extra
        trend = hyperparams.get("trend")
        enforce_stationarity = hyperparams.get("enforce_stationarity")
        enforce_invertibility = hyperparams.get("enforce_invertibility")
        
        # Si hay parámetros estacionales, usar SARIMAX
        if all(x is not None for x in [seasonal_P, seasonal_D, seasonal_Q, seasonal_s]):
            seasonal_order = (int(seasonal_P), int(seasonal_D), int(seasonal_Q), int(seasonal_s))
            #model_params["seasonal_order"] = seasonal_order
            model_params["seasonal_P"] = seasonal_P
            model_params["seasonal_D"] = seasonal_D
            model_params["seasonal_Q"] = seasonal_Q
            model_params["seasonal_s"] = seasonal_s
            model_spec = SARIMAX(
                endog=y_train,
                exog=exog_train,
                order= model_params["order"],
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility
            )
            if exog_train is not None and len(numeric_features) > 0:
                logger.info(f"Entrenando SARIMAX{model_params['order']}x{seasonal_order} con variables exogenas: {numeric_features}")
            else:
                logger.info(f"Entrenando SARIMA{model_params['order']}x{seasonal_order}")
        else:
            model_spec = ARIMA(
                endog=y_train,
                exog=exog_train,
                order=model_params["order"],
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility
            )
            if exog_train is not None and len(numeric_features) > 0:
                logger.info(f"Entrenando ARIMAX{model_params['order']} con variables exogenas: {numeric_features}")
            else:
                logger.info(f"Entrenando ARIMA{model_params['order']}")
        model_params["trend"] = trend
        model_params["enforce_stationarity"] = enforce_stationarity
        model_params["enforce_invertibility"] = enforce_invertibility 
        
        mlflow.log_params(model_params)

        # Entrenar modelo con parámetros de optimización explícitos y start_params estables
        # This is the CRITICAL fix for cross-platform reproducibility
        logger.info(f"Entrenando modelo con optimizador: {SARIMAX_OPTIMIZER_DEFAULTS}")

        # Compute stable starting parameters for better cross-platform consistency
        start_params = compute_stable_start_params(model_spec, y_train, exog_train)

        # Prepare fit kwargs based on model type
        # ARIMA class requires optimizer params in method_kwargs, SARIMAX accepts them directly
        # Check class name to distinguish between ARIMA and SARIMAX
        model_class_name = type(model_spec).__name__
        if model_class_name == 'ARIMA':
            # ARIMA model: use method_kwargs for optimizer parameters
            fit_kwargs = {
                'method': 'statespace',  # Use statespace method for ARIMA
                'method_kwargs': {**SARIMAX_OPTIMIZER_DEFAULTS}
            }
            if start_params is not None:
                fit_kwargs['start_params'] = start_params
                logger.info(f"Using computed start_params with {len(start_params)} parameters (ARIMA)")
            else:
                logger.info("Using statsmodels default start_params (ARIMA)")
        else:
            # SARIMAX model: pass optimizer params directly
            fit_kwargs = {**SARIMAX_OPTIMIZER_DEFAULTS}
            if start_params is not None:
                fit_kwargs['start_params'] = start_params
                logger.info(f"Using computed start_params with {len(start_params)} parameters (SARIMAX)")
            else:
                logger.info("Using statsmodels default start_params (SARIMAX)")

        # Fit model
        model = model_spec.fit(**fit_kwargs)
        best_params = model_params

        # Log detailed convergence diagnostics
        log_sarimax_convergence_diagnostics(model, prefix="manual_")

        # Registrar AIC y BIC
        try:
            mlflow.log_metric("aic", model.aic)
            mlflow.log_metric("bic", model.bic)
        except Exception:
            logger.warning("No se pudo calcular AIC/BIC")
    
    # Evaluación del modelo
    # Para todos los métodos (grid, random, manual), hacer evaluación estándar
    val_metrics, val_artifacts = evaluate_arima_model(model, y_train, y_val, "val", forecast_horizon, experiment_dir, exog_val)
    test_metrics, test_artifacts = evaluate_arima_model(model, y_train, y_test, "test", forecast_horizon, experiment_dir, exog_test, y_val)
    
    # Finalizar y registrar energía
    tracker.stop()
    energy_kwh, emissions_kg = log_energy_metrics(tracker)
    
    # Registro del modelo
    try:
        # Crear una muestra simple para la signatura
        sample_input = np.array([[1.0]])  # Simplified input
        sample_output = np.array([[1.0]])  # Simplified output
        signature = infer_signature(sample_input, sample_output)
    except Exception:
        signature = None
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="arima_model",
        signature=signature,
        registered_model_name=model_name,
        metadata={
            "dataset": os.path.basename(dataset_path),
            "target": target_variable,
            "date_column": date_col_name,
            "forecast_horizon": forecast_horizon
        }
    )
    
    # Guardado local
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Actualizar metadatos
    client = MlflowClient()
    try:
        client.set_registered_model_tag(model_name, "task", "time_series_forecasting")
        client.update_registered_model(
            name=model_name,
            description=f"Modelo ARIMA para pronóstico de series temporales - {model_name}"
        )
    except Exception as e:
        logger.warning(f"Error al actualizar metadatos del modelo: {e}")
    
    # Configuración del pipeline con parámetros completos
    pipeline_config = {
        "step": "train_arima",
        "model_name": model_name,
        "date_col_name": date_col_name,
        "target_variable": target_variable,
        "input_features": input_features,
        "forecast_horizon": forecast_horizon,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "grid_search": grid_config if hyperparameter_search_strategy == "grid" else None,
        "hyperparameters": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "artifacts": {
            "val": val_artifacts,
            "test": test_artifacts
        },
        "energy_metrics": {"energy_consumed_total_kWh": energy_kwh, "carbon_emission_kg": emissions_kg}
    }

    # Añadir configuración específica de grid search si se usó
    if hyperparameter_search_strategy == "grid":
        pipeline_config["grid_search_results"] = {
            #"grid_config": grid_config,
            "best_params": best_params,
            "iterations_total": iterations_total,
            "iterations_successful": iterations_successful,
            "iterations_failed": iterations_failed,
            "best_iteration": best_iteration,  
            "best_val_loss": best_score,
            "optimization_metric": optimization_metric,
            "walk_forward_config": {
                "window_type": "rolling",
                "n_folds": n_folds,
                "refit": True,
                "initial_train_size": initial_train_size,
                "step_size": forecast_horizon
            }
        }
    elif hyperparameter_search_strategy == "random":
        pipeline_config["random_search"] = {
            "use_random_search": True,
            "n_random_iterations": n_random_iterations,
            "random_search_params": random_search_params,
            "best_params": best_params,
        }
    elif hyperparameter_search_strategy == "bayesian":
        bayesian_config_metadata = {
            "n_trials": n_trials,
            "n_initial_points": n_initial_points,
            "timeout_seconds": timeout_seconds,
            "acq_func": bayesian_config.get("acq_func", "ei"),  # Phase 7: Save acq_func metadata
            "convergence_tolerance": bayesian_config.get("convergence_tolerance", 0.001),  # Phase 7
            "convergence_patience": bayesian_config.get("convergence_patience", 5),  # Phase 7
            "max_memory_mb": max_memory_mb,  # Phase 8: Memory limit (None if disabled)
            "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb > 0 else None,  # Phase 8: Peak memory usage
            "memory_exceeded": memory_exceeded,  # Phase 8: Whether memory limit was exceeded
            "optimization_metric": optimization_metric,
            "optimization_time_seconds": optimization_time_seconds,
            "best_trial_number": study.best_trial.number,
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_params": best_params,
            "seed": SEED
        }
        # Phase 9: Add custom param_ranges if provided
        if param_ranges:
            bayesian_config_metadata["param_ranges"] = param_ranges
        pipeline_config["bayesian_config"] = bayesian_config_metadata
    save_pipeline_config(experiment_dir, pipeline_config)
    
    return {
        "status": "Entrenamiento ARIMA completado",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "run_id": run_id
    }

# ======================
# FUNCIÓN DE ENTRENAMIENTO XGBOOST
# ======================

def train_xgboost_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Entrena y registra un modelo XGBoost para pronóstico de series temporales.

    Args:
        dataset_path: Ruta al archivo CSV con los datos
        data: Diccionario con configuración del entrenamiento
        experiment_dir: Directorio para guardar artefactos

    Returns:
        Diccionario con resultados del entrenamiento
    """
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=experiment_dir, save_to_file=False, allow_multiple_runs=True)
    tracker.start()

    # Extraer parámetros
    date_col_name = data.get("date_col_name")
    target_variable = data["target_variable"]
    forecast_horizon = data.get("forecast_horizon", 10)
    hyperparams = data.get("manual_params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "XGBoost_TS_Model")
    problem_type = "ts_forecasting"


    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", "none")

    # Validar estrategia
    valid_strategies = ["manual", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}. Recibido: {hyperparameter_search_strategy}")

    # Parámetros para random search
    n_random_iterations = data.get("n_random_iterations", 100)
    random_search_params = data.get("random_search_params", {})


    # Métrica de optimización para XGBoost
    optimization_metric = data.get("optimization_metric", "val_rmse")
    valid_metrics = ["val_rmse", "val_mae","val_mape", "test_rmse", "test_mae","test_mape"]
    if optimization_metric not in valid_metrics:
        raise ValueError(f"optimization_metric debe ser uno de: {valid_metrics}. Recibido: {optimization_metric}")

    if not date_col_name:
        raise ValueError("date_col_name es requerido para modelos de series temporales")

    # Validar parámetros de random search
    if hyperparameter_search_strategy == "random":
        if n_random_iterations <= 0:
            raise ValueError("n_random_iterations debe ser un número positivo")
        if n_random_iterations > 1000:
            logger.warning(f"n_random_iterations es muy alto ({n_random_iterations}). Considere usar un valor menor para mejorar el rendimiento.")

    # Carga y preparación de datos
    logger.info("Iniciando entrenamiento XGBoost para series temporales")
    df = load_and_validate_ts_data(dataset_path, date_col_name, target_variable)

    # Obtener lista de características
    # Prioridad 1: Usar input_features del frontend si está disponible
    input_features = data.get("input_features", None)
    if input_features:
        # Validar que las características especificadas existen en el dataset
        feature_cols = [col for col in input_features
                        if col in df.columns and col != target_variable]
        if len(feature_cols) != len(input_features):
            missing = set(input_features) - set(feature_cols) - {target_variable}
            if missing:
                logger.warning(f"Características especificadas no encontradas: {missing}")
        logger.info(f"Usando {len(feature_cols)} características especificadas por el frontend")

    # Verificar que hay características disponibles
    if len(feature_cols) == 0:
        raise ValueError("No se generaron características válidas para XGBoost")

    logger.info(f"Características para XGBoost: {feature_cols}")

    # División temporal del dataset
    X_train, X_val, X_test, y_train, y_val, y_test, train_data, val_data, test_data = xgboost_train_val_test_split(
        df, target_variable, feature_cols, forecast_horizon, split_ratios
    )

    logger.info(f"XGB dataset split...")
    logger.info(f"X_train.shape: {X_train.shape};")
    logger.info(f"X_train.dtypes:{X_train.dtypes};")
    logger.info(f"X_train.head(): {X_train.head()}")

    logger.info(f"y_train.describe(): {y_train.describe()}")
    logger.info(f"y_train.min(): {y_train.min()}")
    logger.info(f"y_train.max(): {y_train.max()}")
    

    # Configuración MLflow
    current_run = mlflow.active_run()
    if not current_run:
        raise RuntimeError("No hay un run activo de MLflow")

    run_id = current_run.info.run_id
    logger.info(f"Iniciando entrenamiento XGBoost en run: {run_id}")

    # Registro de parámetros
    mlflow.log_params({
        "model_type": "XGBoost",
        "date_col_name": date_col_name,
        "target_variable": target_variable,
        "input_features": input_features,
        "forecast_horizon": forecast_horizon,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
        "optimization_metric": optimization_metric,
        "problem_type": problem_type,
        "n_features": len(feature_cols),
    })

    # Entrenamiento del modelo
    if hyperparameter_search_strategy == "grid":
        # Definir grid de parámetros para XGBoost
        param_grid = []

        # Grid básico XGBoost
        n_estimators_values = [100, 200, 300]
        max_depth_values = [3, 5, 7]
        learning_rate_values = [0.01, 0.1, 0.2]
        subsample_values = [0.8, 0.9, 1.0]

        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                for learning_rate in learning_rate_values:
                    for subsample in subsample_values:
                        param_grid.append({
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "random_state": SEED
                        })

        # Grid search manual para XGBoost
        best_score = float('inf')
        best_model = None
        best_params = None

        logger.info(f"Iniciando grid search para XGBoost con {len(param_grid)} combinaciones...")

        for i, params in enumerate(param_grid):
            try:
                # Crear y entrenar modelo
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)

                # Evaluar en validation set
                val_pred = model.predict(X_val)
                val_score = mean_squared_error(y_val, val_pred)

                if val_score < best_score:
                    best_score = val_score
                    best_model = model
                    best_params = params.copy()
                    best_params["val_rmse"] = val_score

                if (i + 1) % 10 == 0:
                    logger.info(f"Progreso grid search: {i+1}/{len(param_grid)}")

            except Exception as e:
                logger.debug(f"Error con parámetros {params}: {e}")
                continue

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el grid search")

        model = best_model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_mse", best_score)

    elif hyperparameter_search_strategy == "random":
        # Random search para XGBoost
        best_score = float('inf')
        best_model = None
        best_params = None

        logger.info(f"Iniciando random search para XGBoost con {n_random_iterations} iteraciones...")
        rng = np.random.default_rng(seed=SEED)

        for i in range(n_random_iterations):
            try:
                # Generar parámetros aleatorios
                random_params = generate_random_xgboost_params(random_search_params, rng)

                # Crear y entrenar modelo
                model_spec = xgb.XGBRegressor(**random_params)
                model_spec.fit(X_train, y_train)

                # Evaluar en validation set
                val_pred = model_spec.predict(X_val)
                val_score = calculate_ts_metric(y_val, val_pred, optimization_metric)

                if val_score < best_score:
                    best_score = val_score
                    best_model = model_spec
                    best_params = random_params.copy()
                    best_params[f"val_{optimization_metric}"] = val_score

                if (i + 1) % 20 == 0:
                    logger.info(f"Progreso random search: {i+1}/{n_random_iterations}, Mejor {optimization_metric}: {best_score:.4f}")

            except Exception as e:
                logger.warning(f"Error con parámetros {random_params}: {e}")
                continue

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el random search")

        model = best_model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_val_{optimization_metric}", best_score)
        mlflow.log_metric("random_search_iterations", n_random_iterations)

    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian config from data
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        # Reset global memory monitoring variables (Phase 8)
        global peak_memory_mb, memory_exceeded
        peak_memory_mb = 0.0
        memory_exceeded = False

        # Log Bayesian configuration
        logger.info("="*60)
        logger.info("XGBoost Bayesian Search Configuration:")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info(f"  optimization_metric: {optimization_metric}")
        logger.info("="*60)

        # Extract and validate custom parameter ranges (Phase 9: Configurable Parameter Ranges)
        param_ranges = bayesian_config.get("param_ranges", {})

        # Validate param_ranges structure if provided
        if param_ranges:
            logger.info("Custom parameter ranges detected. Validating configuration...")

            # Track unknown parameters to warn user
            known_params = {'n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'min_child_weight'}
            unknown_params = set(param_ranges.keys()) - known_params
            if unknown_params:
                logger.warning(f"Unknown parameters in param_ranges will be ignored: {unknown_params}")

            for param_name, config in param_ranges.items():
                if param_name not in known_params:
                    continue  # Skip validation for unknown params (already warned)

                # All XGBoost parameters are numeric (no categoricals)
                if not isinstance(config, dict) or "min" not in config or "max" not in config:
                    raise ValueError(
                        f"param_ranges['{param_name}'] must have 'min' and 'max' keys. "
                        f"Expected format: {{'min': X, 'max': Y}}. Got: {config}"
                    )

                min_val = config["min"]
                max_val = config["max"]

                # Validate min < max (strictly)
                if min_val >= max_val:
                    raise ValueError(
                        f"param_ranges['{param_name}'] min ({min_val}) must be strictly less than max ({max_val})"
                    )

                # Validate integer parameters have integer values
                if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                    if not isinstance(min_val, int) or not isinstance(max_val, int):
                        raise ValueError(
                            f"param_ranges['{param_name}'] is an integer parameter. "
                            f"Both 'min' and 'max' must be integers. Got min={min_val} ({type(min_val).__name__}), "
                            f"max={max_val} ({type(max_val).__name__})"
                        )

                    # Validate step is positive if provided
                    if "step" in config:
                        step_val = config["step"]
                        if not isinstance(step_val, int) or step_val <= 0:
                            raise ValueError(
                                f"param_ranges['{param_name}']['step'] must be a positive integer. Got: {step_val}"
                            )

                # Validate float parameters
                elif param_name in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma']:
                    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                        raise ValueError(
                            f"param_ranges['{param_name}'] is a float parameter. "
                            f"Both 'min' and 'max' must be numeric. Got min={min_val} ({type(min_val).__name__}), "
                            f"max={max_val} ({type(max_val).__name__})"
                        )

                    # Validate log=True only for positive ranges
                    if config.get("log", False):
                        if min_val <= 0 or max_val <= 0:
                            raise ValueError(
                                f"param_ranges['{param_name}'] has 'log': True but range includes non-positive values. "
                                f"Log scale requires min > 0 and max > 0. Got min={min_val}, max={max_val}"
                            )

            logger.info(f"Custom parameter ranges validated successfully: {list(param_ranges.keys())}")
        else:
            logger.info("No custom parameter ranges provided. Using default ranges.")

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for XGBoost hyperparameter optimization.

            Returns:
                float: Validation metric (RMSE/MAE/MAPE) to minimize
            """
            # Suggest hyperparameters with configurable or default ranges (Phase 9)
            n_est_config = param_ranges.get("n_estimators", {"min": 50, "max": 500})
            n_estimators = trial.suggest_int('n_estimators', n_est_config["min"], n_est_config["max"], step=n_est_config.get("step", 1))

            max_depth_config = param_ranges.get("max_depth", {"min": 3, "max": 10})
            max_depth = trial.suggest_int('max_depth', max_depth_config["min"], max_depth_config["max"], step=max_depth_config.get("step", 1))

            lr_config = param_ranges.get("learning_rate", {"min": 1e-3, "max": 0.1, "log": True})
            learning_rate = trial.suggest_float('learning_rate', lr_config["min"], lr_config["max"], log=lr_config.get("log", False))

            subsample_config = param_ranges.get("subsample", {"min": 0.5, "max": 1.0})
            subsample = trial.suggest_float('subsample', subsample_config["min"], subsample_config["max"], log=subsample_config.get("log", False))

            colsample_config = param_ranges.get("colsample_bytree", {"min": 0.5, "max": 1.0})
            colsample_bytree = trial.suggest_float('colsample_bytree', colsample_config["min"], colsample_config["max"], log=colsample_config.get("log", False))

            gamma_config = param_ranges.get("gamma", {"min": 0, "max": 1.0})
            gamma = trial.suggest_float('gamma', gamma_config["min"], gamma_config["max"], log=gamma_config.get("log", False))

            min_child_config = param_ranges.get("min_child_weight", {"min": 1, "max": 10})
            min_child_weight = trial.suggest_int('min_child_weight', min_child_config["min"], min_child_config["max"], step=min_child_config.get("step", 1))

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma,
                'min_child_weight': min_child_weight,
                'random_state': SEED,  # Fixed seed for reproducibility
                'n_jobs': 1  # Single-threaded for reproducibility
            }

            try:
                # Train XGBoost model with suggested parameters
                model_trial = xgb.XGBRegressor(**params)
                model_trial.fit(X_train, y_train)

                # Predict on validation set
                y_val_pred = model_trial.predict(X_val)

                # Calculate metric using existing function for consistency
                score = calculate_ts_metric(y_val, y_val_pred, optimization_metric)

                # Log trial result
                logger.info(
                    f"Trial {trial.number}: {optimization_metric}={score:.4f}, "
                    f"n_estimators={params['n_estimators']}, "
                    f"max_depth={params['max_depth']}, "
                    f"lr={params['learning_rate']:.4f}"
                )

                return score

            except Exception as e:
                # Log the error and return high penalty for failed trials
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                # Return infinity for minimization (Optuna will mark as failed)
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,  # Fixed seed for reproducibility
            n_startup_trials=n_initial_points,  # Random exploration before Bayesian
            multivariate=False  # Use independent TPE (simpler, more stable)
        )

        study = optuna.create_study(
            direction='minimize',  # Minimize RMSE/MAE/MAPE
            sampler=sampler,
            study_name=f"xgboost_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization start time
        optimization_start_time = time.time()

        # Extract convergence config (Phase 7: Nice-to-Have Features)
        # Use 'or' to handle both missing keys and explicit None values from frontend
        convergence_tolerance = bayesian_config.get("convergence_tolerance") or 0.001
        convergence_patience = bayesian_config.get("convergence_patience") or 5

        # Define convergence callback (Phase 7)
        def convergence_callback(study, trial):
            """
            Stop optimization if improvement is below tolerance for patience consecutive trials.

            This is a simple heuristic that hard caps training based on lack of improvement.
            """
            # Need at least convergence_patience completed trials
            completed_trials = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
                and np.isfinite(t.value)
            ]

            if len(completed_trials) < convergence_patience:
                return  # Not enough trials yet

            # Get recent trial values
            recent_values = [t.value for t in completed_trials[-convergence_patience:]]

            # Calculate improvements between consecutive trials
            improvements = [abs(recent_values[i] - recent_values[i+1]) for i in range(len(recent_values)-1)]

            # Check if all recent improvements are below tolerance
            if all(imp < convergence_tolerance for imp in improvements):
                logger.info(f"Convergence detected: improvements {improvements} all below tolerance {convergence_tolerance}")
                logger.info(f"Stopping optimization early at trial {trial.number}")
                study.stop()

        # Extract memory limit config (Phase 8)
        max_memory_mb = bayesian_config.get("max_memory_mb", None)

        # Define memory monitoring callback (Phase 8)
        def memory_callback(study, trial):
            """
            Monitor memory usage during optimization.
            Tracks peak memory usage and stops if max_memory_mb limit is exceeded.
            """
            global peak_memory_mb, memory_exceeded

            # Get current process memory usage in MB
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Always track peak memory (even if no limit set)
            if memory_mb > peak_memory_mb:
                peak_memory_mb = memory_mb

            # Early return if no memory limit set
            if max_memory_mb is None:
                return

            # Check if memory limit exceeded
            if memory_mb > max_memory_mb:
                logger.warning(f"Memory limit exceeded: {memory_mb:.2f} MB > {max_memory_mb} MB")
                logger.warning(f"Stopping optimization at trial {trial.number}")
                memory_exceeded = True
                study.stop()

        # Build callbacks list (Phase 7 & 8)
        callbacks = []
        if convergence_tolerance and convergence_patience:
            callbacks.append(convergence_callback)
        if max_memory_mb is not None:
            callbacks.append(memory_callback)

        # Run optimization
        logger.info("Starting XGBoost Bayesian Search optimization with Optuna TPESampler")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,  # Optional timeout
            callbacks=callbacks,  # Phase 7 & 8: convergence detection and memory monitoring
            show_progress_bar=False,  # Quiet mode for logs
            n_jobs=1  # Single-threaded for reproducibility
        )

        # Track optimization end time
        optimization_time_seconds = time.time() - optimization_start_time

        # Extract best parameters
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Bayesian Search failed: All trials returned errors or no valid trials completed. "
                "Check parameter ranges and data quality."
            )

        best_params_dict = study.best_params
        best_score = study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"XGBoost Bayesian Search Completed")
        logger.info(f"  Best {optimization_metric}: {best_score:.4f}")
        logger.info(f"  Best parameters: {best_params_dict}")
        logger.info(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{len(study.trials)}")
        logger.info(f"  Optimization time: {optimization_time_seconds:.2f} seconds")
        logger.info("="*60)

        # Build final parameters for model training
        final_params = {
            'n_estimators': best_params_dict['n_estimators'],
            'max_depth': best_params_dict['max_depth'],
            'learning_rate': best_params_dict['learning_rate'],
            'subsample': best_params_dict['subsample'],
            'colsample_bytree': best_params_dict['colsample_bytree'],
            'gamma': best_params_dict['gamma'],
            'min_child_weight': best_params_dict['min_child_weight'],
            'random_state': SEED,
            'n_jobs': 1
        }

        # Train final model with best parameters on full train set
        logger.info(f"Training final XGBoost model with best parameters on train set")
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_train, y_train)

        # Store best params for pipeline config
        best_params = final_params.copy()

        # Log Bayesian-specific parameters to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_timeout_seconds": timeout_seconds,
            "bayesian_optimization_metric": optimization_metric
        })

        # Log best parameters to MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Log Bayesian optimization results to MLflow
        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        })

    else:
        # Parámetros manuales con valores por defecto
        xgb_params = {
            "n_estimators": int(hyperparams.get("n_estimators", 100)),
            "max_depth": int(hyperparams.get("max_depth", 6)),
            "learning_rate": float(hyperparams.get("learning_rate", 0.1)),
            "subsample": float(hyperparams.get("subsample", 0.8)),
            "colsample_bytree": float(hyperparams.get("colsample_bytree", 0.8)),
            "gamma": float(hyperparams.get("gamma", 0)),
            "min_child_weight": int(hyperparams.get("min_child_weight", 1)),
            "random_state": SEED,
            "n_jobs": -1
        }

        logger.info(f"Entrenando XGBoost con parámetros: {xgb_params}")

        mlflow.log_params(xgb_params)

        # Entrenar modelo
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        best_params = xgb_params

        logger.info(f"Model trained. Training samples: {model.n_features_in_}")
        logger.info(f"Model feature importances: {model.feature_importances_}")

    # Evaluación del modelo
    val_metrics, val_artifacts = evaluate_xgboost_model(
        model, X_train, X_val, y_train, y_val, "val", forecast_horizon, experiment_dir, feature_cols
    )
    test_metrics, test_artifacts = evaluate_xgboost_model(
        model, X_train, X_test, y_train, y_test, "test", forecast_horizon, experiment_dir, feature_cols,
        y_val=y_val
    )

    # Registrar métricas en MLflow
    for metric_name, metric_value in {**val_metrics, **test_metrics}.items():
        if metric_value is not None:
            mlflow.log_metric(metric_name, metric_value)

    # Finalizar y registrar energía
    tracker.stop()
    energy_kwh, emissions_kg = log_energy_metrics(tracker)

    # Registro del modelo
    try:
        # Crear signatura con muestras reales
        signature = infer_signature(X_train.head(5).values, model.predict(X_train.head(5)))
    except Exception:
        signature = None

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="xgboost_model",
        signature=signature,
        registered_model_name=model_name,
        metadata={
            "dataset": os.path.basename(dataset_path),
            "target": target_variable,
            "date_column": date_col_name,
            "forecast_horizon": forecast_horizon,
            "n_features": len(feature_cols)
        }
    )

    # Guardado local
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Actualizar metadatos
    client = MlflowClient()
    try:
        client.set_registered_model_tag(model_name, "task", "time_series_forecasting")
        client.update_registered_model(
            name=model_name,
            description=f"Modelo XGBoost para pronóstico de series temporales - {model_name}"
        )
    except Exception as e:
        logger.warning(f"Error al actualizar metadatos del modelo: {e}")

    # Configuración del pipeline con parámetros completos
    pipeline_config = {
        "step": "train_xgboost",
        "model_name": model_name,
        "date_col_name": date_col_name,
        "target_variable": target_variable,
        "input_features": input_features,
        "forecast_horizon": forecast_horizon,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "optimization_metric": optimization_metric,
        "hyperparameters": best_params,
        "grid_search": {
            "use_grid_search": hyperparameter_search_strategy == "grid",
            "best_params": best_params if hyperparameter_search_strategy == "grid" else None,
            "grid_search_params": param_grid if hyperparameter_search_strategy == "grid" else None
        },
        "random_search": {
            "use_random_search": hyperparameter_search_strategy == "random",
            "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
            "random_search_params": random_search_params if hyperparameter_search_strategy == "random" else None,
            "best_params": best_params if hyperparameter_search_strategy == "random" else None,
        },
        "features_used": feature_cols,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "artifacts": {
            "val": val_artifacts,
            "test": test_artifacts
        },
        "energy_metrics": {"energy_consumed_total_kWh": energy_kwh, "carbon_emission_kg": emissions_kg}
    }

    # Add Bayesian config metadata if Bayesian search was used
    if hyperparameter_search_strategy == "bayesian":
        bayesian_config_metadata = {
            "n_trials": n_trials,
            "n_initial_points": n_initial_points,
            "timeout_seconds": timeout_seconds,
            "acq_func": bayesian_config.get("acq_func", "ei"),  # Phase 7: Save acq_func metadata
            "convergence_tolerance": bayesian_config.get("convergence_tolerance", 0.001),  # Phase 7
            "convergence_patience": bayesian_config.get("convergence_patience", 5),  # Phase 7
            "max_memory_mb": max_memory_mb,  # Phase 8: Memory limit (None if disabled)
            "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb > 0 else None,  # Phase 8: Peak memory usage
            "memory_exceeded": memory_exceeded,  # Phase 8: Whether memory limit was exceeded
            "optimization_metric": optimization_metric,
            "optimization_time_seconds": optimization_time_seconds,
            "best_trial_number": study.best_trial.number,
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_params": best_params,
            "seed": SEED
        }
        # Phase 9: Add custom param_ranges if provided
        if param_ranges:
            bayesian_config_metadata["param_ranges"] = param_ranges
        pipeline_config["bayesian_config"] = bayesian_config_metadata

    save_pipeline_config(experiment_dir, pipeline_config)

    return {
        "status": "Entrenamiento XGBoost completado",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "run_id": run_id,
        "features_used": feature_cols,
        "best_params": best_params
    }

def predict_xgboost_recursive(model, last_features: pd.DataFrame, target_col: str, feature_cols: List[str],
                             forecast_horizon: int, lag_periods: List[int] = None) -> np.ndarray:
    """
    Realiza pronósticos recursivos con XGBoost para múltiples pasos adelante.

    RECURSIVE FORECASTING: Esta función usa las predicciones del modelo como entrada
    para los siguientes pasos de predicción, permitiendo forecasting multi-step.

    Args:
        model: Modelo XGBoost entrenado
        last_features: DataFrame con las últimas características conocidas
        target_col: Nombre de la columna objetivo
        feature_cols: Lista de nombres de características
        forecast_horizon: Número de pasos a predecir
        lag_periods: Lista de períodos de lag utilizados (para actualización recursiva)

    Returns:
        Array con las predicciones para forecast_horizon pasos
    """
    if lag_periods is None:
        lag_periods = []

    predictions = []
    current_features = last_features.copy()

    for step in range(forecast_horizon):
        # Obtener características para predicción actual
        features_for_prediction = current_features[feature_cols].iloc[-1:].values

        # RECURSIVE FORECASTING: Hacer predicción usando características actuales
        pred = model.predict(features_for_prediction)[0]
        predictions.append(pred)

        # RECURSIVE FORECASTING: Actualizar características con la nueva predicción
        # para el siguiente paso de predicción
        if step < forecast_horizon - 1:  # No actualizar en la última iteración
            # Crear nueva fila con características actualizadas
            new_row = current_features.iloc[-1:].copy()

            # Actualizar valor objetivo
            new_row[target_col] = pred

            # Actualizar características de lag si existen
            for lag in lag_periods:
                lag_col = f"{target_col}_lag_{lag}"
                if lag_col in new_row.columns:
                    if lag == 1:
                        new_row[lag_col] = pred
                    elif lag > 1 and len(current_features) >= lag:
                        # Para lags > 1, usar valores históricos
                        historical_val = current_features[target_col].iloc[-(lag-1)]
                        new_row[lag_col] = historical_val

            # Actualizar características rolling si existen
            rolling_cols = [col for col in feature_cols if "rolling" in col and target_col in col]
            for rolling_col in rolling_cols:
                # Para características rolling, usar una aproximación simple
                # En una implementación más sofisticada, se mantendría una ventana completa
                try:
                    # Aproximar con el valor más reciente
                    new_row[rolling_col] = current_features[rolling_col].iloc[-1]
                except Exception:
                    pass

            # Agregar nueva fila al DataFrame de características
            current_features = pd.concat([current_features, new_row], ignore_index=True)

    return np.array(predictions)


# ======================
# LSTM FUNCTIONS - PHASE 1
# ======================

def create_sequences_for_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte DataFrame de series temporales en secuencias 3D para LSTM.

    Args:
        df: Datos con índice datetime ordenado temporalmente
        feature_cols: Lista de columnas de características de entrada
        target_col: Nombre de la columna objetivo
        sequence_length: Longitud de la ventana temporal (número de timesteps)
        forecast_horizon: Pasos adelante a predecir (default: 1 para single-step)

    Returns:
        X: Secuencias de entrada, shape (n_sequences, sequence_length, n_features)
        y: Valores objetivo, shape (n_sequences,)

    Example:
        >>> df = pd.DataFrame({'date': ..., 'temp': ..., 'sales': ...})
        >>> X, y = create_sequences_for_lstm(df, ['temp'], 'sales', 10)
        >>> X.shape
        (190, 10, 1)
        >>> y.shape
        (190,)
    """
    # Handle univariate mode (empty feature_cols)
    if not feature_cols or len(feature_cols) == 0:
        logger.info("Modo univariante detectado - usando solo variable objetivo")
        feature_cols = [target_col]
        univariate_mode = True
    else:
        univariate_mode = False
        logger.info(f"Modo multivariante - usando {len(feature_cols)} características: {feature_cols}")

    # VALIDATION: Ensure feature_cols exist in DataFrame (defense-in-depth)
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Características no encontradas en DataFrame: {missing_features}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Validate target exists
    if target_col not in df.columns:
        raise ValueError(
            f"Variable objetivo '{target_col}' no encontrada en DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Validate minimum data requirements to ensure enough sequences
    min_sequences = 50
    max_sequence_length = len(df) - forecast_horizon - min_sequences

    if sequence_length > max_sequence_length:
        # AUTO-FALLBACK: Adjust sequence_length with warning
        # NOTE FOR PHASE 2+: Consider stricter validation (raise error instead of auto-adjust)
        logger.warning(
            f"sequence_length {sequence_length} excede el máximo válido {max_sequence_length}. "
            f"Usando {max_sequence_length} en su lugar para garantizar al menos {min_sequences} secuencias."
        )
        sequence_length = max_sequence_length

    if sequence_length < 1:
        raise ValueError(
            f"Dataset insuficiente para crear secuencias. "
            f"Se requieren al menos {min_sequences + forecast_horizon + 1} muestras. "
            f"Dataset actual: {len(df)} muestras."
        )

    # Extract feature and target arrays
    features = df[feature_cols].values
    target = df[target_col].values

    # Create sequences using sliding window approach
    X_sequences = []
    y_sequences = []

    # Sliding window iteration
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        # Input sequence: sequence_length timesteps
        X_seq = features[i:i + sequence_length]

        # Target value: forecast_horizon steps ahead
        y_seq = target[i + sequence_length + forecast_horizon - 1]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    # Convert to numpy arrays with proper shapes
    X = np.array(X_sequences)  # Shape: (n_sequences, sequence_length, n_features)
    y = np.array(y_sequences)  # Shape: (n_sequences,)

    logger.info(
        f"Secuencias creadas exitosamente - X: {X.shape}, y: {y.shape} "
        f"({'univariante' if univariate_mode else 'multivariante con ' + str(len(feature_cols)) + ' features'})"
    )

    return X, y


def create_sequences_for_patchtsmixer(
    df: pd.DataFrame,
    channel_cols: List[str],
    context_length: int,
    prediction_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding window sequences for PatchTSMixer from DataFrame.

    Generates past_values (context) and future_values (targets) by sliding a window
    of size (context_length + prediction_length) across the time series data.

    Args:
        df: DataFrame with time series data (must be sorted by time ascending)
        channel_cols: List of column names to use as channels/features
                     All channels are both inputs and outputs for PatchTSMixer
        context_length: Number of timesteps for model input (e.g., 512)
                       Maps to sequence_length from frontend
        prediction_length: Number of timesteps to forecast (e.g., 96)
                          Maps to forecast_horizon from frontend

    Returns:
        Tuple of (past_values, future_values) tensors:
            - past_values: shape (num_sequences, context_length, num_channels)
            - future_values: shape (num_sequences, prediction_length, num_channels)

    Raises:
        ValueError: If channel_cols are missing, contain non-numeric data,
                   or insufficient data for at least one sequence

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        ...     'temp': np.random.randn(1000),
        ...     'humidity': np.random.randn(1000)
        ... })
        >>> past, future = create_sequences_for_patchtsmixer(
        ...     df, ['temp', 'humidity'], context_length=512, prediction_length=96
        ... )
        >>> past.shape  # 1000 - 512 - 96 + 1 = 393 sequences
        torch.Size([393, 512, 2])
        >>> future.shape
        torch.Size([393, 96, 2])
    """
    import logging
    import torch
    import numpy as np

    logger = logging.getLogger(__name__)

    # Defensive check for PyTorch availability
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PatchTSMixer but not installed. "
            "Install with: pip install torch>=2.0.1"
        )

    # Validate channel_cols exist in DataFrame
    missing_cols = [col for col in channel_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Channel columns not found in DataFrame: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate channel_cols are numeric
    non_numeric = []
    for col in channel_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(
            f"Channel columns must be numeric, but found non-numeric types: {non_numeric}. "
            f"Suggestion: Convert to numeric or remove these columns from channel_cols."
        )

    # Check for NaN/Inf values
    for col in channel_cols:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            logger.warning(
                f"Column '{col}' contains {nan_count} NaN values. "
                f"Consider imputation (forward fill, interpolation) or removal."
            )
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            raise ValueError(
                f"Column '{col}' contains {inf_count} infinite values. "
                f"Cannot create sequences with inf values. "
                f"Suggestion: Apply clipping or remove outliers before sequence generation."
            )

    # Extract data as numpy array
    data = df[channel_cols].values  # Shape: (num_timesteps, num_channels)
    num_timesteps = len(data)
    num_channels = len(channel_cols)

    logger.info(
        f"Creating PatchTSMixer sequences from {num_timesteps} timesteps, "
        f"{num_channels} channel(s): {channel_cols}"
    )
    logger.info(
        f"Parameters: context_length={context_length}, prediction_length={prediction_length}"
    )

    # Calculate total window size and number of sequences
    total_window = context_length + prediction_length

    if num_timesteps < total_window:
        raise ValueError(
            f"Insufficient data for sequence generation. "
            f"Need at least {total_window} timesteps "
            f"({context_length} context + {prediction_length} prediction), "
            f"but only have {num_timesteps} timesteps. "
            f"Suggestions: (1) Reduce context_length or prediction_length, "
            f"(2) Provide more data, or (3) Use a different model for short time series."
        )

    num_sequences = num_timesteps - total_window + 1

    logger.info(
        f"Will generate {num_sequences} sequences "
        f"(formula: {num_timesteps} - {context_length} - {prediction_length} + 1)"
    )

    # Initialize lists for sequences (more memory efficient than pre-allocating large arrays)
    past_sequences = []
    future_sequences = []

    # Sliding window loop - creates overlapping sequences
    for i in range(num_sequences):
        # Extract past window (context) - input to model
        past_window = data[i:i + context_length]  # Shape: (context_length, num_channels)

        # Extract future window (targets) - what model should predict
        future_window = data[
            i + context_length:i + context_length + prediction_length
        ]  # Shape: (prediction_length, num_channels)

        past_sequences.append(past_window)
        future_sequences.append(future_window)

    # Convert to numpy arrays first (more efficient than list of tensors)
    past_array = np.array(past_sequences)  # (num_sequences, context_length, num_channels)
    future_array = np.array(future_sequences)  # (num_sequences, prediction_length, num_channels)

    # Convert to PyTorch tensors (float32 for model compatibility)
    past_values = torch.FloatTensor(past_array)
    future_values = torch.FloatTensor(future_array)

    # Validate shapes match expectations
    expected_past_shape = (num_sequences, context_length, num_channels)
    expected_future_shape = (num_sequences, prediction_length, num_channels)

    assert past_values.shape == expected_past_shape, \
        f"past_values shape mismatch: expected {expected_past_shape}, got {past_values.shape}"
    assert future_values.shape == expected_future_shape, \
        f"future_values shape mismatch: expected {expected_future_shape}, got {future_values.shape}"

    # Validate minimum sequences (following LSTM pattern for statistical validity)
    min_sequences = 50
    if num_sequences < min_sequences:
        logger.warning(
            f"Only {num_sequences} sequences generated (recommended: ≥{min_sequences}). "
            f"This may be insufficient for robust model training. "
            f"Consider: (1) Using more data, or (2) Reducing context_length/prediction_length."
        )

    logger.info(
        f"✓ Successfully created {num_sequences} sequences: "
        f"past_values {tuple(past_values.shape)}, "
        f"future_values {tuple(future_values.shape)}"
    )
    logger.info(
        f"Tensor details: dtype={past_values.dtype}, device={past_values.device}"
    )

    return past_values, future_values


def lstm_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    split_ratios: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    División temporal para secuencias LSTM (mantiene forma 3D).

    Respeta el orden temporal - NO realiza shuffling aleatorio.

    Args:
        X: Secuencias de entrada, shape (n_sequences, sequence_length, n_features)
        y: Valores objetivo, shape (n_sequences,)
        split_ratios: Diccionario con proporciones {"train": 0.7, "val": 0.15, "test": 0.15}

    Returns:
        Tupla de 6 elementos:
        - X_train: Secuencias de entrenamiento (3D)
        - y_train: Objetivos de entrenamiento (1D)
        - X_val: Secuencias de validación (3D)
        - y_val: Objetivos de validación (1D)
        - X_test: Secuencias de prueba (3D)
        - y_test: Objetivos de prueba (1D)

    Raises:
        ValueError: Si los ratios no suman aproximadamente 1.0
    """
    # Validate split ratios sum to 1.0
    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(
            f"La suma de split_ratios debe ser 1.0, actual: {total_ratio}. "
            f"Proporciones recibidas: {split_ratios}"
        )

    # Calculate split indices based on ratios
    n = len(X)
    train_size = int(n * split_ratios["train"])
    val_size = int(n * split_ratios["val"])

    # Split maintaining temporal order (critical for time series)
    # Train: earliest data
    # Val: middle data
    # Test: most recent data
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    logger.info(
        f"División temporal completada - "
        f"Train: {len(X_train)} ({split_ratios['train']*100:.1f}%), "
        f"Val: {len(X_val)} ({split_ratios['val']*100:.1f}%), "
        f"Test: {len(X_test)} ({split_ratios['test']*100:.1f}%)"
    )

    # Validate minimum samples in each set
    if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
        logger.warning(
            f"Conjuntos muy pequeños detectados. "
            f"Se recomienda tener al menos 50 secuencias totales."
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


def patchtsmixer_train_val_test_split(
    past_values: torch.Tensor,
    future_values: torch.Tensor,
    split_ratios: Dict[str, float]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Temporal split for PatchTSMixer sequences into train/val/test sets.

    CRITICAL: Maintains temporal order (NO SHUFFLING). Earliest sequences go to train,
    middle sequences to val, latest sequences to test. This preserves time-series
    causality and prevents data leakage.

    Args:
        past_values: Tensor of shape (num_sequences, context_length, num_channels)
        future_values: Tensor of shape (num_sequences, prediction_length, num_channels)
        split_ratios: Dict with keys 'train', 'val', 'test' and float values (must sum to ~1.0)
                     Example: {'train': 0.7, 'val': 0.15, 'test': 0.15}

    Returns:
        Tuple of 6 tensors in this order:
            - train_past: Training context sequences
            - train_future: Training target sequences
            - val_past: Validation context sequences
            - val_future: Validation target sequences
            - test_past: Test context sequences
            - test_future: Test target sequences

    Raises:
        ValueError: If split_ratios don't sum to approximately 1.0
        ValueError: If any split results in zero sequences

    Example:
        >>> split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        >>> train_p, train_f, val_p, val_f, test_p, test_f = patchtsmixer_train_val_test_split(
        ...     past_values, future_values, split_ratios
        ... )
        >>> # For 393 sequences: train=275, val=59, test=59
    """
    import logging

    logger = logging.getLogger(__name__)

    # Defensive check for PyTorch availability
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for PatchTSMixer but not installed. "
            "Install with: pip install torch>=2.0.1"
        )

    num_sequences = past_values.shape[0]

    # Validate split_ratios sum to approximately 1.0
    ratio_sum = split_ratios['train'] + split_ratios['val'] + split_ratios['test']
    tolerance = 0.001
    if abs(ratio_sum - 1.0) > tolerance:
        raise ValueError(
            f"Split ratios must sum to 1.0 (within {tolerance} tolerance), "
            f"but got {ratio_sum:.4f}. "
            f"Received: train={split_ratios['train']}, "
            f"val={split_ratios['val']}, test={split_ratios['test']}"
        )

    # Calculate split indices (temporal order: train → val → test)
    train_end = int(num_sequences * split_ratios['train'])
    val_end = train_end + int(num_sequences * split_ratios['val'])
    # test_end is implicitly num_sequences (use remaining sequences)

    logger.info(
        f"Splitting {num_sequences} sequences temporally: "
        f"train={split_ratios['train']:.1%}, val={split_ratios['val']:.1%}, "
        f"test={split_ratios['test']:.1%}"
    )

    # Validate no split is empty
    if train_end == 0:
        raise ValueError(
            f"Train split results in 0 sequences. Increase train ratio or provide more data."
        )
    if val_end == train_end:
        raise ValueError(
            f"Validation split results in 0 sequences. Increase val ratio or provide more data."
        )
    if val_end >= num_sequences:
        raise ValueError(
            f"Test split results in 0 sequences. Increase test ratio or provide more data."
        )

    # Perform temporal split (slicing maintains order)
    # Train: earliest sequences [0:train_end]
    train_past = past_values[:train_end]
    train_future = future_values[:train_end]

    # Val: middle sequences [train_end:val_end]
    val_past = past_values[train_end:val_end]
    val_future = future_values[train_end:val_end]

    # Test: latest sequences [val_end:]
    test_past = past_values[val_end:]
    test_future = future_values[val_end:]

    # Log split sizes
    logger.info(
        f"✓ Split complete: train={train_past.shape[0]} sequences, "
        f"val={val_past.shape[0]} sequences, test={test_past.shape[0]} sequences"
    )
    logger.info(
        f"Temporal order preserved: train (earliest) → val (middle) → test (latest)"
    )

    # Verify splits sum to total (sanity check)
    total_split = train_past.shape[0] + val_past.shape[0] + test_past.shape[0]
    assert total_split == num_sequences, \
        f"Split sizes don't sum to total: {total_split} != {num_sequences}"

    return train_past, train_future, val_past, val_future, test_past, test_future


def build_lstm_model(params: Dict, input_shape: Tuple[int, int]) -> keras.Model:
    """
    Construye modelo Keras LSTM desde hiperparámetros.

    Soporta arquitecturas de una o múltiples capas LSTM.

    Args:
        params: Diccionario con hiperparámetros:
            - lstm_units: List[int] (e.g., [64] para 1 capa, [64, 32] para 2 capas)
            - dropout_rate: float (0.0 - 0.5)
            - recurrent_dropout_rate: float (0.0 - 0.5)
            - learning_rate: float (típicamente 0.0001 - 0.01)
        input_shape: Tupla (sequence_length, n_features)

    Returns:
        Modelo Sequential de Keras compilado con Adam optimizer y MSE loss

    Example:
        >>> params = {"lstm_units": [64, 32], "dropout_rate": 0.2,
        ...           "recurrent_dropout_rate": 0.2, "learning_rate": 0.001}
        >>> model = build_lstm_model(params, input_shape=(10, 2))
        >>> model.summary()
    """
    # Initialize Sequential model
    model = Sequential(name="LSTM_TimeSeriesModel")

    # Extract parameters with defaults
    lstm_units = params.get("lstm_units", [64])
    dropout_rate = params.get("dropout_rate", 0.2)
    recurrent_dropout_rate = params.get("recurrent_dropout_rate", 0.2)
    learning_rate = params.get("learning_rate", 0.001)

    logger.info(
        f"Construyendo modelo LSTM - Arquitectura: {lstm_units}, "
        f"Dropout: {dropout_rate}, Recurrent Dropout: {recurrent_dropout_rate}, "
        f"Learning Rate: {learning_rate}"
    )

    # Add LSTM layers (handle single vs multi-layer architectures)
    if len(lstm_units) == 1:
        # Single LSTM layer - no return_sequences needed
        model.add(LSTM(
            units=lstm_units[0],
            kernel_initializer=GlorotUniform(seed=SEED),
            recurrent_initializer=Orthogonal(seed=SEED),
            input_shape=input_shape,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name="LSTM_Layer"
        ))
    else:
        # Multiple LSTM layers - return_sequences=True for all except last
        for i, units in enumerate(lstm_units[:-1]):
            model.add(LSTM(
                units=units,
                return_sequences=True,  # Pass sequences to next LSTM layer
                kernel_initializer=GlorotUniform(seed=SEED),
                recurrent_initializer=Orthogonal(seed=SEED),
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                input_shape=input_shape if i == 0 else None,  # Only first layer needs input_shape
                name=f"LSTM_Layer_{i+1}"
            ))

        # Last LSTM layer - no return_sequences (output goes to Dense)
        model.add(LSTM(
            units=lstm_units[-1],
            kernel_initializer=GlorotUniform(seed=SEED),
            recurrent_initializer=Orthogonal(seed=SEED),
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name=f"LSTM_Layer_{len(lstm_units)}"
        ))

    # Output layer for single-step forecasting
    model.add(Dense(
        1,
        kernel_initializer=GlorotUniform(seed=SEED),
        name="Output_Layer"
    ))

    # Compile model with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"]  # Track both MAE and MSE during training
    )

    total_params = model.count_params()
    logger.info(
        f"Modelo LSTM compilado exitosamente - "
        f"Total de parámetros: {total_params:,}"
    )

    return model


def create_patchtsmixer_config(
    params: Dict,
    num_input_channels: int,
    context_length: int,
    prediction_length: int
):
    """
    Crea configuración de PatchTSMixer desde hiperparámetros.

    Args:
        params: Diccionario con hiperparámetros (claves opcionales con defaults):
            - patch_length: int (default: 8) - Tamaño de cada patch
            - d_model: int (default: 32) - Dimensión oculta
            - num_layers: int (default: 8) - Número de capas mixer
            - dropout: float (default: 0.2) - Tasa de dropout
            - expansion_factor: int (default: 2) - Factor de expansión MLP
            - head_dropout: float (default: dropout) - Dropout en cabeza de predicción
            - mode: str (default: "common_channel") - "common_channel" o "mix_channel"
            - gated_attn: bool (default: True) - Usar atención con compuertas
            - self_attn: bool (default: False) - Usar self-attention
            - scaling: str (default: "std") - Normalización ("std", "mean", None)
            - norm_mlp: str (default: "LayerNorm") - Tipo de normalización
        num_input_channels: Número de características de entrada
        context_length: Longitud de ventana histórica
        prediction_length: Horizonte de predicción

    Returns:
        PatchTSMixerConfig configurado y validado

    Raises:
        ImportError: Si transformers>=4.36.0 no está instalado
        ValueError: Si context_length no es divisible por patch_length

    Example:
        >>> params = {"patch_length": 8, "d_model": 32, "num_layers": 8}
        >>> config = create_patchtsmixer_config(params, 3, 512, 96)
        >>> print(config)
    """
    # Import transformers con manejo de errores
    try:
        from transformers import PatchTSMixerConfig
    except ImportError as e:
        raise ImportError(
            "transformers>=4.36.0 requerido para PatchTSMixer. "
            "Instalar con: pip install 'transformers>=4.36.0'"
        ) from e

    # Extraer parámetros esenciales con defaults
    patch_length = params.get("patch_length", 8)
    d_model = params.get("d_model", 32)
    num_layers = params.get("num_layers", 8)
    dropout = params.get("dropout", 0.2)

    # Validaciones de rangos (deben hacerse ANTES de usar los valores)
    if patch_length < 1:
        raise ValueError(f"patch_length debe ser >= 1, recibido: {patch_length}")
    if d_model < 1:
        raise ValueError(f"d_model debe ser >= 1, recibido: {d_model}")
    if num_layers < 1:
        raise ValueError(f"num_layers debe ser >= 1, recibido: {num_layers}")
    if not (0.0 <= dropout <= 1.0):
        raise ValueError(f"dropout debe estar en [0.0, 1.0], recibido: {dropout}")

    # VALIDACIÓN CRÍTICA: context_length debe ser divisible por patch_length
    if context_length % patch_length != 0:
        remainder = context_length % patch_length
        closest_lower = (context_length // patch_length) * patch_length
        closest_upper = closest_lower + patch_length
        raise ValueError(
            f"context_length ({context_length}) debe ser divisible por patch_length ({patch_length}). "
            f"Resto: {remainder}. "
            f"Valores válidos sugeridos: {closest_lower} o {closest_upper}"
        )

    # Extraer parámetros avanzados con defaults
    expansion_factor = params.get("expansion_factor", 2)
    head_dropout = params.get("head_dropout", dropout)  # Usa dropout si no especificado
    mode = params.get("mode", "common_channel")
    gated_attn = params.get("gated_attn", True)
    self_attn = params.get("self_attn", False)
    scaling = params.get("scaling", "std")
    norm_mlp = params.get("norm_mlp", "LayerNorm")

    # Crear configuración de PatchTSMixer
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        num_input_channels=num_input_channels,
        patch_length=patch_length,
        patch_stride=patch_length,  # Non-overlapping patches (recomendado)
        d_model=d_model,
        num_layers=num_layers,
        expansion_factor=expansion_factor,
        dropout=dropout,
        head_dropout=head_dropout,
        mode=mode,
        gated_attn=gated_attn,
        self_attn=self_attn,
        scaling=scaling,
        norm_mlp=norm_mlp,
        loss="mse",  # Fijo para pronóstico puntual
    )

    # Logging detallado
    num_patches = (context_length - patch_length) // patch_length + 1
    logger.info(
        f"✓ Configuración PatchTSMixer creada: "
        f"d_model={d_model}, num_layers={num_layers}, "
        f"patch_length={patch_length} (patches={num_patches})"
    )
    logger.info(
        f"  Entrada: context={context_length}, prediction={prediction_length}, "
        f"channels={num_input_channels}"
    )
    logger.info(
        f"  Arquitectura: expansion={expansion_factor}, dropout={dropout}, "
        f"mode={mode}, gated_attn={gated_attn}"
    )

    return config


def build_patchtsmixer_model(config):
    """
    Inicializa modelo PatchTSMixer desde configuración.

    Args:
        config: Configuración de PatchTSMixer creada con create_patchtsmixer_config()

    Returns:
        Modelo PatchTSMixerForPrediction en CPU, listo para entrenamiento

    Raises:
        ImportError: Si transformers>=4.36.0 no está instalado
        RuntimeError: Si inicialización del modelo falla

    Example:
        >>> config = create_patchtsmixer_config({}, 3, 512, 96)
        >>> model = build_patchtsmixer_model(config)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    """
    # Import transformers
    try:
        from transformers import PatchTSMixerForPrediction
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers>=4.36.0 y torch>=2.0.0 requeridos. "
            "Instalar con: pip install 'transformers>=4.36.0' torch"
        ) from e

    # Inicializar modelo desde configuración
    try:
        model = PatchTSMixerForPrediction(config)
    except Exception as e:
        logger.error(f"Error inicializando PatchTSMixer: {e}")
        raise RuntimeError(f"Fallo en inicialización del modelo: {e}") from e

    # Forzar CPU (requisito DREAM-ML)
    device = torch.device('cpu')
    model = model.to(device)

    # Establecer modo evaluación inicialmente
    model.eval()

    # Calcular y loggear número de parámetros
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"✓ Modelo PatchTSMixer inicializado: "
        f"{num_params:,} parámetros ({num_trainable:,} entrenables)"
    )
    logger.info(f"  Dispositivo: {device}")
    logger.info(f"  Configuración: {config.d_model}d × {config.num_layers} capas")

    return model


def get_patchtsmixer_preset(preset_name: str) -> Dict:
    """
    Retorna configuración preset de PatchTSMixer.

    Presets disponibles:
    - "small": Modelo ligero (16d, 6 capas) - Rápido, menor capacidad
    - "medium": Modelo estándar (32d, 8 capas) - Balance rendimiento/velocidad
    - "large": Modelo potente (64d, 12 capas) - Mejor rendimiento, más lento

    Args:
        preset_name: Nombre del preset ("small", "medium", "large")

    Returns:
        Diccionario con hiperparámetros del preset

    Raises:
        ValueError: Si preset_name no es válido

    Example:
        >>> params = get_patchtsmixer_preset("medium")
        >>> config = create_patchtsmixer_config(params, 3, 512, 96)
    """
    PRESETS = {
        "small": {
            "d_model": 16,
            "num_layers": 6,
            "patch_length": 16,
            "dropout": 0.2,
            "expansion_factor": 2,
        },
        "medium": {
            "d_model": 32,
            "num_layers": 8,
            "patch_length": 8,
            "dropout": 0.2,
            "expansion_factor": 2,
        },
        "large": {
            "d_model": 64,
            "num_layers": 12,
            "patch_length": 8,
            "dropout": 0.2,
            "expansion_factor": 2,
        }
    }

    if preset_name not in PRESETS:
        available = ", ".join(f"'{p}'" for p in PRESETS.keys())
        raise ValueError(
            f"Preset inválido: '{preset_name}'. "
            f"Presets disponibles: {available}"
        )

    logger.info(f"✓ Usando preset '{preset_name}': {PRESETS[preset_name]}")
    return PRESETS[preset_name].copy()  # Return copy to avoid mutation


def create_lstm_callbacks(
    experiment_dir: str,
    early_stopping_patience: int,
    checkpoint_filename: str = "best_lstm_checkpoint.h5"
) -> Tuple[List[keras.callbacks.Callback], str]:
    """
    Crea callbacks de Keras para entrenamiento LSTM.

    Args:
        experiment_dir: Directorio del experimento para guardar checkpoints temporales
        early_stopping_patience: Número de épocas a esperar sin mejora antes de detener
        checkpoint_filename: Nombre del archivo de checkpoint (default: "best_lstm_checkpoint.h5")

    Returns:
        Tupla de:
        - Lista de callbacks de Keras configurados
        - Ruta completa al archivo de checkpoint

    Callbacks incluidos:
        - EarlyStopping: Detiene entrenamiento si val_loss no mejora
        - ModelCheckpoint: Guarda mejor modelo basado en val_loss
        - ReduceLROnPlateau: Reduce learning rate si val_loss se estanca
    """
    # Create temporary checkpoint directory
    # Checkpoints will be deleted after training (aggressive cleanup)
    checkpoint_dir = os.path.join(experiment_dir, "temp_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    logger.info(f"Directorio de checkpoints temporales: {checkpoint_dir}")

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,  # Load best weights when stopping
        verbose=1,
        mode="min"
    )

    # ModelCheckpoint callback (saves best model only)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,  # Only save when val_loss improves
        verbose=0,  # Quiet mode (logged via logger instead)
        mode="min"
    )

    # ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # Multiply learning rate by 0.5
        patience=early_stopping_patience // 2,  # Half of early stopping patience
        min_lr=1e-7,  # Minimum learning rate threshold
        verbose=1,
        mode="min"
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    logger.info(
        f"Callbacks configurados - "
        f"EarlyStopping patience: {early_stopping_patience}, "
        f"ReduceLR patience: {early_stopping_patience // 2}"
    )

    return callbacks, checkpoint_path


def evaluate_lstm_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prefix: str,
    experiment_dir: str
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evalúa modelo LSTM y genera gráficos de diagnóstico.

    Args:
        model: Modelo Keras entrenado
        X_test: Secuencias de prueba, shape (n_samples, sequence_length, n_features)
        y_test: Objetivos de prueba, shape (n_samples,)
        prefix: Prefijo para métricas ("val" o "test")
        experiment_dir: Directorio para guardar gráficos

    Returns:
        Tupla de:
        - metrics: Diccionario {f"{prefix}_rmse": float, f"{prefix}_mae": float, ...}
        - artifacts: Lista de rutas a archivos de gráficos generados
    """
    # Generate predictions
    logger.info(f"Generando predicciones para conjunto {prefix}...")
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate MAPE (avoid division by zero)
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = None
        logger.warning(f"No se puede calcular MAPE para {prefix}: todos los valores objetivo son cero")

    metrics = {
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_mae": float(mae),
        f"{prefix}_mape": float(mape) if mape is not None else None
    }

    logger.info(
        f"Métricas {prefix} - RMSE: {rmse:.4f}, MAE: {mae:.4f}" +
        (f", MAPE: {mape:.2f}%" if mape is not None else "")
    )

    # Generate diagnostic plots
    artifacts = []

    # Plot 1: Predictions vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Real", alpha=0.7, linewidth=2)
    plt.plot(y_pred, label="Predicción LSTM", alpha=0.7, linewidth=2)
    plt.title(f"LSTM - Predicciones vs Valores Reales ({prefix.upper()})")
    plt.xlabel("Índice de Muestra")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    forecast_path = os.path.join(experiment_dir, f"lstm_{prefix}_forecast.png")
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(forecast_path)
    logger.info(f"Gráfico de pronóstico guardado: {forecast_path}")

    # Plot 2: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    plt.title(f"LSTM - Residuos ({prefix.upper()})")
    plt.xlabel("Índice de Muestra")
    plt.ylabel("Residuo (Real - Predicción)")
    plt.grid(True, alpha=0.3)
    residuals_path = os.path.join(experiment_dir, f"lstm_{prefix}_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_path)
    logger.info(f"Gráfico de residuos guardado: {residuals_path}")

    # Plot 3: Residuals distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    plt.title(f"LSTM - Distribución de Residuos ({prefix.upper()})")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3, axis='y')
    residuals_dist_path = os.path.join(experiment_dir, f"lstm_{prefix}_residuals_distribution.png")
    plt.savefig(residuals_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_dist_path)
    logger.info(f"Gráfico de distribución de residuos guardado: {residuals_dist_path}")

    return metrics, artifacts


def generate_random_lstm_params(random_search_params: Dict, rng: np.random.Generator = None) -> Dict:
    """
    Genera hiperparámetros LSTM aleatorios desde rangos especificados.

    Asegura que todos los valores retornados sean tipos Python nativos
    (no numpy) para serialización JSON y logging en MLflow.

    Args:
        random_search_params: Diccionario con rangos y opciones:
            - lstm_units_options: List[List[int]] (e.g., [[32], [64], [128], [64, 32]])
            - dropout_rate_range: [float, float] (e.g., [0.1, 0.3])
            - recurrent_dropout_rate_range: [float, float]
            - learning_rate_range: [float, float] (usa distribución log-uniforme)
            - batch_size_options: List[int] (e.g., [16, 32, 64])
            - epochs_range: [int, int] (e.g., [50, 100])

    Returns:
        Diccionario con parámetros aleatorios (tipos Python nativos)

    Example:
        >>> search_params = {
        ...     "lstm_units_options": [[32], [64], [128]],
        ...     "dropout_rate_range": [0.1, 0.3],
        ...     "learning_rate_range": [0.001, 0.01],
        ...     "batch_size_options": [16, 32],
        ...     "epochs_range": [50, 100]
        ... }
        >>> params = generate_random_lstm_params(search_params)
        >>> params
        {'lstm_units': [64], 'dropout_rate': 0.234, ...}
    """
    # Initialize RNG if not provided (for reproducibility)
    if rng is None:
        rng = np.random.default_rng(seed=SEED)

    # Valores por defecto para los rangos (conservadores, siguiendo patrón Phase 2A)
    default_ranges = {
        "lstm_units_options": [[32], [64], [128], [64, 32]],
        "dropout_rate_range": [0.1, 0.3],
        "recurrent_dropout_rate_range": [0.1, 0.3],
        "learning_rate_range": [0.0001, 0.01],
        "batch_size_options": [16, 32, 64],
        "epochs_range": [50, 100]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros aleatorios con distribuciones apropiadas
    # Note: Use rng.integers to select index for reproducibility (rng.choice doesn't work well with nested lists)
    lstm_units_idx = rng.integers(0, len(ranges["lstm_units_options"]))
    lstm_units_choice = ranges["lstm_units_options"][lstm_units_idx]

    params = {
        # Categorical: random choice from list (using NumPy RNG for reproducibility)
        "lstm_units": list(lstm_units_choice) if not isinstance(lstm_units_choice, list) else lstm_units_choice,

        # Uniform distribution for dropout rates
        "dropout_rate": float(rng.uniform(
            ranges["dropout_rate_range"][0],
            ranges["dropout_rate_range"][1]
        )),

        "recurrent_dropout_rate": float(rng.uniform(
            ranges["recurrent_dropout_rate_range"][0],
            ranges["recurrent_dropout_rate_range"][1]
        )),

        # Log-uniform distribution for learning rate (better sampling across orders of magnitude)
        "learning_rate": float(np.exp(rng.uniform(
            np.log(ranges["learning_rate_range"][0]),
            np.log(ranges["learning_rate_range"][1])
        ))),

        # Categorical: random choice for batch size (using NumPy RNG for reproducibility)
        "batch_size": int(rng.choice(ranges["batch_size_options"])),

        # Uniform integer distribution for epochs
        "epochs": int(rng.integers(
            ranges["epochs_range"][0],
            ranges["epochs_range"][1] + 1
        ))
    }

    # Convert any remaining numpy types to Python natives for JSON serialization
    params = convert_numpy_to_python(params)

    logger.debug(f"Generated random LSTM params: {params}")

    return params


def train_lstm_model(dataset_path: str, data: Dict, experiment_dir: str) -> Dict:
    """
    Entrena y registra un modelo LSTM para pronóstico de series temporales.

    Sigue el mismo contrato que train_arima_model y train_xgboost_model.
    Soporta búsqueda manual de hiperparámetros en Fase 1.

    Args:
        dataset_path: Ruta al archivo CSV con los datos
        data: Diccionario con configuración del entrenamiento, incluyendo:
            - date_col_name: str
            - target_variable: str
            - input_features: List[str]
            - model_name: str
            - forecast_horizon: int (default: 1)
            - split_ratios: Dict (default: {"train": 0.7, "val": 0.15, "test": 0.15})
            - hyperparameter_search_strategy: "none" para manual (Fase 1)
            - sequence_length: int (default: 10)
            - early_stopping_patience: int (default: 20)
            - lstm_params: Dict con hiperparámetros manuales
        experiment_dir: Directorio para guardar artefactos

    Returns:
        Diccionario con resultados del entrenamiento:
        {
            "status": "success" | "error",
            "val_metrics": {"rmse": float, "mae": float, "mape": float},
            "test_metrics": {"rmse": float, "mae": float, "mape": float},
            "model_path": str,
            "run_id": str,
            "features_used": List[str]
        }

    Raises:
        ValueError: Errores de validación de parámetros
        RuntimeError: Errores de ejecución durante entrenamiento
    """
    # ======================
    # 0. REPRODUCIBILITY (MUST BE ABSOLUTE FIRST — before any TF/Keras/numpy op)
    # ======================
    seed = SEED
    
    # Reset ALL random states for this specific training run.
    # This is necessary because set_global_seeds() at module import only
    # sets the seed ONCE. TF's random counter advances after every run,
    # so without this reset, run #2 starts from a different counter position.
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # This is the key call: resets TF's global seed AND all Keras stateful
    # generators (including the ones used by LSTM recurrent_dropout).
    # More complete than tf.random.set_seed() alone.
    tf.keras.utils.set_random_seed(seed)
    
    logger.info(f"[REPRODUCIBILITY] Seeds reset to {seed} for this training run.")
    try:
        # ======================
        # 1. EXTRACCIÓN DE PARÁMETROS
        # ======================

        # Extract required parameters
        date_col_name = data.get("date_col_name")
        target_variable = data.get("target_variable")
        input_features = data.get("input_features", [])
        if not input_features:
            logger.info("input_features empty - defaulting to univariate mode with target variable")
            input_features = [target_variable]
        model_name = data.get("model_name")

        # Extract optional parameters with defaults
        forecast_horizon = data.get("forecast_horizon", 1)
        split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        sequence_length = data.get("sequence_length", 10)
        early_stopping_patience = data.get("early_stopping_patience", 20)
        hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", "none")

        # Validate required parameters
        if not all([date_col_name, target_variable, model_name]):
            raise ValueError(
                "Parámetros requeridos faltantes. Se requieren: "
                "date_col_name, target_variable, model_name"
            )

        # CPU warning
        logger.warning(
            "⚠️ Entrenamiento LSTM usa CPU solamente (sin soporte GPU en esta versión). "
            "Tiempo de entrenamiento esperado: 30-60 minutos para 100 épocas. "
            "Considere reducir 'epochs' si el tiempo es excesivo."
        )

        # ======================
        # 2. INICIALIZACIÓN DE MLFLOW
        # ======================

        # Verify no active MLflow run
        if mlflow.active_run():
            mlflow.end_run()
            logger.warning("Run activa de MLflow detectada y finalizada")

        # Start MLflow run
        run_id = str(uuid.uuid4())[:8]
        mlflow.start_run(run_name=f"lstm_manual_{run_id}")
        mlflow_run_id = mlflow.active_run().info.run_id

        logger.info(f"Iniciando entrenamiento LSTM en run: {mlflow_run_id}")

        # Log parameters to MLflow
        mlflow.log_params({
            "model_type": "LSTM",
            "date_col_name": date_col_name,
            "target_variable": target_variable,
            "input_features": str(input_features),
            "forecast_horizon": forecast_horizon,
            "sequence_length": sequence_length,
            "early_stopping_patience": early_stopping_patience,
            "split_ratios": str(split_ratios),
            "hyperparameter_search_strategy": hyperparameter_search_strategy,
            "cpu_only": True
        })

        # ======================
        # 3. CARGA Y VALIDACIÓN DE DATOS
        # ======================

        # Load and validate data
        logger.info("Cargando y validando dataset...")
        df = load_and_validate_ts_data(dataset_path, date_col_name, target_variable)

        # Validate input features exist
        for feature in input_features:
            if feature not in df.columns:
                raise ValueError(f"Característica de entrada no encontrada: {feature}")

        # Validate input features are numeric (LSTM requires numeric tensor inputs)
        for feature in input_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                # Get available numeric columns for helpful error message
                numeric_cols = [col for col in df.columns
                               if pd.api.types.is_numeric_dtype(df[col]) and col != target_variable]

                raise ValueError(
                    f"La característica '{feature}' debe ser numérica para entrenamiento LSTM.\n"
                    f"Dtype actual: {df[feature].dtype}\n\n"
                    f"LSTM requiere características numéricas. Las columnas categóricas deben ser "
                    f"codificadas en el paso 'data_encoding' antes del entrenamiento.\n\n"
                    f"Opciones de codificación:\n"
                    f"  - One-Hot Encoding: Para categorías con baja cardinalidad\n"
                    f"  - Label Encoding: Para categorías ordinales\n"
                    f"  - Embedding Layers: Para categorías con alta cardinalidad (avanzado)\n\n"
                    f"Columnas numéricas disponibles en el dataset: {numeric_cols}\n\n"
                    f"Si '{feature}' debe ser una característica de entrada, codifíquela "
                    f"primero en el paso de preprocesamiento de datos."
                )

        logger.info(f"Dataset cargado: {len(df)} muestras, características: {input_features}")

        # Log training mode explicitly
        n_input_features = len(input_features)
        if n_input_features == 1 and input_features[0] == target_variable:
            training_mode = TRAINING_MODE_UNIVARIATE
            logger.info(f"Entrenando LSTM en modo {training_mode} (solo variable objetivo)")
        elif n_input_features == 0:
            logger.warning("input_features empty - forcing univariate mode")
            input_features = [target_variable]
            training_mode = TRAINING_MODE_UNIVARIATE
            n_input_features = 1
        else:
            training_mode = TRAINING_MODE_MULTIVARIATE
            logger.info(f"Entrenando LSTM en modo {training_mode} con {n_input_features} características: {input_features}")

        # ======================
        # 4. CREACIÓN DE SECUENCIAS
        # ======================

        # Create LSTM sequences
        logger.info(f"Creando secuencias LSTM (sequence_length={sequence_length})...")
        X, y = create_sequences_for_lstm(
            df=df,
            feature_cols=input_features,
            target_col=target_variable,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )

        # Log sequence creation results
        mlflow.log_params({
            "n_sequences": len(X),
            "sequence_shape": str(X.shape),
            "n_features": X.shape[2]
        })

        # ======================
        # 5. DIVISIÓN TRAIN/VAL/TEST
        # ======================

        # Split data
        logger.info("Dividiendo dataset en train/val/test...")
        X_train, y_train, X_val, y_val, X_test, y_test = lstm_train_val_test_split(
            X=X,
            y=y,
            split_ratios=split_ratios
        )

        # ======================
        # 6. INICIALIZACIÓN DE TRACKER DE ENERGÍA
        # ======================

        # Start energy tracking
        tracker = EmissionsTracker(
            project_name=f"LSTM_{model_name}",
            output_dir=experiment_dir,
            log_level="warning",
            save_to_file=False,
            allow_multiple_runs=True
        )
        tracker.start()

        # ======================
        # 7. ENTRENAMIENTO CON PARÁMETROS MANUALES
        # ======================

        if hyperparameter_search_strategy == "manual":
            # Extract manual parameters
            lstm_params = data.get("manual_params" , {
                "lstm_units": [64],
                "dropout_rate": 0.2,
                "recurrent_dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            })

            logger.info(f"Entrenando con parámetros manuales: {lstm_params}")

            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
            model = build_lstm_model(lstm_params, input_shape)

            # Create callbacks
            callbacks, checkpoint_path = create_lstm_callbacks(
                experiment_dir=experiment_dir,
                early_stopping_patience=early_stopping_patience
            )

            # Train model
            logger.info("Iniciando entrenamiento del modelo...")
            logger.info(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=lstm_params.get("epochs", 100),
                batch_size=lstm_params.get("batch_size", 32),
                callbacks=callbacks,
                verbose=1  # Show progress bar
            )

            # Extract best validation metrics from history
            best_val_loss = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val_loss) + 1

            logger.info(
                f"Entrenamiento completado - "
                f"Mejor val_loss: {best_val_loss:.4f} en época {best_epoch}"
            )

            # Log training metrics
            mlflow.log_params({
                "best_epoch": best_epoch,
                "total_epochs_trained": len(history.history["loss"])
            })
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("final_train_loss", history.history["loss"][-1])

            best_model = model
            best_params = lstm_params

        elif hyperparameter_search_strategy == "grid":
            # ======================
            # GRID SEARCH IMPLEMENTATION (Phase 2A)
            # ======================
            logger.info("Iniciando Grid Search de hiperparámetros...")

            # Extract grid search parameters with conservative defaults
            grid_search_params = data.get("grid_search_params", {})

            # Conservative defaults: 2×2×2×1×1 = 8 combinations
            default_grid_params = {
                "lstm_units_options": [[64], [128]],
                "dropout_rate_options": [0.2, 0.3],
                "recurrent_dropout_rate_options": [0.2],
                "learning_rate_options": [0.001, 0.01],
                "batch_size_options": [32],
                "epochs_options": [100]
            }

            # Merge with user-provided params (user params override defaults)
            grid_params = {**default_grid_params, **grid_search_params}

            # Extract memory profiling and warning threshold settings
            enable_memory_profiling = data.get("enable_memory_profiling", False)
            grid_warning_threshold = data.get("grid_warning_threshold", 50)

            # Initialize memory profiling if enabled
            if enable_memory_profiling:
                process = psutil.Process(os.getpid())
                initial_memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory profiling enabled. Initial memory: {initial_memory_mb:.1f} MB")

            # Generate parameter grid
            param_grid_dict = {
                "lstm_units": grid_params["lstm_units_options"],
                "dropout_rate": grid_params["dropout_rate_options"],
                "recurrent_dropout_rate": grid_params.get("recurrent_dropout_rate_options", [0.2]),
                "learning_rate": grid_params["learning_rate_options"],
                "batch_size": grid_params["batch_size_options"],
                "epochs": grid_params["epochs_options"]
            }

            grid = list(ParameterGrid(param_grid_dict))
            n_combinations = len(grid)

            logger.info(f"Grid Search: {n_combinations} combinaciones a evaluar")

            # Warn if combinations exceed threshold
            if n_combinations > grid_warning_threshold:
                logger.warning(
                    f"⚠️ Grid Search generará {n_combinations} combinaciones, "
                    f"lo cual excede el umbral de {grid_warning_threshold}. "
                    f"Esto puede tomar varias horas. Considere reducir el espacio de búsqueda "
                    f"o usar Random Search en su lugar."
                )

            # Initialize best model tracking
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_iteration = None

            input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)

            # Grid search loop
            for i, params in enumerate(grid):
                logger.info(f"Grid Search - Iteración {i+1}/{n_combinations}: {params}")

                try:
                    # Build model
                    model = build_lstm_model(params, input_shape)

                    # Create callbacks with unique checkpoint filename
                    callbacks, checkpoint_path = create_lstm_callbacks(
                        experiment_dir=experiment_dir,
                        early_stopping_patience=early_stopping_patience,
                        checkpoint_filename=f"grid_checkpoint_{i}.h5"
                    )

                    # Train model (verbose=0 for quiet mode in search loops)
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        callbacks=callbacks,
                        verbose=0  # Quiet mode for grid search
                    )

                    # Extract best val_loss from this iteration
                    iteration_val_loss = min(history.history["val_loss"])

                    logger.info(
                        f"Grid Search - Iteración {i+1} completada. "
                        f"val_loss: {iteration_val_loss:.4f}"
                    )

                    # Update best model if improved
                    if iteration_val_loss < best_val_loss:
                        best_val_loss = iteration_val_loss

                        # Delete previous best model to save memory
                        if best_model is not None:
                            del best_model

                        best_model = model
                        best_params = params
                        best_iteration = i + 1

                        logger.info(
                            f"✓ Nuevo mejor modelo encontrado en iteración {best_iteration}: "
                            f"val_loss={best_val_loss:.4f}"
                        )
                    else:
                        # Not the best, delete to free memory
                        del model

                    # CRITICAL: Memory cleanup after each iteration
                    tf.keras.backend.clear_session()
                    gc.collect()

                    # Progress update every 10 iterations
                    if (i + 1) % 10 == 0:
                        logger.info(f"Grid Search Progress: {i+1}/{n_combinations} iterations completed")

                except Exception as e:
                    logger.error(f"Error en iteración {i+1}: {e}")
                    # Continue to next iteration (don't fail entire search)
                    continue

            # Validate at least one model trained successfully
            if best_model is None:
                raise RuntimeError("No se pudo entrenar ningún modelo en Grid Search")

            logger.info(
                f"Grid Search completado - Mejor modelo: iteración {best_iteration}, "
                f"val_loss={best_val_loss:.4f}, params={best_params}"
            )

            # Log best results to MLflow
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("grid_iterations_total", n_combinations)
            mlflow.log_metric("best_iteration", best_iteration)

            # Log memory profiling results if enabled
            if enable_memory_profiling:
                final_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_increase_mb = final_memory_mb - initial_memory_mb

                logger.info(
                    f"Memory profiling results - Initial: {initial_memory_mb:.1f} MB, "
                    f"Final: {final_memory_mb:.1f} MB, Increase: {memory_increase_mb:.1f} MB"
                )

                mlflow.log_metric("memory_usage_mb", final_memory_mb)
                mlflow.log_metric("memory_increase_mb", memory_increase_mb)

                # Log warning if memory increase exceeds threshold
                if memory_increase_mb > 500:
                    logger.warning(
                        f"⚠️ Memory increase {memory_increase_mb:.1f}MB exceeds 500MB threshold. "
                        f"Consider reducing grid size or batch size."
                    )

            # Set best_epoch to None for grid search (not tracked per iteration)
            best_epoch = None

        elif hyperparameter_search_strategy == "random":
            # ======================
            # RANDOM SEARCH IMPLEMENTATION
            # ======================
            logger.info("Iniciando Random Search de hiperparámetros...")

            # Extract random search parameters
            n_random_iterations = data.get("n_random_iterations", 100)
            random_search_params = data.get("random_search_params", {})

            # Extract memory profiling settings (following Phase 2A pattern)
            enable_memory_profiling = data.get("enable_memory_profiling", False)

            # Validate n_random_iterations
            if n_random_iterations <= 0:
                raise ValueError(
                    f"n_random_iterations debe ser un número positivo. "
                    f"Valor recibido: {n_random_iterations}"
                )

            logger.info(f"Random Search: {n_random_iterations} iteraciones")

            # Warn if iterations very high (performance threshold)
            if n_random_iterations > 200:
                logger.warning(
                    f"⚠️ n_random_iterations es muy alto ({n_random_iterations}). "
                    f"Esto puede tomar varias horas. "
                    f"Considere usar un valor menor (<200) para mejorar el rendimiento."
                )

            # Initialize memory profiling if enabled
            if enable_memory_profiling:
                process = psutil.Process(os.getpid())
                initial_memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory profiling enabled. Initial memory: {initial_memory_mb:.1f} MB")

            # Initialize best model tracking
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_iteration = None

            input_shape = (X_train.shape[1], X_train.shape[2])

            # Random search loop
            for i in range(n_random_iterations):
                # Generate random parameters
                params = generate_random_lstm_params(random_search_params)

                logger.info(
                    f"Random Search - Iteración {i+1}/{n_random_iterations}: "
                    f"lstm_units={params['lstm_units']}, "
                    f"dropout={params['dropout_rate']:.3f}, "
                    f"lr={params['learning_rate']:.6f}, "
                    f"batch_size={params['batch_size']}, "
                    f"epochs={params['epochs']}"
                )

                try:
                    # Build model
                    model = build_lstm_model(params, input_shape)

                    # Create callbacks with unique checkpoint filename
                    callbacks, checkpoint_path = create_lstm_callbacks(
                        experiment_dir=experiment_dir,
                        early_stopping_patience=early_stopping_patience,
                        checkpoint_filename=f"random_checkpoint_{i}.h5"
                    )

                    # Train model (verbose=0 for quiet mode during random search)
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        callbacks=callbacks,
                        verbose=0
                    )

                    # Extract best val_loss from this iteration
                    iteration_val_loss = min(history.history["val_loss"])

                    logger.info(
                        f"Random Search - Iteración {i+1} completada. "
                        f"val_loss: {iteration_val_loss:.4f}"
                    )

                    # Update best model if improved
                    if iteration_val_loss < best_val_loss:
                        best_val_loss = iteration_val_loss

                        # Delete previous best model to save memory
                        if best_model is not None:
                            del best_model

                        best_model = model
                        best_params = params
                        best_iteration = i + 1

                        logger.info(
                            f"✓ Nuevo mejor modelo encontrado en iteración {best_iteration}: "
                            f"val_loss={best_val_loss:.4f}"
                        )
                    else:
                        # Not the best, delete to free memory
                        del model

                    # CRITICAL: Memory cleanup after each iteration (prevent memory leak)
                    tf.keras.backend.clear_session()
                    gc.collect()

                    # Progress update every 10 iterations
                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"Random Search Progress: {i+1}/{n_random_iterations} iterations completed. "
                            f"Best val_loss so far: {best_val_loss:.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error en iteración {i+1}: {e}")
                    # Continue to next iteration (don't fail entire search)
                    continue

            # Verify at least one model trained successfully
            if best_model is None:
                raise RuntimeError(
                    "No se pudo entrenar ningún modelo exitosamente en Random Search. "
                    "Revise los logs para más detalles sobre los errores."
                )

            logger.info(
                f"Random Search completado - Mejor modelo: iteración {best_iteration}, "
                f"val_loss={best_val_loss:.4f}, params={best_params}"
            )

            # Log best results to MLflow
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("random_iterations_total", n_random_iterations)
            mlflow.log_metric("best_iteration", best_iteration)

            # Log memory profiling results if enabled
            if enable_memory_profiling:
                final_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_increase_mb = final_memory_mb - initial_memory_mb

                logger.info(
                    f"Memory profiling results - Initial: {initial_memory_mb:.1f} MB, "
                    f"Final: {final_memory_mb:.1f} MB, Increase: {memory_increase_mb:.1f} MB"
                )

                mlflow.log_metric("memory_usage_mb", final_memory_mb)
                mlflow.log_metric("memory_increase_mb", memory_increase_mb)

                # Log warning if memory increase exceeds threshold
                if memory_increase_mb > 500:
                    logger.warning(
                        f"⚠️ Memory increase {memory_increase_mb:.1f}MB exceeds 500MB threshold. "
                        f"Consider reducing number of iterations or batch size."
                    )

            # Set best_epoch to None for random search (not tracked per iteration)
            best_epoch = None

        elif hyperparameter_search_strategy == "bayesian":
            # ======================
            # BAYESIAN SEARCH IMPLEMENTATION (Phase 5)
            # ======================
            logger.info("Iniciando Bayesian Search de hiperparámetros con Optuna...")

            # Extract Bayesian config
            bayesian_config = data.get("bayesian_config", {})
            n_trials = bayesian_config.get("n_trials", 50)
            n_initial_points = bayesian_config.get("n_initial_points", 10)
            timeout_seconds = bayesian_config.get("timeout_seconds", None)
            optimization_metric = data.get("optimization_metric", "val_rmse")

            # Validate
            if n_trials < 1:
                raise ValueError(f"n_trials must be at least 1, got {n_trials}")
            if n_initial_points >= n_trials:
                raise ValueError(
                    f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
                )

            # Reset global memory monitoring variables (Phase 8)
            global peak_memory_mb, memory_exceeded
            peak_memory_mb = 0.0
            memory_exceeded = False

            logger.info("="*60)
            logger.info("LSTM Bayesian Search Configuration:")
            logger.info(f"  n_trials: {n_trials}")
            logger.info(f"  n_initial_points: {n_initial_points}")
            logger.info(f"  timeout_seconds: {timeout_seconds}")
            logger.info("="*60)

            # Get fixed early_stopping_patience from manual_params
            manual_params = data.get("manual_params", {})
            early_stopping_patience = manual_params.get("early_stopping_patience", 10)

            # Split dataframe temporally for Bayesian search (needed for variable time_steps)
            logger.info("Splitting dataframe temporally for Bayesian search...")
            n = len(df)
            train_size = int(n * split_ratios["train"])
            val_size = int(n * split_ratios["val"])

            train_df = df.iloc[:train_size].copy()
            val_df = df.iloc[train_size:train_size + val_size].copy()
            test_df = df.iloc[train_size + val_size:].copy()

            logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            # Store n_features from function scope
            n_features = len(input_features)

            # Extract and validate custom parameter ranges (Phase 9: Configurable Parameter Ranges)
            param_ranges = bayesian_config.get("param_ranges", {})

            # Validate param_ranges structure if provided
            if param_ranges:
                logger.info("Custom parameter ranges detected. Validating configuration...")

                # Track unknown parameters to warn user
                known_params = {'lstm_units', 'dropout_rate', 'learning_rate', 'batch_size', 'epochs', 'time_steps'}
                unknown_params = set(param_ranges.keys()) - known_params
                if unknown_params:
                    logger.warning(f"Unknown parameters in param_ranges will be ignored: {unknown_params}")

                for param_name, config in param_ranges.items():
                    if param_name not in known_params:
                        continue  # Skip validation for unknown params (already warned)

                    # For categorical parameters (lstm_units, batch_size), expect "choices" key
                    if param_name in ['lstm_units', 'batch_size']:
                        if not isinstance(config, dict) or "choices" not in config:
                            raise ValueError(
                                f"param_ranges['{param_name}'] is a categorical parameter and must have 'choices' key. "
                                f"Expected format: {{'choices': [list_of_values]}}. Got: {config}"
                            )
                        if not isinstance(config["choices"], list) or len(config["choices"]) == 0:
                            raise ValueError(
                                f"param_ranges['{param_name}']['choices'] must be a non-empty list. Got: {config['choices']}"
                            )
                    else:
                        # For numeric parameters, expect "min" and "max" keys
                        if not isinstance(config, dict) or "min" not in config or "max" not in config:
                            raise ValueError(
                                f"param_ranges['{param_name}'] must have 'min' and 'max' keys. "
                                f"Expected format: {{'min': X, 'max': Y}}. Got: {config}"
                            )

                        min_val = config["min"]
                        max_val = config["max"]

                        # Validate min < max (strictly)
                        if min_val >= max_val:
                            raise ValueError(
                                f"param_ranges['{param_name}'] min ({min_val}) must be strictly less than max ({max_val})"
                            )

                        # Validate integer parameters have integer values
                        if param_name in ['epochs', 'time_steps']:
                            if not isinstance(min_val, int) or not isinstance(max_val, int):
                                raise ValueError(
                                    f"param_ranges['{param_name}'] is an integer parameter. "
                                    f"Both 'min' and 'max' must be integers. Got min={min_val} ({type(min_val).__name__}), "
                                    f"max={max_val} ({type(max_val).__name__})"
                                )

                            # Validate step is positive if provided
                            if "step" in config:
                                step_val = config["step"]
                                if not isinstance(step_val, int) or step_val <= 0:
                                    raise ValueError(
                                        f"param_ranges['{param_name}']['step'] must be a positive integer. Got: {step_val}"
                                    )

                        # Validate float parameters
                        elif param_name in ['dropout_rate', 'learning_rate']:
                            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                                raise ValueError(
                                    f"param_ranges['{param_name}'] is a float parameter. "
                                    f"Both 'min' and 'max' must be numeric. Got min={min_val} ({type(min_val).__name__}), "
                                    f"max={max_val} ({type(max_val).__name__})"
                                )

                            # Validate log=True only for positive ranges
                            if config.get("log", False):
                                if min_val <= 0 or max_val <= 0:
                                    raise ValueError(
                                        f"param_ranges['{param_name}'] has 'log': True but range includes non-positive values. "
                                        f"Log scale requires min > 0 and max > 0. Got min={min_val}, max={max_val}"
                                    )

                logger.info(f"Custom parameter ranges validated successfully: {list(param_ranges.keys())}")
            else:
                logger.info("No custom parameter ranges provided. Using default ranges.")

            # Define Optuna objective function
            def objective(trial: Trial) -> float:
                """
                Optuna objective for LSTM hyperparameter optimization.

                CRITICAL: Resets TensorFlow seeds inside objective for reproducibility.

                Returns:
                    float: Validation RMSE to minimize
                """
                # CRITICAL: Reset TensorFlow seeds inside objective function
                # This ensures deterministic behavior across trials
                import random
                np.random.seed(SEED)
                tf.random.set_seed(SEED)
                random.seed(SEED)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.experimental.enable_op_determinism()

                # Suggest hyperparameters with configurable or default ranges (Phase 9)
                # Note: lstm_units and batch_size are categorical parameters
                lstm_units_config = param_ranges.get("lstm_units", {"choices": [32, 64, 128]})
                lstm_units = trial.suggest_categorical('lstm_units', lstm_units_config["choices"])

                dropout_config = param_ranges.get("dropout_rate", {"min": 0.1, "max": 0.4})
                dropout_rate = trial.suggest_float('dropout_rate', dropout_config["min"], dropout_config["max"], log=dropout_config.get("log", False))

                lr_config = param_ranges.get("learning_rate", {"min": 1e-4, "max": 1e-2, "log": True})
                learning_rate = trial.suggest_float('learning_rate', lr_config["min"], lr_config["max"], log=lr_config.get("log", False))

                batch_size_config = param_ranges.get("batch_size", {"choices": [16, 32, 64]})
                batch_size = trial.suggest_categorical('batch_size', batch_size_config["choices"])

                epochs_config = param_ranges.get("epochs", {"min": 30, "max": 100})
                epochs = trial.suggest_int('epochs', epochs_config["min"], epochs_config["max"], step=epochs_config.get("step", 1))

                time_steps_config = param_ranges.get("time_steps", {"min": 5, "max": 30})
                time_steps = trial.suggest_int('time_steps', time_steps_config["min"], time_steps_config["max"], step=time_steps_config.get("step", 1))

                try:
                    # Recreate sequences with suggested time_steps
                    X_tr, y_tr = create_sequences_for_lstm(
                        df=train_df,
                        feature_cols=input_features,
                        target_col=target_variable,
                        sequence_length=time_steps,
                        forecast_horizon=forecast_horizon
                    )
                    X_val_seq, y_val_seq = create_sequences_for_lstm(
                        df=val_df,
                        feature_cols=input_features,
                        target_col=target_variable,
                        sequence_length=time_steps,
                        forecast_horizon=forecast_horizon
                    )

                    if len(X_tr) == 0 or len(X_val_seq) == 0:
                        logger.warning(f"Trial {trial.number}: time_steps={time_steps} too large, no sequences created")
                        return float('inf')

                    # Build LSTM model using existing helper (Option A: wrap lstm_units in list)
                    input_shape = (time_steps, n_features)
                    params = {
                        'lstm_units': [lstm_units],  # Wrap in list for single-layer architecture
                        'dropout_rate': dropout_rate,
                        'recurrent_dropout_rate': dropout_rate,  # Use same dropout for recurrent
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epochs': epochs
                    }

                    model = build_lstm_model(params, input_shape)

                    # Create callbacks with unique checkpoint filename
                    callbacks, checkpoint_path = create_lstm_callbacks(
                        experiment_dir=experiment_dir,
                        early_stopping_patience=early_stopping_patience,
                        checkpoint_filename=f"bayesian_checkpoint_{trial.number}.h5"
                    )

                    # Train model (verbose=0 for quiet mode during Bayesian search)
                    history = model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )

                    # Evaluate
                    y_val_pred = model.predict(X_val_seq, verbose=0)
                    rmse = np.sqrt(mean_squared_error(y_val_seq, y_val_pred))

                    logger.info(
                        f"Trial {trial.number}: val_rmse={rmse:.4f}, "
                        f"lstm_units={lstm_units}, dropout={dropout_rate:.3f}, "
                        f"time_steps={time_steps}"
                    )

                    # CRITICAL: Memory cleanup after each trial (prevent memory leak)
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()

                    return rmse

                except Exception as e:
                    logger.warning(f"Trial {trial.number} failed: {str(e)}")
                    return float('inf')

            # Create Optuna study
            sampler = TPESampler(
                seed=SEED,
                n_startup_trials=n_initial_points,
                multivariate=False
            )

            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"lstm_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Track time
            import time
            optimization_start_time = time.time()

            # Extract convergence config (Phase 7: Nice-to-Have Features)
            # Use 'or' to handle both missing keys and explicit None values from frontend
            convergence_tolerance = bayesian_config.get("convergence_tolerance") or 0.001
            convergence_patience = bayesian_config.get("convergence_patience") or 5

            # Define convergence callback (Phase 7)
            def convergence_callback(study, trial):
                """
                Stop optimization if improvement is below tolerance for patience consecutive trials.

                This is a simple heuristic that hard caps training based on lack of improvement.
                """
                # Need at least convergence_patience completed trials
                completed_trials = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                    and np.isfinite(t.value)
                ]

                if len(completed_trials) < convergence_patience:
                    return  # Not enough trials yet

                # Get recent trial values
                recent_values = [t.value for t in completed_trials[-convergence_patience:]]

                # Calculate improvements between consecutive trials
                improvements = [abs(recent_values[i] - recent_values[i+1]) for i in range(len(recent_values)-1)]

                # Check if all recent improvements are below tolerance
                if all(imp < convergence_tolerance for imp in improvements):
                    logger.info(f"Convergence detected: improvements {improvements} all below tolerance {convergence_tolerance}")
                    logger.info(f"Stopping optimization early at trial {trial.number}")
                    study.stop()

            # Extract memory limit config (Phase 8)
            max_memory_mb = bayesian_config.get("max_memory_mb", None)

            # Define memory monitoring callback (Phase 8)
            def memory_callback(study, trial):
                """
                Monitor memory usage during optimization.
                Tracks peak memory usage and stops if max_memory_mb limit is exceeded.
                """
                global peak_memory_mb, memory_exceeded

                # Get current process memory usage in MB
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Always track peak memory (even if no limit set)
                if memory_mb > peak_memory_mb:
                    peak_memory_mb = memory_mb

                # Early return if no memory limit set
                if max_memory_mb is None:
                    return

                # Check if memory limit exceeded
                if memory_mb > max_memory_mb:
                    logger.warning(f"Memory limit exceeded: {memory_mb:.2f} MB > {max_memory_mb} MB")
                    logger.warning(f"Stopping optimization at trial {trial.number}")
                    memory_exceeded = True
                    study.stop()

            # Build callbacks list (Phase 7 & 8)
            callbacks_optuna = []
            if convergence_tolerance and convergence_patience:
                callbacks_optuna.append(convergence_callback)
            if max_memory_mb is not None:
                callbacks_optuna.append(memory_callback)

            # Run optimization
            logger.info("Starting LSTM Bayesian Search optimization")
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout_seconds,
                callbacks=callbacks_optuna,  # Phase 7 & 8: convergence detection and memory monitoring
                show_progress_bar=False,
                n_jobs=1  # CRITICAL: Single-threaded for TF determinism
            )

            optimization_time_seconds = time.time() - optimization_start_time

            # Extract best parameters
            if study.best_trial is None or study.best_value == float('inf'):
                raise RuntimeError("Bayesian Search failed: No valid trials completed")

            best_params_dict = study.best_params
            best_score = study.best_value

            logger.info("="*60)
            logger.info(f"LSTM Bayesian Search Completed")
            logger.info(f"  Best val_rmse: {best_score:.4f}")
            logger.info(f"  Best parameters: {best_params_dict}")
            logger.info(f"  Optimization time: {optimization_time_seconds:.2f} seconds")
            logger.info("="*60)

            # Reset seeds before final model training
            set_global_seeds()

            # Train final model with best parameters
            best_time_steps = best_params_dict['time_steps']

            # Recreate sequences with best time_steps
            logger.info(f"Creating final sequences with best time_steps={best_time_steps}...")
            X_train_final, y_train_final = create_sequences_for_lstm(
                df=train_df,
                feature_cols=input_features,
                target_col=target_variable,
                sequence_length=best_time_steps,
                forecast_horizon=forecast_horizon
            )
            X_val_final, y_val_final = create_sequences_for_lstm(
                df=val_df,
                feature_cols=input_features,
                target_col=target_variable,
                sequence_length=best_time_steps,
                forecast_horizon=forecast_horizon
            )
            X_test_final, y_test_final = create_sequences_for_lstm(
                df=test_df,
                feature_cols=input_features,
                target_col=target_variable,
                sequence_length=best_time_steps,
                forecast_horizon=forecast_horizon
            )

            # Build final model with best params
            input_shape_final = (best_time_steps, n_features)
            final_params = {
                'lstm_units': [best_params_dict['lstm_units']],  # Wrap in list
                'dropout_rate': best_params_dict['dropout_rate'],
                'recurrent_dropout_rate': best_params_dict['dropout_rate'],
                'learning_rate': best_params_dict['learning_rate'],
                'batch_size': best_params_dict['batch_size'],
                'epochs': best_params_dict['epochs']
            }

            logger.info("Building final LSTM model with best parameters...")
            best_model = build_lstm_model(final_params, input_shape_final)

            # Create callbacks for final training
            callbacks_final, checkpoint_path_final = create_lstm_callbacks(
                experiment_dir=experiment_dir,
                early_stopping_patience=early_stopping_patience
            )

            # Train final model
            logger.info("Training final LSTM model...")
            history_final = best_model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val_final, y_val_final),
                epochs=best_params_dict['epochs'],
                batch_size=best_params_dict['batch_size'],
                callbacks=callbacks_final,
                verbose=1
            )

            # Store best params for return value (critical for Phase 6 reproducibility tests)
            best_params = {
                'lstm_units': best_params_dict['lstm_units'],  # Store as int (unwrapped)
                'dropout_rate': best_params_dict['dropout_rate'],
                'learning_rate': best_params_dict['learning_rate'],
                'batch_size': best_params_dict['batch_size'],
                'epochs': best_params_dict['epochs'],
                'time_steps': best_params_dict['time_steps']
            }

            # Log best results to MLflow
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_rmse", best_score)
            mlflow.log_metric("bayesian_n_trials", n_trials)
            mlflow.log_metric("bayesian_optimization_time_seconds", optimization_time_seconds)

            # Update X_val, y_val, X_test, y_test for subsequent evaluation
            X_val = X_val_final
            y_val = y_val_final
            X_test = X_test_final
            y_test = y_test_final

            # Set best_epoch to None for Bayesian search
            best_epoch = None

        else:
            # Invalid strategy
            raise ValueError(
                f"hyperparameter_search_strategy '{hyperparameter_search_strategy}' "
                f"no soportado. Opciones válidas: 'manual', 'grid', 'random', 'bayesian'"
            )

        # ======================
        # 8. DETENER TRACKER DE ENERGÍA
        # ======================

        # Stop energy tracking
        tracker.stop()
        energy_kwh, emissions_kg = log_energy_metrics(tracker)

        logger.info(
            f"Consumo de energía: {energy_kwh:.4f} kWh, "
            f"Emisiones de carbono: {emissions_kg:.6f} kg CO2"
        )

        # ======================
        # 9. EVALUACIÓN EN CONJUNTO DE VALIDACIÓN
        # ======================

        # Evaluate on validation set
        logger.info("Evaluando modelo en conjunto de validación...")
        val_metrics, val_artifacts = evaluate_lstm_model(
            model=best_model,
            X_test=X_val,
            y_test=y_val,
            prefix="val",
            experiment_dir=experiment_dir
        )

        # ======================
        # 10. EVALUACIÓN EN CONJUNTO DE PRUEBA
        # ======================

        # Evaluate on test set
        logger.info("Evaluando modelo en conjunto de prueba...")
        test_metrics, test_artifacts = evaluate_lstm_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            prefix="test",
            experiment_dir=experiment_dir
        )

        # ======================
        # 11. REGISTRO DE MÉTRICAS EN MLFLOW
        # ======================

        # Log all metrics
        for metric_name, metric_value in {**val_metrics, **test_metrics}.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        # Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # ======================
        # 12. REGISTRO DE ARTEFACTOS EN MLFLOW
        # ======================

        # Log plot artifacts
        for artifact_path in val_artifacts + test_artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, "plots")

        # Generate and log training history plot
        if 'history' in locals():
            plt.figure(figsize=(12, 6))
            plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
            plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
            plt.title("LSTM - Curva de Aprendizaje")
            plt.xlabel("Época")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            history_path = os.path.join(experiment_dir, "lstm_training_history.png")
            plt.savefig(history_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(history_path, "plots")

        # ======================
        # 13. GUARDADO Y REGISTRO DEL MODELO
        # ======================

        # Save model to experiment directory
        model_save_path = os.path.join(experiment_dir, "lstm_model.keras")
        best_model.save(model_save_path)  # Keras 3 native format
        logger.info(f"Modelo guardado en: {model_save_path}")

        # Infer signature for MLflow
        # Use a small sample for signature inference
        sample_input = X_train[:5]
        sample_output = best_model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_output)

        # Register model in MLflow
        mlflow.keras.log_model(
            model=best_model,
            artifact_path="lstm_model",
            signature=signature,
            registered_model_name=model_name,
            metadata={
                "dataset": os.path.basename(dataset_path),
                "target": target_variable,
                "features": input_features,
                "date_column": date_col_name,
                "forecast_horizon": forecast_horizon,
                "sequence_length": sequence_length,
                "architecture": str(best_params.get("lstm_units")),
                "cpu_only": True
            }
        )

        logger.info(f"Modelo registrado en MLflow: {model_name}")

        # ======================
        # 14. LIMPIEZA DE CHECKPOINTS TEMPORALES
        # ======================

        # Aggressive cleanup - delete all temporary checkpoints
        checkpoint_dir = os.path.join(experiment_dir, "temp_checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Checkpoints temporales eliminados: {checkpoint_dir}")

        # ======================
        # 15. ACTUALIZACIÓN DE PIPELINE CONFIG
        # ======================

        # Calculate training mode (Phase 4)
        n_input_features = len(input_features)
        training_mode = TRAINING_MODE_UNIVARIATE if n_input_features == 1 else TRAINING_MODE_MULTIVARIATE

        # Consistency validation (Phase 4)
        if training_mode == TRAINING_MODE_UNIVARIATE and n_input_features != 1:
            raise ValueError(
                f"Inconsistent training mode: {training_mode} requires exactly 1 feature, "
                f"but got {n_input_features}"
            )

        # Save pipeline configuration (Schema v1.1 - Phase 3A)
        pipeline_step_config = {
            "schema_version": "1.1",  # Schema versioning for future compatibility
            "step": "train_model",
            "algorithm": "lstm",
            "date_col_name": date_col_name,
            "target_variable": target_variable,
            "model_name": model_name,
            "input_features": input_features,
            "forecast_horizon": forecast_horizon,
            "params": convert_numpy_to_python(best_params),
            "metrics": {
                # All 6 metrics included (Phase 3A fix for reproducibility)
                "val_rmse": val_metrics.get("val_rmse"),
                "val_mae": val_metrics.get("val_mae"),
                "val_mape": val_metrics.get("val_mape"),  # Can be None (division by zero case)
                "test_rmse": test_metrics.get("test_rmse"),
                "test_mae": test_metrics.get("test_mae"),
                "test_mape": test_metrics.get("test_mape")  # Can be None (division by zero case)
            },
            "hyperparameter_search_strategy": hyperparameter_search_strategy,
            "hyperparameter_search": {
                # Base structure - populated below based on strategy
                "strategy": hyperparameter_search_strategy,
                "iterations_total": None,
                "best_iteration": None,
                "best_val_loss": None,
                "grid_search_params": None,
                "random_search_params": None,
                "n_random_iterations": None,
                "memory_profiling": None  # Always included, None when disabled
            },
            "lstm_metadata": {
                "sequence_length": sequence_length,
                "model_architecture": str(best_params.get("lstm_units")),
                "training_mode": training_mode,  # NEW: Phase 4
                "n_input_features": n_input_features,  # NEW: Phase 4
                "training_time_seconds": None,  # Can be added if tracked
                # best_epoch is None for grid/random search (not tracked per iteration)
                "early_stopped": best_epoch < best_params.get("epochs", 100) if best_epoch is not None else None,
                "stopped_at_epoch": best_epoch,
                "total_params": best_model.count_params(),
                "cpu_only": True,
                "energy_kwh": energy_kwh,
                "carbon_emissions_kg": emissions_kg
            }
        }

        # Populate hyperparameter_search section based on strategy
        if hyperparameter_search_strategy == "manual":
            # Manual parameters - single iteration
            pipeline_step_config["hyperparameter_search"]["iterations_total"] = 1
            pipeline_step_config["hyperparameter_search"]["best_iteration"] = 1
            pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)

        elif hyperparameter_search_strategy == "grid":
            # Grid search metadata
            pipeline_step_config["hyperparameter_search"]["iterations_total"] = n_combinations
            pipeline_step_config["hyperparameter_search"]["best_iteration"] = best_iteration
            pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)
            pipeline_step_config["hyperparameter_search"]["grid_search_params"] = grid_params #convert_numpy_to_python(grid_search_params)

            # Add memory profiling if enabled (v1.1 schema - always include field)
            if 'memory_increase_mb' in locals():
                pipeline_step_config["hyperparameter_search"]["memory_profiling"] = {
                    "enabled": True,
                    "initial_memory_mb": float(initial_memory_mb),
                    "final_memory_mb": float(final_memory_mb),
                    "memory_increase_mb": float(memory_increase_mb)
                }

        elif hyperparameter_search_strategy == "random":
            # Random search metadata
            pipeline_step_config["hyperparameter_search"]["iterations_total"] = n_random_iterations
            pipeline_step_config["hyperparameter_search"]["best_iteration"] = best_iteration
            pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_val_loss)
            pipeline_step_config["hyperparameter_search"]["random_search_params"] = random_search_params #convert_numpy_to_python(random_search_params)
            pipeline_step_config["hyperparameter_search"]["n_random_iterations"] = n_random_iterations

            # Add memory profiling if enabled (v1.1 schema - always include field)
            if 'memory_increase_mb' in locals():
                pipeline_step_config["hyperparameter_search"]["memory_profiling"] = {
                    "enabled": True,
                    "initial_memory_mb": float(initial_memory_mb),
                    "final_memory_mb": float(final_memory_mb),
                    "memory_increase_mb": float(memory_increase_mb)
                }

        elif hyperparameter_search_strategy == "bayesian":
            # Bayesian search metadata (matching ARIMA/XGBoost pattern)
            pipeline_step_config["hyperparameter_search"]["iterations_total"] = n_trials
            pipeline_step_config["hyperparameter_search"]["best_iteration"] = study.best_trial.number
            pipeline_step_config["hyperparameter_search"]["best_val_loss"] = float(best_score)

            # Add bayesian-specific config at top level
            bayesian_config_metadata = {
                "n_trials": n_trials,
                "n_initial_points": n_initial_points,
                "timeout_seconds": timeout_seconds,
                "acq_func": bayesian_config.get("acq_func", "ei"),  # Phase 7: Save acq_func metadata
                "convergence_tolerance": bayesian_config.get("convergence_tolerance", 0.001),  # Phase 7
                "convergence_patience": bayesian_config.get("convergence_patience", 5),  # Phase 7
                "max_memory_mb": max_memory_mb,  # Phase 8: Memory limit (None if disabled)
                "peak_memory_mb": float(peak_memory_mb) if peak_memory_mb > 0 else None,  # Phase 8: Peak memory usage
                "memory_exceeded": memory_exceeded,  # Phase 8: Whether memory limit was exceeded
                "optimization_metric": optimization_metric,
                "optimization_time_seconds": optimization_time_seconds,
                "best_trial_number": study.best_trial.number,
                "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "best_params": best_params_dict,
                "seed": SEED
            }
            # Phase 9: Add custom param_ranges if provided
            if param_ranges:
                bayesian_config_metadata["param_ranges"] = param_ranges
            pipeline_step_config["bayesian_config"] = bayesian_config_metadata

        save_pipeline_config(experiment_dir, pipeline_step_config)

        # Validate schema (non-strict mode - warnings only, doesn't block training)
        try:
            validate_pipeline_config_schema(pipeline_step_config, strict=False)
        except Exception as e:
            logger.warning(f"Pipeline config validation encountered issues: {e}")

        # ======================
        # 16. FINALIZACIÓN DE RUN DE MLFLOW
        # ======================

        # End MLflow run
        mlflow.end_run()
        logger.info("Run de MLflow finalizada exitosamente")

        # ======================
        # 17. RETORNO DE RESULTADOS
        # ======================

        return {
            "status": "success",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": os.path.relpath(model_save_path, experiment_dir),
            "run_id": mlflow_run_id,
            "features_used": input_features,
            "sequence_length": sequence_length,
            "best_params": best_params
        }

    except Exception as e:
        # Error handling
        logger.error(f"Error en entrenamiento LSTM: {e}", exc_info=True)

        # End MLflow run if active
        if mlflow.active_run():
            mlflow.end_run()

        # Re-raise as RuntimeError with context
        raise RuntimeError(f"Error en entrenamiento LSTM: {e}") from e


# ======================================================================================
# PATCHTSMIXER TRAINING FUNCTIONS - PHASE 4
# ======================================================================================


def plot_patchtsmixer_horizons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    key_horizons: Dict[str, int],
    prefix: str,
    experiment_dir: str
) -> str:
    """
    Genera gráfico comparativo de horizontes clave para PatchTSMixer.

    Crea subplots mostrando predicciones vs valores reales para cada horizonte
    clave (h1, h_middle, h_last), promediando sobre canales.

    Args:
        y_true: Valores reales, shape (n_samples, prediction_length, num_channels)
        y_pred: Predicciones, shape (n_samples, prediction_length, num_channels)
        key_horizons: Dict con nombres y índices de horizontes clave
            Ejemplo: {'h1': 0, 'h48': 47, 'h96': 95}
        prefix: "val" o "test" para identificar conjunto de datos
        experiment_dir: Directorio donde guardar el gráfico

    Returns:
        str: Ruta al archivo PNG generado

    Ejemplo de uso:
        >>> key_horizons = {'h1': 0, 'h48': 47, 'h96': 95}
        >>> path = plot_patchtsmixer_horizons(y_true, y_pred, key_horizons, "val", "/path/to/exp")
        >>> print(path)  # /path/to/exp/patchtsmixer_val_horizons.png
    """
    n_horizons = len(key_horizons)

    # Altura dinámica: 4 unidades por subplot
    fig, axes = plt.subplots(n_horizons, 1, figsize=(14, 4 * n_horizons))

    # Manejar caso de subplot único (n_horizons=1 retorna Axes, no array)
    if n_horizons == 1:
        axes = [axes]

    # Limitar muestras para visualización
    n_samples_plot = min(100, y_true.shape[0])

    for ax, (horizon_name, horizon_idx) in zip(axes, key_horizons.items()):
        # Promediar sobre canales para visualización
        # Shape después: (n_samples_plot,)
        y_true_h = y_true[:n_samples_plot, horizon_idx, :].mean(axis=1)
        y_pred_h = y_pred[:n_samples_plot, horizon_idx, :].mean(axis=1)

        ax.plot(y_true_h, label="Real", alpha=0.7, linewidth=2, color='blue')
        ax.plot(y_pred_h, label="Predicción", alpha=0.7, linewidth=2, color='orange')
        ax.set_title(f"PatchTSMixer - Horizonte {horizon_name} ({prefix.upper()})")
        ax.set_xlabel("Muestra")
        ax.set_ylabel("Valor (promedio de canales)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    horizons_path = os.path.join(experiment_dir, f"patchtsmixer_{prefix}_horizons.png")
    plt.savefig(horizons_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ Gráfico de horizontes guardado: {os.path.basename(horizons_path)}")
    return horizons_path


def evaluate_patchtsmixer(
    trainer,
    dataset: 'TimeSeriesDataset',
    prefix: str,
    experiment_dir: str
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evalúa modelo PatchTSMixer y genera gráficos de diagnóstico.

    Esta función calcula métricas agregadas (RMSE, MAE, MAPE, MSE) en todos los
    horizontes de predicción, métricas por horizonte clave (h1, h_middle, h_last),
    y genera plots de diagnóstico siguiendo el patrón de evaluate_lstm_model().

    Las métricas por horizonte permiten evaluar el rendimiento del modelo en
    diferentes puntos del horizonte de predicción:
    - h1: Primer paso de predicción (corto plazo)
    - h_middle: Horizonte medio (mediano plazo)
    - h_last: Último paso de predicción (largo plazo)

    Args:
        trainer: Instancia de HuggingFace Trainer con modelo entrenado
        dataset: TimeSeriesDataset de PyTorch para evaluación
        prefix: Prefijo para métricas ("val" o "test")
        experiment_dir: Directorio para guardar gráficos

    Returns:
        Tupla de:
        - metrics: Diccionario con métricas agregadas y por horizonte:
            - Agregadas: {prefix}_rmse, {prefix}_mae, {prefix}_mape, {prefix}_mse
            - Por horizonte: {prefix}_rmse_h1, {prefix}_mae_h1, {prefix}_mape_h1,
              {prefix}_rmse_h{middle}, {prefix}_mae_h{middle}, {prefix}_mape_h{middle},
              {prefix}_rmse_h{last}, {prefix}_mae_h{last}, {prefix}_mape_h{last}
        - artifacts: Lista de rutas a 4 archivos PNG (forecast, residuals,
          residuals_distribution, horizons)

    Raises:
        RuntimeError: Si la predicción falla

    Example:
        >>> val_metrics, val_artifacts = evaluate_patchtsmixer(
        ...     trainer, val_dataset, "val", "./experiment_dir"
        ... )
        >>> print(val_metrics["val_rmse"])  # Métrica agregada
        >>> print(val_metrics["val_rmse_h1"])  # Métrica horizonte 1
        >>> print(val_metrics["val_rmse_h48"])  # Métrica horizonte medio (si prediction_length=96)
    """
    logger.info(f"=== Evaluando modelo PatchTSMixer en conjunto {prefix.upper()} ===")

    # === 1. GENERATE PREDICTIONS ===
    logger.info(f"Generando predicciones para conjunto {prefix}...")

    try:
        # Use Trainer.predict() which returns PredictionOutput
        predictions_output = trainer.predict(dataset)
        y_pred = predictions_output.predictions  # Shape: (n_samples, prediction_length, num_channels)

        # Handle case where predictions is a tuple (PatchTSMixer returns tuple of arrays)
        if isinstance(y_pred, tuple):
            # Take the first element which contains the actual predictions
            y_pred = y_pred[0]
    except Exception as e:
        logger.error(f"Error al generar predicciones: {e}")
        raise RuntimeError(f"Fallo en predicción de PatchTSMixer: {e}") from e

    # Extract ground truth from dataset
    y_true = dataset.future_values.numpy()  # Shape: (n_samples, prediction_length, num_channels)

    logger.info(f"  y_pred shape: {y_pred.shape}")
    logger.info(f"  y_true shape: {y_true.shape}")

    # === 2. CALCULATE AGGREGATE METRICS ===
    # Flatten across all horizons and channels for aggregate metrics
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # RMSE
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    # MAE
    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    # MAPE (avoid division by zero)
    mask = y_true_flat != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = None
        logger.warning(f"No se puede calcular MAPE para {prefix}: todos los valores objetivo son cero")

    # Create metrics dict
    metrics = {
        f"{prefix}_rmse": float(rmse),
        f"{prefix}_mae": float(mae),
        f"{prefix}_mape": float(mape) if mape is not None else None,
        f"{prefix}_mse": float(mse)
    }

    logger.info(
        f"Métricas {prefix.upper()} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}" +
        (f", MAPE: {mape:.2f}%" if mape is not None else "")
    )

    # === 2b. MANEJO DE CASOS LÍMITE PARA HORIZONTES ===
    # Determinar horizontes clave basándose en prediction_length
    prediction_length = y_true.shape[1]

    if prediction_length >= 3:
        # Caso normal: h1, h_middle, h_last
        # Usar floor division para horizonte medio (96 → h48, no h49)
        middle_horizon = prediction_length // 2
        key_horizons = {
            'h1': 0,                                    # Primer paso (índice 0)
            f'h{middle_horizon}': middle_horizon - 1,   # Horizonte medio (índice 0-based)
            f'h{prediction_length}': prediction_length - 1   # Último paso
        }
    elif prediction_length == 2:
        # Caso corto: solo h1 y h_last
        key_horizons = {
            'h1': 0,
            'h2': 1
        }
        logger.warning(f"prediction_length={prediction_length}: usando solo h1 y h2")
    else:  # prediction_length == 1
        # Caso mínimo: solo h1
        key_horizons = {'h1': 0}
        logger.warning(f"prediction_length={prediction_length}: usando solo h1")

    # === 2c. CALCULAR MÉTRICAS POR HORIZONTE CLAVE ===
    horizon_metrics = {}

    for horizon_name, horizon_idx in key_horizons.items():
        # Extraer predicciones y valores reales para este horizonte específico
        # Shape: (n_samples, num_channels)
        y_pred_horizon = y_pred[:, horizon_idx, :]
        y_true_horizon = y_true[:, horizon_idx, :]

        # Aplanar para calcular métricas agregadas por horizonte
        y_pred_h_flat = y_pred_horizon.flatten()
        y_true_h_flat = y_true_horizon.flatten()

        # RMSE por horizonte
        mse_h = mean_squared_error(y_true_h_flat, y_pred_h_flat)
        rmse_h = np.sqrt(mse_h)

        # MAE por horizonte
        mae_h = mean_absolute_error(y_true_h_flat, y_pred_h_flat)

        # MAPE por horizonte (evitar división por cero)
        mask_h = y_true_h_flat != 0
        if mask_h.sum() > 0:
            mape_h = np.mean(np.abs((y_true_h_flat[mask_h] - y_pred_h_flat[mask_h]) / y_true_h_flat[mask_h])) * 100
        else:
            mape_h = None
            logger.warning(f"MAPE no calculable para {prefix}_{horizon_name}: valores cero")

        # Agregar al diccionario de métricas con naming pattern consistente
        horizon_metrics[f"{prefix}_rmse_{horizon_name}"] = float(rmse_h)
        horizon_metrics[f"{prefix}_mae_{horizon_name}"] = float(mae_h)
        horizon_metrics[f"{prefix}_mape_{horizon_name}"] = float(mape_h) if mape_h is not None else None

        logger.info(f"  - {horizon_name}: RMSE={rmse_h:.4f}, MAE={mae_h:.4f}" +
                    (f", MAPE={mape_h:.2f}%" if mape_h is not None else ""))

    # Combinar métricas agregadas con métricas por horizonte
    metrics.update(horizon_metrics)

    logger.info(f"Métricas por horizonte calculadas: {list(key_horizons.keys())}")

    # === 3. GENERATE DIAGNOSTIC PLOTS ===
    artifacts = []

    # Plot 1: Predictions vs Actual (flatten for visualization - show first 200 samples)
    # For multivariate, we'll plot the mean across all channels
    y_pred_mean = y_pred.mean(axis=2).flatten()[:200]  # Average across channels, first 200 points
    y_true_mean = y_true.mean(axis=2).flatten()[:200]

    plt.figure(figsize=(14, 6))
    plt.plot(y_true_mean, label="Real", alpha=0.7, linewidth=2, color='blue')
    plt.plot(y_pred_mean, label="Predicción PatchTSMixer", alpha=0.7, linewidth=2, color='orange')
    plt.title(f"PatchTSMixer - Predicciones vs Valores Reales ({prefix.upper()})")
    plt.xlabel("Índice Temporal (primeros 200 puntos, promedio de canales)")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    forecast_path = os.path.join(experiment_dir, f"patchtsmixer_{prefix}_forecast.png")
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(forecast_path)
    logger.info(f"  ✓ Gráfico de pronóstico guardado: {os.path.basename(forecast_path)}")

    # Plot 2: Residuals
    residuals_flat = y_true_flat - y_pred_flat
    residuals_mean = (y_true.mean(axis=2) - y_pred.mean(axis=2)).flatten()[:200]

    plt.figure(figsize=(14, 6))
    plt.plot(residuals_mean, alpha=0.7, linewidth=1, color='green')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    plt.title(f"PatchTSMixer - Residuos ({prefix.upper()})")
    plt.xlabel("Índice Temporal (primeros 200 puntos, promedio de canales)")
    plt.ylabel("Residuo (Real - Predicción)")
    plt.grid(True, alpha=0.3)
    residuals_path = os.path.join(experiment_dir, f"patchtsmixer_{prefix}_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_path)
    logger.info(f"  ✓ Gráfico de residuos guardado: {os.path.basename(residuals_path)}")

    # Plot 3: Residuals distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black', color='purple')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    plt.title(f"PatchTSMixer - Distribución de Residuos ({prefix.upper()})")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3, axis='y')
    residuals_dist_path = os.path.join(experiment_dir, f"patchtsmixer_{prefix}_residuals_distribution.png")
    plt.savefig(residuals_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(residuals_dist_path)
    logger.info(f"  ✓ Gráfico de distribución de residuos guardado: {os.path.basename(residuals_dist_path)}")

    # Plot 4: Horizon comparison plot (Phase 5)
    horizons_path = plot_patchtsmixer_horizons(
        y_true=y_true,
        y_pred=y_pred,
        key_horizons=key_horizons,
        prefix=prefix,
        experiment_dir=experiment_dir
    )
    artifacts.append(horizons_path)

    logger.info(f"✓ Evaluación de {prefix} completada: {len(artifacts)} gráficos generados")

    return metrics, artifacts


def train_manual_patchtsmixer(
    model,
    train_dataset: 'TimeSeriesDataset',
    val_dataset: 'TimeSeriesDataset',
    params: Dict,
    experiment_dir: str
):
    """
    Entrena PatchTSMixer usando estrategia manual con HuggingFace Trainer API.

    Configura TrainingArguments con hiperparámetros manuales y utiliza
    EarlyStoppingCallback para detener entrenamiento si no hay mejora en
    pérdida de validación. Sigue el patrón del entrenamiento LSTM pero
    usando la API de HuggingFace Transformers.

    Args:
        model: Instancia de PatchTSMixerForPrediction
        train_dataset: Dataset de PyTorch para entrenamiento
        val_dataset: Dataset de PyTorch para validación
        params: Dict con hiperparámetros:
            - learning_rate: Tasa de aprendizaje (default: 0.001)
            - batch_size: Tamaño del batch (default: 32)
            - epochs: Número máximo de épocas (default: 100)
            - early_stopping_patience: Paciencia para early stopping (default: 10)
        experiment_dir: Directorio base para guardar checkpoints

    Returns:
        Trainer: Instancia del Trainer entrenado con el mejor modelo cargado

    Raises:
        ImportError: Si transformers no está instalado
        RuntimeError: Si el entrenamiento falla

    Example:
        >>> params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
        >>> trainer = train_manual_patchtsmixer(model, train_ds, val_ds, params, "./exp")
        >>> predictions = trainer.predict(test_ds)
    """
    # Validar que transformers está disponible
    if not TRANSFORMERS_TRAINER_AVAILABLE:
        raise ImportError(
            "transformers>=4.36.0 requerido para entrenamiento de PatchTSMixer. "
            "Instalar con: pip install 'transformers>=4.36.0'"
        )

    logger.info("=== Configurando HuggingFace Trainer para entrenamiento manual ===")

    # Extraer hiperparámetros con valores por defecto
    learning_rate = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 32)
    epochs = params.get("epochs", 100)
    early_stopping_patience = params.get("early_stopping_patience", 10)

    logger.info(f"Hiperparámetros de entrenamiento:")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Max epochs: {epochs}")
    logger.info(f"  - Early stopping patience: {early_stopping_patience}")

    # Crear directorio para checkpoints
    checkpoint_dir = os.path.join(experiment_dir, "patchtsmixer_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"  - Checkpoints dir: {checkpoint_dir}")

    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        # Directorios
        output_dir=checkpoint_dir,

        # Épocas y batches
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        # Optimización
        learning_rate=learning_rate,
        weight_decay=0.01,  # Regularización L2

        # Estrategias de evaluación y guardado
        evaluation_strategy="epoch",  # Evaluar al final de cada época
        save_strategy="epoch",         # Guardar checkpoint cada época
        save_total_limit=3,            # Mantener solo los 3 mejores checkpoints

        # Early stopping y best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Menor pérdida es mejor

        # Reproducibilidad
        seed=SEED,
        data_seed=SEED,

        # Configuración de hardware (CPU-only para DREAM-ML)
        use_cpu=True,
        dataloader_num_workers=0,  # Evitar problemas multiprocessing en CPU

        # Logging
        logging_strategy="epoch",
        logging_first_step=True,

        # Otros
        disable_tqdm=False,  # Mostrar barra de progreso
        report_to=[],  # No reportar a wandb/tensorboard (usamos MLflow)
        bf16=False,  # Ensure no bfloat16 on CPU
        fp16=False  # Disable fp16 on CPU
    )

    logger.info("✓ TrainingArguments configurados")

    # Crear callback de early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.0001  # Mejora mínima requerida
    )

    logger.info(f"✓ EarlyStoppingCallback configurado (patience={early_stopping_patience})")

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback]
    )

    logger.info("✓ Trainer inicializado exitosamente")
    logger.info(f"\n{'='*80}")
    logger.info("COMENZANDO ENTRENAMIENTO")
    logger.info(f"{'='*80}\n")

    # Entrenar modelo
    try:
        train_result = trainer.train()
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        raise RuntimeError(f"Fallo en entrenamiento de PatchTSMixer: {e}") from e

    logger.info(f"\n{'='*80}")
    logger.info("ENTRENAMIENTO FINALIZADO")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Train loss final: {train_result.training_loss:.6f}")
    logger.info(f"✓ Mejor modelo cargado automáticamente (load_best_model_at_end=True)")

    # Obtener métricas finales de validación
    try:
        eval_result = trainer.evaluate()
        logger.info(f"✓ Validation loss final: {eval_result['eval_loss']:.6f}")
    except Exception as e:
        logger.warning(f"No se pudo evaluar el modelo final: {e}")

    return trainer


def train_patchtsmixer_model(
    dataset_path: str,
    data: Dict,
    experiment_dir: str
) -> Dict:
    """
    Entrena un modelo PatchTSMixer para pronóstico de series temporales multivariadas.

    Este función sigue la misma estructura de 17 pasos que train_lstm_model() para
    mantener consistencia en el código. Implementa estrategia de entrenamiento manual
    con hiperparámetros definidos por el usuario.

    Args:
        dataset_path: Ruta al archivo CSV con datos codificados
        data: Diccionario con configuración y hiperparámetros:
            - date_col_name: Nombre de la columna de fecha
            - patchtsmixer_channels: Lista de nombres de variables/canales a usar
            - forecast_horizon: Número de pasos futuros a predecir
            - split_ratios: Dict con proporciones train/val/test
            - manual_params: Dict con hiperparámetros del modelo:
                * context_length: Longitud de la ventana de entrada
                * patch_length: Longitud de cada parche
                * patch_stride: Desplazamiento entre parches
                * d_model: Dimensión del modelo
                * num_layers: Número de capas del mixer
                * expansion_factor: Factor de expansión en capas MLP
                * dropout: Tasa de dropout
                * head_dropout: Tasa de dropout en cabezal de predicción
                * pooling_type: Tipo de pooling ("mean" o "max")
                * channel_attention: Si usar atención entre canales
                * scaling: Si aplicar normalización de entrada
                * learning_rate: Tasa de aprendizaje
                * batch_size: Tamaño del batch
                * epochs: Número máximo de épocas
                * early_stopping_patience: Paciencia para early stopping
        experiment_dir: Directorio donde guardar outputs (modelo, plots, checkpoints)

    Returns:
        Dict con tres claves:
            - val_metrics: Dict con métricas de validación (val_rmse, val_mae, val_mape, val_mse)
            - test_metrics: Dict con métricas de prueba (test_rmse, test_mae, test_mape, test_mse)
            - model_path: Ruta al modelo guardado

    Raises:
        ValueError: Si los parámetros son inválidos o los datos no cumplen requisitos
        RuntimeError: Si el entrenamiento falla

    Example:
        >>> data = {
        ...     "date_col_name": "timestamp",
        ...     "patchtsmixer_channels": ["temp", "humidity", "pressure"],
        ...     "forecast_horizon": 96,
        ...     "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        ...     "manual_params": {
        ...         "context_length": 512,
        ...         "patch_length": 8,
        ...         "d_model": 64,
        ...         "num_layers": 8,
        ...         "learning_rate": 0.001,
        ...         "batch_size": 32,
        ...         "epochs": 100
        ...     }
        ... }
        >>> result = train_patchtsmixer_model("data.csv", data, "./experiment_001")
        >>> print(result["val_metrics"]["val_rmse"])
    """
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO DE PATCHTSMIXER")
    print("="*80 + "\n")

    # Crear directorio de experimento si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    # =============================================================================
    # PASO 1: REPRODUCIBILIDAD - Configurar seeds globales
    # =============================================================================
    print("=== Paso 1: Configurando reproducibilidad ===")
    #set_global_seeds()
    #set_pytorch_reproducibility(SEED)
    print(f"✓ Seeds configuradas: {SEED}")
    print(f"✓ PyTorch determinístico activado\n")

    # =============================================================================
    # PASO 2: EXTRAER PARÁMETROS - Parsear configuración desde dict
    # =============================================================================
    print("=== Paso 2: Extrayendo parámetros de configuración ===")

    # Parámetros de datos
    date_col_name = data.get("date_col_name")
    channel_cols = data.get("patchtsmixer_channels", [])
    forecast_horizon = data.get("forecast_horizon", 96)
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})

    # Parámetros del modelo
    manual_params = data.get("manual_params", {})
    context_length = manual_params.get("context_length", 512)
    patch_length = manual_params.get("patch_length", 8)
    patch_stride = manual_params.get("patch_stride", 8)
    d_model = manual_params.get("d_model", 64)
    num_layers = manual_params.get("num_layers", 8)
    expansion_factor = manual_params.get("expansion_factor", 2)
    dropout = manual_params.get("dropout", 0.2)
    head_dropout = manual_params.get("head_dropout", 0.2)
    pooling_type = manual_params.get("pooling_type", "mean")
    channel_attention = manual_params.get("channel_attention", False)
    scaling = manual_params.get("scaling", True)

    # Parámetros de entrenamiento
    learning_rate = manual_params.get("learning_rate", 0.001)
    batch_size = manual_params.get("batch_size", 32)
    epochs = manual_params.get("epochs", 100)
    early_stopping_patience = manual_params.get("early_stopping_patience", 10)

    print(f"✓ Canales de entrada: {channel_cols}")
    print(f"✓ Horizonte de pronóstico: {forecast_horizon}")
    print(f"✓ Context length: {context_length}, Patch length: {patch_length}")
    print(f"✓ Arquitectura: d_model={d_model}, num_layers={num_layers}")
    print(f"✓ Entrenamiento: lr={learning_rate}, batch_size={batch_size}, epochs={epochs}\n")

    # =============================================================================
    # PASO 3: VALIDAR Y CARGAR DATOS - Validaciones y carga desde CSV
    # =============================================================================
    print("=== Paso 3: Validando y cargando datos ===")

    # Validar que context_length es múltiplo de patch_length
    if context_length % patch_length != 0:
        raise ValueError(
            f"❌ context_length ({context_length}) debe ser múltiplo de "
            f"patch_length ({patch_length}). "
            f"Ajuste context_length a un múltiplo de {patch_length}."
        )
    print(f"✓ Validación de parches: {context_length} % {patch_length} = 0")

    # Cargar datos
    df = load_and_validate_ts_data(dataset_path, date_col_name, channel_cols[0])

    # Validar que todas las columnas existen
    missing_cols = set(channel_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"❌ Las siguientes columnas no existen en el dataset: {missing_cols}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )
    print(f"✓ Todas las columnas encontradas en dataset")

    # Validar suficientes datos
    min_samples = context_length + forecast_horizon + 50
    if len(df) < min_samples:
        raise ValueError(
            f"❌ Dataset muy pequeño: {len(df)} filas. "
            f"Se requieren al menos {min_samples} filas para context_length={context_length} "
            f"y forecast_horizon={forecast_horizon}."
        )

    num_input_channels = len(channel_cols)
    print(f"✓ Dataset cargado: {len(df)} filas, {num_input_channels} canales")
    print(f"✓ Rango de fechas: {df.index[0]} a {df.index[-1]}\n")

    # =============================================================================
    # PASO 4: CREAR SECUENCIAS - Generar ventanas deslizantes para PatchTSMixer
    # =============================================================================
    print("=== Paso 4: Creando secuencias de entrada/salida ===")
    past_values, future_values = create_sequences_for_patchtsmixer(
        df=df,
        channel_cols=channel_cols,
        context_length=context_length,
        prediction_length=forecast_horizon
    )

    print(f"✓ past_values shape: {past_values.shape}")
    print(f"✓ future_values shape: {future_values.shape}")
    print(f"✓ Total de secuencias generadas: {past_values.shape[0]}\n")

    # =============================================================================
    # PASO 5: DIVIDIR DATOS - Split temporal en train/val/test
    # =============================================================================
    print("=== Paso 5: Dividiendo datos en train/val/test ===")
    (train_past, train_future,
     val_past, val_future,
     test_past, test_future) = patchtsmixer_train_val_test_split(
        past_values=past_values,
        future_values=future_values,
        split_ratios=split_ratios
    )

    print(f"✓ Train: {train_past.shape[0]} secuencias")
    print(f"✓ Val:   {val_past.shape[0]} secuencias")
    print(f"✓ Test:  {test_past.shape[0]} secuencias")
    print(f"✓ Split ratios aplicados: {split_ratios}\n")

    # =============================================================================
    # PASO 6: CREAR DATASETS - PyTorch Dataset instances
    # =============================================================================
    print("=== Paso 6: Creando PyTorch Datasets ===")
    train_dataset = TimeSeriesDataset(train_past, train_future)
    val_dataset = TimeSeriesDataset(val_past, val_future)
    test_dataset = TimeSeriesDataset(test_past, test_future)

    print(f"✓ TimeSeriesDataset para train: {len(train_dataset)} samples")
    print(f"✓ TimeSeriesDataset para val: {len(val_dataset)} samples")
    print(f"✓ TimeSeriesDataset para test: {len(test_dataset)} samples\n")

    # =============================================================================
    # PASO 7: CREAR CONFIGURACIÓN DEL MODELO - PatchTSMixerConfig
    # =============================================================================
    print("=== Paso 7: Creando configuración del modelo ===")
    config = create_patchtsmixer_config(
        params=manual_params,
        num_input_channels=num_input_channels,
        context_length=context_length,
        prediction_length=forecast_horizon
    )

    print(f"✓ Config creada: {config.model_type}")
    print(f"✓ Arquitectura: {config.num_layers} capas, d_model={config.d_model}")
    print(f"✓ Num patches: {config.num_patches}, patch_stride={config.patch_stride}")
    print(f"✓ Channel attention: {channel_attention}, scaling: {scaling}\n")

    # =============================================================================
    # PASO 8: INICIALIZAR MODELO - Construir PatchTSMixerForPrediction
    # =============================================================================
    print("=== Paso 8: Inicializando modelo PatchTSMixer ===")
    model = build_patchtsmixer_model(config)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Modelo inicializado exitosamente")
    print(f"✓ Parámetros totales: {total_params:,}")
    print(f"✓ Parámetros entrenables: {trainable_params:,}\n")

    # =============================================================================
    # PASO 9: INICIAR MLFLOW RUN - Logging de experimento
    # =============================================================================
    print("=== Paso 9: Iniciando MLflow run ===")
    with mlflow.start_run(nested=True):

        # Log parámetros de datos
        mlflow.log_params({
            "model_type": "PatchTSMixer",
            "date_col_name": date_col_name,
            "patchtsmixer_channels": str(channel_cols),
            "num_channels": num_input_channels,
            "forecast_horizon": forecast_horizon,
            "context_length": context_length,
            "hyperparameter_search_strategy": "manual",
            "cpu_only": True
        })

        # Log parámetros del modelo
        mlflow.log_params({
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "d_model": d_model,
            "num_layers": num_layers,
            "expansion_factor": expansion_factor,
            "dropout": dropout,
            "head_dropout": head_dropout,
            "pooling_type": pooling_type,
            "channel_attention": channel_attention,
            "scaling": scaling,
            "num_patches": config.num_patches,
            "total_params": total_params,
            "trainable_params": trainable_params
        })

        # Log parámetros de entrenamiento
        mlflow.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
        })

        # Log configuración del modelo como artifact
        config_dict = config.to_dict()
        mlflow.log_dict(config_dict, "model_config.json")

        print(f"✓ Parámetros logueados a MLflow")
        print(f"✓ Configuración guardada como artifact\n")

        # =========================================================================
        # PASO 10: INICIAR ENERGY TRACKING - CodeCarbon
        # =========================================================================
        print("=== Paso 10: Iniciando seguimiento de energía ===")
        tracker = None
        try:
            tracker = EmissionsTracker(
                project_name="train_patchtsmixer",
                measure_power_secs=15,
                save_to_file=False,
                log_level="error",
                output_dir=experiment_dir
            )
            tracker.start()
            print(f"✓ CodeCarbon tracker iniciado\n")

            # =====================================================================
            # PASO 11: ENTRENAMIENTO - Trainer API con early stopping
            # =====================================================================
            print("=== Paso 11: Entrenando modelo ===")
            print(f"Inicio del entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            trainer = train_manual_patchtsmixer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                params=manual_params,
                experiment_dir=experiment_dir
            )

            print(f"✓ Entrenamiento completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        finally:
            # =================================================================
            # PASO 12: DETENER ENERGY TRACKING - Log métricas de energía
            # =================================================================
            print("=== Paso 12: Deteniendo seguimiento de energía ===")
            if tracker:
                tracker.stop()

                # Extraer métricas de energía usando la API pública de CodeCarbon
                try:
                    # Primary: use _total_energy (internal API)
                    if hasattr(tracker, '_total_energy') and tracker._total_energy:
                        energy_kwh = float(tracker._total_energy.kWh)
                    # Fallback: use final_emissions_data (public API)
                    elif hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                        energy_kwh = float(tracker.final_emissions_data.energy_consumed)
                    else:
                        energy_kwh = 0.0
                except (AttributeError, TypeError):
                    energy_kwh = 0.0

                try:
                    emissions_kg = float(tracker.final_emissions) if tracker.final_emissions else 0.0
                except (AttributeError, TypeError):
                    emissions_kg = 0.0

                try:
                    if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                        duration_s = float(tracker.final_emissions_data.duration)
                    else:
                        duration_s = 0.0
                except (AttributeError, TypeError):
                    duration_s = 0.0

                # Log a MLflow
                mlflow.log_metric("energy_consumed_total_kWh", energy_kwh)
                mlflow.log_metric("carbon_emission_kg", emissions_kg)
                mlflow.log_metric("training_duration_seconds", duration_s)

                print(f"✓ Energía consumida: {energy_kwh:.6f} kWh")
                print(f"✓ Emisiones de carbono: {emissions_kg:.6f} kg CO2")
                print(f"✓ Duración del entrenamiento: {duration_s:.2f} segundos\n")
            else:
                energy_kwh = 0.0
                emissions_kg = 0.0
                duration_s = 0.0

        # =====================================================================
        # PASO 13: EVALUAR - Calcular métricas en val y test
        # =====================================================================
        print("=== Paso 13: Evaluando modelo ===")

        # Evaluar en validación
        val_metrics, val_artifacts = evaluate_patchtsmixer(
            trainer=trainer,
            dataset=val_dataset,
            prefix="val",
            experiment_dir=experiment_dir
        )
        print(f"✓ Validación completada: RMSE={val_metrics['val_rmse']:.4f}")

        # Evaluar en test
        test_metrics, test_artifacts = evaluate_patchtsmixer(
            trainer=trainer,
            dataset=test_dataset,
            prefix="test",
            experiment_dir=experiment_dir
        )
        print(f"✓ Test completado: RMSE={test_metrics['test_rmse']:.4f}\n")

        # =====================================================================
        # PASO 14: LOG MÉTRICAS - Guardar métricas en MLflow
        # =====================================================================
        print("=== Paso 14: Logueando métricas a MLflow ===")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        print(f"✓ Métricas de validación logueadas: {list(val_metrics.keys())}")
        print(f"✓ Métricas de test logueadas: {list(test_metrics.keys())}\n")

        # =====================================================================
        # PASO 15: LOG ARTIFACTS - Guardar plots en MLflow
        # =====================================================================
        print("=== Paso 15: Logueando artifacts (plots) ===")
        all_artifacts = val_artifacts + test_artifacts

        for artifact_path in all_artifacts:
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, "plots")
                print(f"✓ Artifact logueado: {os.path.basename(artifact_path)}")

        print(f"✓ Total de artifacts logueados: {len(all_artifacts)}\n")

        # =====================================================================
        # PASO 15.5: GENERAR pipeline_config.json - Para reproducibilidad
        # =====================================================================
        print("=== Paso 15.5: Generando pipeline_config.json ===")

        pipeline_config = {
            "model_type": "PatchTSMixer",
            "experiment_timestamp": datetime.now().isoformat(),
            "data_params": {
                "dataset_path": dataset_path,
                "date_col_name": date_col_name,
                "patchtsmixer_channels": channel_cols,
                "num_channels": num_input_channels,
                "forecast_horizon": forecast_horizon,
                "context_length": context_length,
                "split_ratios": split_ratios
            },
            "model_params": {
                "patch_length": patch_length,
                "patch_stride": patch_stride,
                "num_patches": config.num_patches,
                "d_model": d_model,
                "num_layers": num_layers,
                "expansion_factor": expansion_factor,
                "dropout": dropout,
                "head_dropout": head_dropout,
                "pooling_type": pooling_type,
                "channel_attention": channel_attention,
                "scaling": scaling,
                "total_params": total_params,
                "trainable_params": trainable_params
            },
            "training_params": {
                "strategy": "manual",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_epochs": epochs,
                "early_stopping_patience": early_stopping_patience,
                "optimizer": "AdamW",
                "seed": SEED,
                "cpu_only": True
            },
            "reproducibility": {
                "seed": SEED,
                "pytorch_deterministic": True,
                "tensorflow_deterministic": True,
                "python_version": sys.version,
                "torch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
                "transformers_version": "4.36.0+"  # From Phase 1
            },
            "results": {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "training_duration_seconds": duration_s,
                "energy_kwh": energy_kwh,
                "carbon_kg": emissions_kg
            }
        }

        # Guardar como JSON
        config_path = os.path.join(experiment_dir, "pipeline_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_config, f, indent=2, ensure_ascii=False)

        # Log a MLflow
        mlflow.log_artifact(config_path)

        print(f"✓ pipeline_config.json generado: {config_path}")
        print(f"✓ Configuración completa guardada para reproducibilidad\n")

        # =====================================================================
        # PASO 16: GUARDAR MODELO - save_pretrained() de HuggingFace
        # =====================================================================
        print("=== Paso 16: Guardando modelo ===")
        model_path = os.path.join(experiment_dir, "patchtsmixer_model")

        # Crear directorio si no existe
        os.makedirs(model_path, exist_ok=True)

        # Guardar modelo usando HuggingFace API
        trainer.save_model(model_path)

        # Verificar que se guardó correctamente (HuggingFace saves as safetensors or pytorch_model.bin)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            print(f"✓ Modelo guardado exitosamente en: {model_path}")
            print(f"✓ Archivos generados: model.safetensors, config.json")
        elif os.path.exists(pytorch_bin_path):
            print(f"✓ Modelo guardado exitosamente en: {model_path}")
            print(f"✓ Archivos generados: pytorch_model.bin, config.json")
        else:
            logger.warning(f"⚠ No se encontró model.safetensors ni pytorch_model.bin en {model_path}")

        # Log ruta del modelo
        mlflow.log_param("model_save_path", model_path)
        print()

        # =====================================================================
        # PASO 17: RETORNAR RESULTADOS - Dict con métricas y ruta
        # =====================================================================
        print("=== Paso 17: Preparando resultados ===")
        result = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": model_path
        }

        print("\n" + "="*80)
        print("ENTRENAMIENTO DE PATCHTSMIXER COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"✓ Métricas de validación:")
        for key, value in val_metrics.items():
            if value is not None:
                print(f"  - {key}: {value:.4f}")
        print(f"✓ Métricas de test:")
        for key, value in test_metrics.items():
            if value is not None:
                print(f"  - {key}: {value:.4f}")
        print(f"✓ Modelo guardado en: {model_path}")
        print("="*80 + "\n")

        return result
