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

"""
Test suite for PatchTSMixer Full Pipeline - Phase 8

Tests cover:
- End-to-end full pipeline validation (large dataset)
- Model inference after training
- Memory cleanup validation
- Performance vs naive baseline
- Pipeline config schema completeness for reproducibility

NOTE: Tests in this file complement (not duplicate) tests from:
- test_patchtsmixer_training.py (manual training, reproducibility, plots, errors)
- test_patchtsmixer_integration.py (service routing, MLflow/DVC integration)
"""

import pytest
import numpy as np
import pandas as pd
import os
import json
import gc
import logging
from unittest.mock import patch

import torch
from transformers import PatchTSMixerForPrediction

from apiTimeSeries.train import train_patchtsmixer_model


# ======================================================================================
# LOGGING CONFIGURATION (Following LSTM pattern)
# ======================================================================================

@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logger for test capture."""
    api_logger = logging.getLogger('apiTimeSeries')
    original_propagate = api_logger.propagate
    api_logger.propagate = True
    api_logger.setLevel(logging.INFO)
    yield
    api_logger.propagate = original_propagate


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================

def create_large_synthetic_dataset(tmp_path, n_rows=2000, n_channels=3, seed=42):
    """
    Generate large synthetic multivariate time series dataset for full pipeline tests.

    Includes realistic temporal patterns:
    - Linear trend
    - Weekly seasonality (168 hours)
    - Random noise

    Args:
        tmp_path: pytest tmp_path fixture
        n_rows: Number of timesteps (default 2000 for full pipeline test)
        n_channels: Number of variables/channels
        seed: Random seed for reproducibility

    Returns:
        Path to the created CSV file
    """
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')

    data = {'date': dates}

    for i in range(n_channels):
        # Create realistic temporal patterns
        trend = np.linspace(10, 20, n_rows)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_rows) / 168)  # Weekly seasonality
        noise = np.random.normal(0, 1, n_rows)
        data[f'channel_{i}'] = trend + seasonal + noise

    df = pd.DataFrame(data)
    csv_path = os.path.join(tmp_path, 'large_synthetic_data.csv')
    df.to_csv(csv_path, index=False)

    return csv_path


def get_full_pipeline_config(n_channels=3):
    """
    Return full pipeline test configuration.

    Uses larger model parameters than unit tests but still fast enough for CI.
    """
    channels = [f'channel_{i}' for i in range(n_channels)]

    return {
        "date_col_name": "date",
        "patchtsmixer_channels": channels,
        "forecast_horizon": 96,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 512,
            "patch_length": 8,
            "d_model": 16,
            "num_layers": 4,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 3,
            "early_stopping_patience": 2,
        }
    }


# ======================================================================================
# FULL PIPELINE TESTS
# ======================================================================================

class TestFullPipeline:
    """End-to-end full pipeline tests with larger datasets."""

    def test_full_pipeline_large_dataset(self, tmp_path):
        """
        Test complete training pipeline with large multivariate dataset (2000 rows).

        This is the primary full pipeline test that validates:
        - Training completes successfully
        - All metrics are valid
        - Model files are saved correctly
        - Pipeline config is generated
        """
        # Arrange
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=2000, n_channels=3)
        experiment_dir = os.path.join(tmp_path, "full_pipeline_exp")
        data = get_full_pipeline_config(n_channels=3)

        # Act
        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Assert - Result structure
        assert isinstance(result, dict)
        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "model_path" in result

        # Assert - All metrics are finite
        for split in ["val_metrics", "test_metrics"]:
            for metric_name, value in result[split].items():
                if value is not None:  # MAPE can be None
                    assert np.isfinite(value), f"{split}.{metric_name} is not finite: {value}"

        # Assert - Model files exist
        model_path = result["model_path"]
        assert os.path.exists(model_path)

        model_weights_exists = (
            os.path.exists(os.path.join(model_path, "model.safetensors")) or
            os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
        )
        assert model_weights_exists, f"Model weights not found in {model_path}"
        assert os.path.exists(os.path.join(model_path, "config.json"))

        # Assert - Pipeline config exists
        config_path = os.path.join(experiment_dir, "pipeline_config.json")
        assert os.path.exists(config_path)


# ======================================================================================
# MODEL INFERENCE TESTS
# ======================================================================================

class TestModelInference:
    """Tests for loading trained model and running inference."""

    def test_model_inference_after_training(self, tmp_path):
        """
        Test that trained model can be loaded and used for inference.

        Validates:
        - Model loads successfully via from_pretrained()
        - Inference produces output of correct shape
        - Output values are finite (no NaN/Inf)
        """
        # Arrange - Train a small model
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=500, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "inference_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 8,
                "num_layers": 2,
                "dropout": 0.1,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 2,
                "early_stopping_patience": 1
            }
        }

        # Train the model
        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Act - Load and run inference
        model_path = result["model_path"]
        model = PatchTSMixerForPrediction.from_pretrained(model_path)
        model.eval()

        # Create sample input matching training config
        n_channels = len(data["patchtsmixer_channels"])
        context_length = data["manual_params"]["context_length"]
        sample_input = torch.randn(1, context_length, n_channels)

        with torch.no_grad():
            output = model(past_values=sample_input)

        # Assert - Output shape is correct
        prediction_length = data["forecast_horizon"]
        expected_shape = (1, prediction_length, n_channels)
        assert output.prediction_outputs.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.prediction_outputs.shape}"

        # Assert - Output values are finite
        assert torch.isfinite(output.prediction_outputs).all(), \
            "Inference produced NaN/Inf values"

    def test_batch_inference(self, tmp_path):
        """
        Test inference with batch size > 1.

        Validates model handles batched inputs correctly.
        """
        # Arrange - Train a small model
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=400, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "batch_inference_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 8,
                "num_layers": 2,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 1,
                "early_stopping_patience": 1
            }
        }

        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Act - Load and run batched inference
        model = PatchTSMixerForPrediction.from_pretrained(result["model_path"])
        model.eval()

        batch_size = 4
        n_channels = len(data["patchtsmixer_channels"])
        context_length = data["manual_params"]["context_length"]
        batch_input = torch.randn(batch_size, context_length, n_channels)

        with torch.no_grad():
            output = model(past_values=batch_input)

        # Assert
        expected_shape = (batch_size, data["forecast_horizon"], n_channels)
        assert output.prediction_outputs.shape == expected_shape


# ======================================================================================
# MEMORY TESTS
# ======================================================================================

class TestMemoryManagement:
    """Tests for memory cleanup and leak detection."""

    @pytest.mark.skipif(
        not hasattr(os, 'getpid'),
        reason="psutil/memory profiling not available"
    )
    def test_training_memory_cleanup(self, tmp_path):
        """
        Test that memory doesn't leak during training.

        Following LSTM pattern: memory increase should be < 500MB.
        """
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        # Arrange
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=500, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "memory_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 8,
                "num_layers": 2,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 2,
                "early_stopping_patience": 1
            }
        }

        # Force garbage collection before measuring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        # Act - Run training
        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Force garbage collection after training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb

        # Assert - Memory increase should be reasonable
        assert memory_increase_mb < 500, \
            f"Memory leak detected: {memory_increase_mb:.1f}MB increase"


# ======================================================================================
# PERFORMANCE BASELINE TESTS
# ======================================================================================

class TestPerformanceBaseline:
    """Tests that validate model performance against naive baselines."""

    def test_performance_vs_naive_baseline(self, tmp_path):
        """
        Test that PatchTSMixer beats naive forecast baseline.

        Naive baseline: last observed value repeated for all horizons.
        """
        # Arrange - Create dataset with clear pattern
        np.random.seed(42)
        n_rows = 500
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')

        # Create predictable pattern: trend + seasonality
        trend = np.linspace(10, 30, n_rows)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_rows) / 24)  # Daily seasonality
        noise = np.random.normal(0, 0.5, n_rows)  # Small noise
        values = trend + seasonal + noise

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        csv_path = os.path.join(tmp_path, 'predictable_data.csv')
        df.to_csv(csv_path, index=False)

        experiment_dir = os.path.join(tmp_path, "baseline_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["value"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 16,
                "num_layers": 4,
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 5,  # More epochs for better learning
                "early_stopping_patience": 3
            }
        }

        # Act - Train model
        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Get model's RMSE
        model_rmse = result["test_metrics"]["test_rmse"]

        # Calculate naive baseline RMSE (using last value persistence)
        # Split data same way as training
        n_test = int(n_rows * 0.15)
        test_start = n_rows - n_test

        # Naive forecast: repeat last known value
        naive_predictions = np.tile(values[test_start - 1], (n_test - 24, 24))
        actual = np.array([values[i:i+24] for i in range(test_start, n_rows - 24)])

        naive_rmse = np.sqrt(np.mean((naive_predictions - actual) ** 2))

        # Assert - Model should beat naive baseline
        assert model_rmse < naive_rmse, \
            f"Model RMSE ({model_rmse:.4f}) should be less than naive baseline ({naive_rmse:.4f})"


# ======================================================================================
# PIPELINE CONFIG SCHEMA TESTS
# ======================================================================================

class TestPipelineConfigSchema:
    """Tests for pipeline_config.json schema completeness for reproducibility."""

    def test_pipeline_config_has_reproducibility_fields(self, tmp_path):
        """
        Test that pipeline_config.json contains all fields required for reproducibility.

        Required fields:
        - reproducibility.seed
        - reproducibility.torch_version
        - reproducibility.transformers_version
        - reproducibility.python_version
        - reproducibility.device
        """
        # Arrange
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "schema_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 8,
                "num_layers": 2,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 1,
                "early_stopping_patience": 1
            }
        }

        # Act
        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        config_path = os.path.join(experiment_dir, "pipeline_config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Assert - Required sections exist
        assert "reproducibility" in config, "Missing 'reproducibility' section"

        repro = config["reproducibility"]
        required_repro_keys = ["seed", "torch_version", "transformers_version", "python_version"]
        for key in required_repro_keys:
            assert key in repro, f"Missing reproducibility key: {key}"

        # Assert - Seed is the expected value (42)
        assert repro["seed"] == 42, f"Expected seed=42, got {repro['seed']}"

    def test_pipeline_config_model_params_complete(self, tmp_path):
        """
        Test that pipeline_config.json contains all model parameters.
        """
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "model_params_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 16,
                "num_layers": 4,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 1,
                "early_stopping_patience": 1
            }
        }

        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        config_path = os.path.join(experiment_dir, "pipeline_config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Assert - model_params contains all required keys
        assert "model_params" in config
        model_params = config["model_params"]

        required_model_keys = [
            "context_length", "patch_length", "d_model",
            "num_layers", "prediction_length"
        ]
        for key in required_model_keys:
            assert key in model_params, f"Missing model_params key: {key}"

        # Verify values match input
        assert model_params["d_model"] == 16
        assert model_params["num_layers"] == 4
        assert model_params["patch_length"] == 8


# ======================================================================================
# NATIVE TYPES VALIDATION (Following LSTM pattern)
# ======================================================================================

class TestNativeTypes:
    """Tests that validate metrics are native Python types (not numpy)."""

    def test_metrics_are_native_python_types(self, tmp_path):
        """
        Test that all metrics are Python native types, not numpy types.

        This is important for JSON serialization and MLflow logging.
        """
        csv_path = create_large_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)
        experiment_dir = os.path.join(tmp_path, "types_exp")

        data = {
            "date_col_name": "date",
            "patchtsmixer_channels": ["channel_0", "channel_1"],
            "forecast_horizon": 24,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "manual_params": {
                "context_length": 96,
                "patch_length": 8,
                "d_model": 8,
                "num_layers": 2,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 1,
                "early_stopping_patience": 1
            }
        }

        result = train_patchtsmixer_model(csv_path, data, experiment_dir)

        # Check all metrics are native types
        for split in ["val_metrics", "test_metrics"]:
            for key, value in result[split].items():
                if value is not None:
                    assert not isinstance(value, np.generic), \
                        f"{split}.{key} is numpy type: {type(value)}, should be native Python"
                    assert isinstance(value, (int, float)), \
                        f"{split}.{key} is {type(value)}, expected int or float"
