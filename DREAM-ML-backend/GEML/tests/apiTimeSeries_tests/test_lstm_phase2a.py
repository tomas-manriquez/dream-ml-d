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
Test suite for LSTM Phase 2A: Grid Search Hyperparameter Optimization

Tests include:
- Grid search with small grid (2×2 grid = 4 combinations)
- Default parameter validation (8 combinations)
- Best model selection (lowest val_loss)
- Warning threshold validation
- Memory profiling (conditional)
- Memory profiling metrics in MLflow
- Memory cleanup (memory increase < 500MB)
- Progress logging (every 10 iterations)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import logging
from unittest.mock import patch, MagicMock
from sklearn.model_selection import ParameterGrid

# Import functions from train.py
from apiTimeSeries.train import (
    create_sequences_for_lstm,
    lstm_train_val_test_split,
    build_lstm_model,
    train_lstm_model
)


@pytest.fixture(autouse=True)
def configure_test_logging():
    """
    Configure logger for test capture.
    Override settings.py propagate=False to allow caplog to capture messages.
    """
    # Get the apiTimeSeries logger
    api_logger = logging.getLogger('apiTimeSeries')

    # Store original propagate value
    original_propagate = api_logger.propagate

    # Enable propagation for tests
    api_logger.propagate = True
    api_logger.setLevel(logging.INFO)

    yield

    # Restore original setting after test
    api_logger.propagate = original_propagate


@pytest.fixture
def synthetic_lstm_dataset():
    """Generate synthetic time series dataset for testing"""
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Simple sine wave with noise
    values = np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    df.set_index('date', inplace=True)

    # Create temporary CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


@pytest.fixture
def experiment_dir():
    """Create temporary experiment directory"""
    temp_dir = tempfile.mkdtemp(prefix="lstm_test_phase2a_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_grid_search_small_grid(synthetic_lstm_dataset, experiment_dir):
    """Test grid search with small 2×2 grid (4 combinations)"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_grid_small",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 5,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32], [64]],
            "dropout_rate_options": [0.2, 0.3],
            "recurrent_dropout_rate_options": [0.2],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [10]  # Small for fast testing
        },
        "enable_memory_profiling": False,
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"
        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "best_params" in result
        assert result["best_params"]["lstm_units"] in [[32], [64]]
        assert result["best_params"]["dropout_rate"] in [0.2, 0.3]


def test_grid_search_default_params(synthetic_lstm_dataset, experiment_dir):
    """Test grid search uses conservative defaults (8 combinations)"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_grid_defaults",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 5,
        "hyperparameter_search_strategy": "grid",
        # No grid_search_params provided - should use defaults
        "enable_memory_profiling": False,
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Verify grid_iterations_total was logged (should be 8 with defaults)
        metric_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
        assert "grid_iterations_total" in metric_calls
        assert metric_calls["grid_iterations_total"] == 8


def test_grid_search_best_model_selection(synthetic_lstm_dataset, experiment_dir):
    """Test that grid search selects model with lowest val_loss"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_best_selection",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32]],
            "dropout_rate_options": [0.1, 0.2],
            "recurrent_dropout_rate_options": [0.1],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [5]  # Very small for fast testing
        },
        "enable_memory_profiling": False,
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Verify best_iteration was logged (should be between 1 and 2)
        metric_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
        assert "best_iteration" in metric_calls
        assert 1 <= metric_calls["best_iteration"] <= 2

@pytest.mark.slow
def test_grid_search_warning_threshold(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test that warning is logged when combinations exceed threshold"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_warning",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32], [64], [128]],
            "dropout_rate_options": [0.1, 0.2, 0.3],
            "recurrent_dropout_rate_options": [0.1, 0.2],
            "learning_rate_options": [0.001, 0.01],
            "batch_size_options": [16, 32],
            "epochs_options": [10]  # Total: 3×3×2×2×2×1 = 72 combinations
        },
        "enable_memory_profiling": False,
        "grid_warning_threshold": 50  # Should trigger warning
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        with caplog.at_level(logging.WARNING, logger='apiTimeSeries'):
            result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Check that warning was logged
        warning_messages = [r.message for r in caplog.records if r.levelname == 'WARNING']
        assert any("excede el umbral" in msg for msg in warning_messages), \
            f"Expected warning not found. Captured warnings: {warning_messages}"


def test_memory_profiling_conditional(synthetic_lstm_dataset, experiment_dir):
    """Test that memory profiling only runs when enabled"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_memory_disabled",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32]],
            "dropout_rate_options": [0.2],
            "recurrent_dropout_rate_options": [0.2],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [5]
        },
        "enable_memory_profiling": False,  # Disabled
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Verify memory metrics were NOT logged
        metric_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
        assert "memory_usage_mb" not in metric_calls
        assert "memory_increase_mb" not in metric_calls


def test_memory_profiling_metrics_logged(synthetic_lstm_dataset, experiment_dir):
    """Test that memory metrics are logged when profiling is enabled"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_memory_enabled",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32]],
            "dropout_rate_options": [0.2],
            "recurrent_dropout_rate_options": [0.2],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [5]
        },
        "enable_memory_profiling": True,  # Enabled
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Verify memory metrics WERE logged
        metric_calls = {call[0][0]: call[0][1] for call in mock_log_metric.call_args_list}
        assert "memory_usage_mb" in metric_calls
        assert "memory_increase_mb" in metric_calls
        assert metric_calls["memory_usage_mb"] > 0
        assert metric_calls["memory_increase_mb"] >= 0


def test_grid_search_memory_cleanup(synthetic_lstm_dataset, experiment_dir):
    """Test memory doesn't leak during grid search (increase < 500MB)"""
    import psutil
    import os

    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_memory_cleanup",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32], [64]],
            "dropout_rate_options": [0.2, 0.3],
            "recurrent_dropout_rate_options": [0.2],
            "learning_rate_options": [0.001, 0.01],
            "batch_size_options": [16, 32],
            "epochs_options": [5]  # Total: 2×2×1×2×2×1 = 16 combinations
        },
        "enable_memory_profiling": True,
        "grid_warning_threshold": 50
    }

    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / 1024 / 1024

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

    final_memory_mb = process.memory_info().rss / 1024 / 1024
    memory_increase_mb = final_memory_mb - initial_memory_mb

    print(f"Memory increase: {memory_increase_mb:.1f} MB")

    # Should be < 500MB increase
    assert memory_increase_mb < 500, f"Memory leak detected: {memory_increase_mb:.1f}MB increase exceeds 500MB threshold"
    assert result["status"] == "success"

@pytest.mark.slow
def test_grid_search_progress_logging(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test that progress is logged every 10 iterations"""
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_progress",
        "forecast_horizon": 1,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32], [64]],
            "dropout_rate_options": [0.1, 0.2, 0.3],
            "recurrent_dropout_rate_options": [0.1, 0.2],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [5]  # Total: 2×3×2×1×1×1 = 12 combinations
        },
        "enable_memory_profiling": False,
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        with caplog.at_level(logging.INFO, logger='apiTimeSeries'):
            result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Check that progress logging occurred at iteration 10
        progress_messages = [r.message for r in caplog.records
                            if "Grid Search Progress" in r.message]
        assert len(progress_messages) >= 1, \
            f"No progress messages found. Total INFO records: {len([r for r in caplog.records if r.levelname == 'INFO'])}"
        assert any("10/12" in msg for msg in progress_messages), \
            f"Expected '10/12' in progress messages. Found: {progress_messages}"


def test_grid_search_actually_logs_messages(synthetic_lstm_dataset, experiment_dir, caplog):
    """
    Validate that the implementation actually logs messages
    (not just that tests can capture them)
    """
    data = {
        "date_col_name": "date",
        "target_variable": "value",
        "input_features": ["value"],
        "model_name": "test_logging_validation",
        "sequence_length": 5,
        "early_stopping_patience": 3,
        "hyperparameter_search_strategy": "grid",
        "grid_search_params": {
            "lstm_units_options": [[32]],
            "dropout_rate_options": [0.2],
            "recurrent_dropout_rate_options": [0.2],
            "learning_rate_options": [0.001],
            "batch_size_options": [32],
            "epochs_options": [5]
        },
        "grid_warning_threshold": 50
    }

    with patch('mlflow.start_run'), \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.keras.log_model'):

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run

        with caplog.at_level(logging.INFO, logger='apiTimeSeries'):
            result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        assert result["status"] == "success"

        # Verify key log messages are present
        all_messages = [r.message for r in caplog.records]
        assert any("Iniciando Grid Search" in msg for msg in all_messages), \
            "Grid Search start message not logged"
        assert any("Grid Search: 1 combinaciones a evaluar" in msg for msg in all_messages), \
            "Grid combination count not logged"
        assert any("Grid Search completado" in msg for msg in all_messages), \
            "Grid Search completion message not logged"
