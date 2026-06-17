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
Test suite for LSTM Phase 2B: Random Search Hyperparameter Optimization

Tests include:
- Random parameter generation with type validation
- Random search with small iterations (5 iterations)
- Custom parameter ranges override defaults
- Best model selection (lowest val_loss)
- Warning threshold validation (> 200 iterations)
- Memory profiling (conditional, enabled/disabled)
- Memory cleanup (memory increase < 500MB)
- Progress logging (every 10 iterations)
- MLflow metrics logging
- Python native types (not numpy)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import logging
from unittest.mock import patch, MagicMock

# Import functions from train.py
from apiTimeSeries.train import (
    generate_random_lstm_params,
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
    temp_dir = tempfile.mkdtemp(prefix="lstm_test_phase2b_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_generate_random_lstm_params():
    """Test random parameter generation with default ranges"""
    search_params = {}  # Use all defaults

    params = generate_random_lstm_params(search_params)

    # Verify all keys present
    assert "lstm_units" in params
    assert "dropout_rate" in params
    assert "recurrent_dropout_rate" in params
    assert "learning_rate" in params
    assert "batch_size" in params
    assert "epochs" in params

    # Verify types (Python natives, not numpy)
    assert isinstance(params["dropout_rate"], float)
    assert isinstance(params["recurrent_dropout_rate"], float)
    assert isinstance(params["learning_rate"], float)
    assert isinstance(params["batch_size"], int)
    assert isinstance(params["epochs"], int)
    assert isinstance(params["lstm_units"], list)

    # Verify ranges (using defaults: dropout [0.1, 0.3], lr [0.0001, 0.01], etc.)
    assert 0.1 <= params["dropout_rate"] <= 0.3
    assert 0.1 <= params["recurrent_dropout_rate"] <= 0.3
    assert 0.0001 <= params["learning_rate"] <= 0.01
    assert params["batch_size"] in [16, 32, 64]
    assert 50 <= params["epochs"] <= 100
    assert params["lstm_units"] in [[32], [64], [128], [64, 32]]


def test_generate_random_lstm_params_with_user_ranges():
    """Test random parameter generation with custom user ranges"""
    search_params = {
        "lstm_units_options": [[32], [64]],
        "dropout_rate_range": [0.0, 0.2],
        "recurrent_dropout_rate_range": [0.0, 0.2],
        "learning_rate_range": [0.001, 0.005],
        "batch_size_options": [16, 32],
        "epochs_range": [20, 50]
    }

    params = generate_random_lstm_params(search_params)

    # Verify custom ranges are respected
    assert params["lstm_units"] in [[32], [64]]
    assert 0.0 <= params["dropout_rate"] <= 0.2
    assert 0.0 <= params["recurrent_dropout_rate"] <= 0.2
    assert 0.001 <= params["learning_rate"] <= 0.005
    assert params["batch_size"] in [16, 32]
    assert 20 <= params["epochs"] <= 50


def test_random_search_small_iterations(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test random search with 5 iterations"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow:
        # Mock MLflow to prevent actual logging
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_small",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 5,
            "random_search_params": {
                "lstm_units_options": [[32], [64]],
                "dropout_rate_range": [0.1, 0.2],
                "recurrent_dropout_rate_range": [0.1, 0.2],
                "learning_rate_range": [0.001, 0.01],
                "batch_size_options": [16, 32],
                "epochs_range": [5, 10]  # Small for fast test
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify completion
        assert result["status"] == "success"
        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "best_params" in result

        # Verify MLflow logging
        assert mock_mlflow.log_metric.called
        assert mock_mlflow.log_params.called

        # Verify random search logs
        assert "Iniciando Random Search" in caplog.text
        assert "Random Search completado" in caplog.text


def test_random_search_best_model_selection(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test that best model is selected based on lowest val_loss"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_best",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 3,
            "random_search_params": {
                "lstm_units_options": [[32]],
                "dropout_rate_range": [0.1, 0.2],
                "recurrent_dropout_rate_range": [0.1, 0.2],
                "learning_rate_range": [0.001, 0.01],
                "batch_size_options": [16],
                "epochs_range": [5, 10]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify best model was selected
        assert result["status"] == "success"
        assert "best_params" in result

        # Verify "Nuevo mejor modelo encontrado" appears in logs
        assert "Nuevo mejor modelo encontrado" in caplog.text


def test_random_search_parameter_types_native(synthetic_lstm_dataset, experiment_dir):
    """Test that random search parameters are Python native types (not numpy)"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Capture log_params calls
        log_params_calls = []
        def capture_log_params(params):
            log_params_calls.append(params)
        mock_mlflow.log_params.side_effect = capture_log_params

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_types",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 2,
            "random_search_params": {
                "epochs_range": [5, 8]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify all logged params are Python natives
        for call_params in log_params_calls:
            for key, value in call_params.items():
                # Check that values are not numpy types
                assert not isinstance(value, np.generic), f"Parameter {key} is numpy type: {type(value)}"


def test_random_search_warning_threshold(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test that warning is logged when n_random_iterations > 200"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_warning",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 2,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 250,  # > 200, should trigger warning
            "random_search_params": {
                "lstm_units_options": [[32]],
                "epochs_range": [3, 5]
            }
        }

        # Patch the training loop to exit after warning is logged
        original_generate = generate_random_lstm_params
        call_count = [0]

        def limited_generate(params):
            call_count[0] += 1
            if call_count[0] > 2:  # Only run 2 iterations
                raise StopIteration("Test early exit")
            return original_generate(params)

        with patch('apiTimeSeries.train.generate_random_lstm_params', side_effect=limited_generate):
            try:
                result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)
            except RuntimeError as e:
                # Expected - training will fail after 2 iterations
                # But we should have seen the warning
                pass

        # Verify warning was logged BEFORE the training loop started
        assert "n_random_iterations es muy alto (250)" in caplog.text
        assert "Considere usar un valor menor (<200)" in caplog.text


def test_random_search_memory_profiling_disabled(synthetic_lstm_dataset, experiment_dir):
    """Test random search with memory profiling DISABLED (default)"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow, \
         patch('apiTimeSeries.train.EmissionsTracker') as mock_tracker:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Mock EmissionsTracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance

        log_metric_calls = []
        def capture_log_metric(key, value):
            log_metric_calls.append((key, value))
        mock_mlflow.log_metric.side_effect = capture_log_metric

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_no_profiling",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 2,
            "enable_memory_profiling": False,  # Explicitly disabled
            "random_search_params": {
                "epochs_range": [5, 8]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify memory metrics were NOT logged
        metric_keys = [key for key, _ in log_metric_calls]
        assert "memory_usage_mb" not in metric_keys
        assert "memory_increase_mb" not in metric_keys


def test_random_search_memory_profiling_enabled(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test random search with memory profiling ENABLED"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow, \
         patch('apiTimeSeries.train.EmissionsTracker') as mock_tracker:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Mock EmissionsTracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance

        log_metric_calls = []
        def capture_log_metric(key, value):
            log_metric_calls.append((key, value))
        mock_mlflow.log_metric.side_effect = capture_log_metric

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_with_profiling",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 2,
            "enable_memory_profiling": True,  # Explicitly enabled
            "random_search_params": {
                "epochs_range": [5, 8]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify memory metrics WERE logged
        metric_keys = [key for key, _ in log_metric_calls]
        assert "memory_usage_mb" in metric_keys
        assert "memory_increase_mb" in metric_keys

        # Verify memory profiling logs
        assert "Memory profiling enabled" in caplog.text
        assert "Memory profiling results" in caplog.text


@pytest.mark.slow
def test_random_search_memory_cleanup(synthetic_lstm_dataset, experiment_dir):
    #Test that memory cleanup prevents leaks during random search.
    #Memory increase should be < 500MB for 20 iterations.

    import psutil

    with patch('apiTimeSeries.train.mlflow') as mock_mlflow, \
         patch('apiTimeSeries.train.EmissionsTracker') as mock_tracker:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Mock EmissionsTracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance

        # Measure memory before
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_memory",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 3,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 20,  # More iterations to stress test memory
            "random_search_params": {
                "lstm_units_options": [[32]],
                "epochs_range": [3, 5]  # Keep epochs low for speed
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Measure memory after
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb

        print(f"\nMemory test results:")
        print(f"  Initial: {initial_memory_mb:.1f} MB")
        print(f"  Final: {final_memory_mb:.1f} MB")
        print(f"  Increase: {memory_increase_mb:.1f} MB")

        # Verify memory increase is below threshold
        assert memory_increase_mb < 500, \
            f"Memory leak detected: {memory_increase_mb:.1f}MB increase exceeds 500MB threshold"

@pytest.mark.slow
def test_random_search_progress_logging(synthetic_lstm_dataset, experiment_dir, caplog):
    """Test that progress is logged every 10 iterations"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow, \
         patch('apiTimeSeries.train.EmissionsTracker') as mock_tracker:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Mock EmissionsTracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_progress",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 3,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 15,  # Will trigger progress at iteration 10
            "random_search_params": {
                "lstm_units_options": [[32]],
                "epochs_range": [3, 5]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify progress log at iteration 10
        assert "Random Search Progress: 10/15 iterations completed" in caplog.text


def test_random_search_mlflow_metrics(synthetic_lstm_dataset, experiment_dir):
    """Test that all required MLflow metrics are logged for random search"""
    with patch('apiTimeSeries.train.mlflow') as mock_mlflow, \
         patch('apiTimeSeries.train.EmissionsTracker') as mock_tracker:
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run
        mock_mlflow.start_run.return_value.__enter__ = lambda self: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda self, *args: None

        # Mock EmissionsTracker
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance

        log_metric_calls = []
        def capture_log_metric(key, value):
            log_metric_calls.append((key, value))
        mock_mlflow.log_metric.side_effect = capture_log_metric

        data = {
            "date_col_name": "date",
            "target_variable": "value",
            "input_features": ["value"],
            "model_name": "test_random_mlflow",
            "forecast_horizon": 1,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "sequence_length": 5,
            "early_stopping_patience": 5,
            "hyperparameter_search_strategy": "random",
            "n_random_iterations": 3,
            "random_search_params": {
                "epochs_range": [5, 8]
            }
        }

        result = train_lstm_model(synthetic_lstm_dataset, data, experiment_dir)

        # Verify required metrics were logged
        metric_keys = [key for key, _ in log_metric_calls]
        assert "best_val_loss" in metric_keys
        assert "random_iterations_total" in metric_keys
        assert "best_iteration" in metric_keys

        # Verify metric values
        best_val_loss_value = next(val for key, val in log_metric_calls if key == "best_val_loss")
        assert isinstance(best_val_loss_value, float)

        random_iterations_value = next(val for key, val in log_metric_calls if key == "random_iterations_total")
        assert random_iterations_value == 3

        best_iteration_value = next(val for key, val in log_metric_calls if key == "best_iteration")
        assert 1 <= best_iteration_value <= 3  # Should be within range
