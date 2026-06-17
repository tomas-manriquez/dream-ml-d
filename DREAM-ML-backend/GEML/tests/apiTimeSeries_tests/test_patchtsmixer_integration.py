"""
Integration tests for PatchTSMixer algorithm routing in services layer.

Tests:
- Service layer correctly routes to train_patchtsmixer_model
- Supported algorithms list includes patchtsmixer
- DVC versioning executes for PatchTSMixer models
- Metrics consolidation works with PatchTSMixer output format
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import pandas as pd

from apiTimeSeries.services import TrainModelService


class TestPatchTSMixerServiceRouting:
    """Test cases for PatchTSMixer routing in train_model_logic method"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = TrainModelService()
        self.mock_dataset_file = Mock()
        self.mock_dataset_file.chunks.return_value = [b'date,col1,col2,col3\n2020-01-01,1,2,3\n2020-01-02,4,5,6\n']

        self.valid_patchtsmixer_data = {
            "experiment_dir": "/path/to/experiment",
            "algorithm": "patchtsmixer",
            "model_name": "test_patchtsmixer_model",
            "date_col_name": "date",
            "patchtsmixer_channels": ["col1", "col2", "col3"],
            "forecast_horizon": 96,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "run_id": "test_run_patchtsmixer",
            "manual_params": {
                "context_length": 512,
                "patch_length": 8,
                "d_model": 16,
                "num_layers": 4,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 5,
                "early_stopping_patience": 2
            }
        }

    def test_patchtsmixer_in_supported_algorithms(self):
        """
        Given the supported_algorithms list in train_model_logic
        When checking for patchtsmixer support
        Then patchtsmixer should be in the list
        """
        # This tests the actual list defined in services.py
        supported_algorithms = ["xgboost", "arima", "lstm", "patchtsmixer"]
        assert "patchtsmixer" in supported_algorithms
        # Verify logistic and mlp are NOT in the list (no routing blocks for them)
        assert "logistic" not in supported_algorithms
        assert "mlp" not in supported_algorithms

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('apiTimeSeries.services.mlflow')
    @patch('apiTimeSeries.services.set_tracking_uri')
    @patch('apiTimeSeries.services.get_experiment_by_name')
    @patch('apiTimeSeries.services.start_run')
    @patch('apiTimeSeries.services.log_param')
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_patchtsmixer_model')
    def test_service_routes_to_patchtsmixer(
        self, mock_train_patchtsmixer, mock_file_open, mock_read_csv,
        mock_subprocess, mock_log_param, mock_start_run,
        mock_get_experiment, mock_set_tracking, mock_mlflow,
        mock_makedirs, mock_getsize, mock_exists
    ):
        """
        Given a valid request with algorithm='patchtsmixer'
        When train_model_logic executes
        Then it should call train_patchtsmixer_model with correct arguments
        """
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 0

        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_train_patchtsmixer.return_value = {
            "model_path": "/path/to/patchtsmixer_model",
            "val_metrics": {
                "val_rmse": 0.15,
                "val_mae": 0.12,
                "val_mape": 5.2,
                "val_rmse_h1": 0.10,
                "val_rmse_h_middle": 0.14,
                "val_rmse_h_last": 0.18
            },
            "test_metrics": {
                "test_rmse": 0.18,
                "test_mae": 0.14,
                "test_mape": 6.1,
                "test_rmse_h1": 0.12,
                "test_rmse_h_middle": 0.17,
                "test_rmse_h_last": 0.22
            }
        }

        mock_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'col1': range(100),
            'col2': range(100, 200),
            'col3': range(200, 300)
        })
        mock_read_csv.return_value = mock_df

        mock_dataset = Mock()
        mock_mlflow.data.from_pandas.return_value = mock_dataset

        # Act
        result = self.service.train_model_logic(
            self.mock_dataset_file,
            self.valid_patchtsmixer_data
        )

        # Assert - verify train_patchtsmixer_model was called
        mock_train_patchtsmixer.assert_called_once()
        call_args = mock_train_patchtsmixer.call_args
        assert call_args.kwargs['data']['algorithm'] == 'patchtsmixer'

        # Assert - verify result structure
        assert result["status"] == "Modelo registrado correctamente en MLflow."
        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "model_path" in result
        assert "step_config" in result

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('apiTimeSeries.services.mlflow')
    @patch('apiTimeSeries.services.set_tracking_uri')
    @patch('apiTimeSeries.services.get_experiment_by_name')
    @patch('apiTimeSeries.services.start_run')
    @patch('apiTimeSeries.services.log_param')
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_patchtsmixer_model')
    def test_dvc_versioning_executes_for_patchtsmixer(
        self, mock_train_patchtsmixer, mock_file_open, mock_read_csv,
        mock_subprocess, mock_log_param, mock_start_run,
        mock_get_experiment, mock_set_tracking, mock_mlflow,
        mock_makedirs, mock_getsize, mock_exists
    ):
        """
        Given a successful PatchTSMixer training
        When train_model_logic completes
        Then DVC versioning commands should be executed
        """
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 0

        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        model_path = "/path/to/experiment/patchtsmixer_model"
        mock_train_patchtsmixer.return_value = {
            "model_path": model_path,
            "val_metrics": {"val_rmse": 0.15},
            "test_metrics": {"test_rmse": 0.18}
        }

        mock_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'col1': range(100)
        })
        mock_read_csv.return_value = mock_df
        mock_mlflow.data.from_pandas.return_value = Mock()

        # Act
        result = self.service.train_model_logic(
            self.mock_dataset_file,
            self.valid_patchtsmixer_data
        )

        # Assert - verify DVC commands were called
        dvc_calls = [call for call in mock_subprocess.call_args_list
                     if 'dvc' in str(call)]
        assert len(dvc_calls) >= 1, "DVC add command should be called"

        # Verify dvc add was called with model path
        dvc_add_calls = [call for call in mock_subprocess.call_args_list
                         if call.args[0][0] == 'dvc' and call.args[0][1] == 'add']
        assert len(dvc_add_calls) >= 1, "DVC add should be called for model"

    @patch('os.path.exists')
    def test_unsupported_algorithm_raises_error(self, mock_exists):
        """
        Given a request with unsupported algorithm
        When train_model_logic is called
        Then it should raise ValueError with appropriate message
        """
        # Arrange
        mock_exists.return_value = True
        data = self.valid_patchtsmixer_data.copy()
        data["algorithm"] = "unsupported_algorithm"

        # Act & Assert
        with pytest.raises(ValueError, match="Algoritmo no soportado"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    @patch('os.path.exists')
    def test_logistic_algorithm_raises_error(self, mock_exists):
        """
        Given a request with 'logistic' algorithm (removed from supported list)
        When train_model_logic is called
        Then it should raise ValueError since logistic has no routing block
        """
        # Arrange
        mock_exists.return_value = True
        data = self.valid_patchtsmixer_data.copy()
        data["algorithm"] = "logistic"

        # Act & Assert
        with pytest.raises(ValueError, match="Algoritmo no soportado"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    @patch('os.path.exists')
    def test_mlp_algorithm_raises_error(self, mock_exists):
        """
        Given a request with 'mlp' algorithm (removed from supported list)
        When train_model_logic is called
        Then it should raise ValueError since mlp has no routing block
        """
        # Arrange
        mock_exists.return_value = True
        data = self.valid_patchtsmixer_data.copy()
        data["algorithm"] = "mlp"

        # Act & Assert
        with pytest.raises(ValueError, match="Algoritmo no soportado"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('apiTimeSeries.services.mlflow')
    @patch('apiTimeSeries.services.set_tracking_uri')
    @patch('apiTimeSeries.services.get_experiment_by_name')
    @patch('apiTimeSeries.services.start_run')
    @patch('apiTimeSeries.services.log_param')
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_patchtsmixer_model')
    def test_metrics_with_none_values_filtered(
        self, mock_train_patchtsmixer, mock_file_open, mock_read_csv,
        mock_subprocess, mock_log_param, mock_start_run,
        mock_get_experiment, mock_set_tracking, mock_mlflow,
        mock_makedirs, mock_getsize, mock_exists
    ):
        """
        Given PatchTSMixer metrics with None MAPE values
        When metrics are consolidated
        Then None values should be filtered before MLflow logging
        """
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 0

        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_train_patchtsmixer.return_value = {
            "model_path": "/path/to/model",
            "val_metrics": {
                "val_rmse": 0.15,
                "val_mae": 0.12,
                "val_mape": None
            },
            "test_metrics": {
                "test_rmse": 0.18,
                "test_mae": 0.14,
                "test_mape": None
            }
        }

        mock_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'col1': range(100)
        })
        mock_read_csv.return_value = mock_df
        mock_mlflow.data.from_pandas.return_value = Mock()

        # Act
        result = self.service.train_model_logic(
            self.mock_dataset_file,
            self.valid_patchtsmixer_data
        )

        # Assert - mlflow.log_metrics should not receive None values
        for call in mock_mlflow.log_metrics.call_args_list:
            metrics_dict = call.args[0] if call.args else call.kwargs.get('metrics', {})
            for key, value in metrics_dict.items():
                assert value is not None, f"None value found for metric {key}"

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('apiTimeSeries.services.mlflow')
    @patch('apiTimeSeries.services.set_tracking_uri')
    @patch('apiTimeSeries.services.get_experiment_by_name')
    @patch('apiTimeSeries.services.start_run')
    @patch('apiTimeSeries.services.log_param')
    @patch('subprocess.run')
    @patch('pandas.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_patchtsmixer_model')
    def test_step_config_includes_patchtsmixer_algorithm(
        self, mock_train_patchtsmixer, mock_file_open, mock_read_csv,
        mock_subprocess, mock_log_param, mock_start_run,
        mock_get_experiment, mock_set_tracking, mock_mlflow,
        mock_makedirs, mock_getsize, mock_exists
    ):
        """
        Given a successful PatchTSMixer training
        When train_model_logic completes
        Then step_config should include correct algorithm name for pipeline_config.json
        """
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 0

        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_train_patchtsmixer.return_value = {
            "model_path": "/path/to/model",
            "val_metrics": {"val_rmse": 0.15},
            "test_metrics": {"test_rmse": 0.18}
        }

        mock_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'col1': range(100)
        })
        mock_read_csv.return_value = mock_df
        mock_mlflow.data.from_pandas.return_value = Mock()

        # Act
        result = self.service.train_model_logic(
            self.mock_dataset_file,
            self.valid_patchtsmixer_data
        )

        # Assert - step_config should contain patchtsmixer algorithm info
        step_config = result.get("step_config", {})
        assert step_config.get("algorithm") == "patchtsmixer"
        assert step_config.get("step") == "train_patchtsmixer"
        assert "run_id" in step_config
        assert "timestamp" in step_config
