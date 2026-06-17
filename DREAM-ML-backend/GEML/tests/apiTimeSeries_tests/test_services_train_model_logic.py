import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
import pandas as pd

# Assuming the service class is in apiTimeSeries.services
from apiTimeSeries.services import TrainModelService


class TestTrainModelLogic:
    """Test cases for train_model_logic method"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = TrainModelService()
        self.mock_dataset_file = Mock()
        self.mock_dataset_file.chunks.return_value = [b'test,data\n1,2\n3,4\n']
        
        self.valid_data = {
            "experiment_dir": "/path/to/experiment",
            "algorithm": "arima",
            "model_name": "test_model",
            "input_features": ["feature1", "feature2"],
            "target_variable": "target",
            "date_col_name": "date",
            "split_ratios": {"train": 0.8, "val": 0.2},
            "run_id": "test_run",
            "params": {"p": 1, "d": 1, "q": 1},
            "use_grid_search": False,
            "problem_type": "ts_forecasting",
            "forecast_horizon": 30
        }

    # Validation Tests
    def test_train_model_logic_missing_experiment_dir(self):
        """
        Given a request with missing 'experiment_dir' in data
        When train_model_logic is called
        Then it should raise FileNotFoundError
        """
        # Arrange
        data = self.valid_data.copy()
        del data["experiment_dir"]

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Directorio de experimento no encontrado o inválido"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    def test_train_model_logic_empty_experiment_dir(self):
        """
        Given a request with empty 'experiment_dir' in data
        When train_model_logic is called
        Then it should raise FileNotFoundError
        """
        # Arrange
        data = self.valid_data.copy()
        data["experiment_dir"] = ""

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Directorio de experimento no encontrado o inválido"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    @patch('os.path.exists')
    def test_train_model_logic_nonexistent_experiment_dir(self, mock_exists):
        """
        Given a request with non-existent 'experiment_dir' path
        When train_model_logic is called
        Then it should raise FileNotFoundError
        """
        # Arrange
        mock_exists.return_value = False

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Directorio de experimento no encontrado o inválido"):
            self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

    @patch('os.path.exists')
    def test_train_model_logic_unsupported_algorithm(self, mock_exists):
        """
        Given a request with unsupported algorithm
        When train_model_logic is called
        Then it should raise ValueError
        """
        # Arrange
        mock_exists.return_value = True
        data = self.valid_data.copy()
        data["algorithm"] = "unsupported_algorithm"

        # Act & Assert
        with pytest.raises(ValueError, match="Algoritmo no soportado"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    # MLflow Integration Tests
    @patch('os.path.exists')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    def test_train_model_logic_mlflow_experiment_not_found(self, mock_get_experiment, mock_set_tracking, mock_exists):
        """
        Given a request with non-existent MLflow experiment
        When train_model_logic is called
        Then it should raise ValueError
        """
        # Arrange
        mock_exists.return_value = True
        mock_get_experiment.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="No se encontró el experimento"):
            self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

    # Success Case Tests
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_input')
    @patch('mlflow.data.from_pandas')
    @patch('mlflow.set_tag')
    @patch('subprocess.run')
    @patch('pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_arima_model')
    def test_train_model_logic_successful_execution(self, mock_train_arima, mock_file_open, mock_read_csv,
                                                   mock_subprocess, mock_set_tag, mock_from_pandas, mock_log_input,
                                                   mock_log_metrics, mock_log_param, mock_start_run,
                                                   mock_get_experiment, mock_set_tracking, mock_makedirs,
                                                   mock_getsize, mock_exists):
        """
        Given a valid request with complete data payload
        When train_model_logic executes successfully
        Then it should return success response with run_id, metrics, and model_path
        """
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 0  # File doesn't exist initially
        
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        mock_train_arima.return_value = {
            "model_path": "/path/to/model.pkl",
            "val_metrics": {"mae": 0.1, "mse": 0.2},
            "test_metrics": {"mae": 0.15, "mse": 0.25}
        }
        
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        mock_read_csv.return_value = mock_df
        
        mock_dataset = Mock()
        mock_from_pandas.return_value = mock_dataset

        # Act
        result = self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

        # Assert
        assert result["status"] == "Modelo registrado correctamente en MLflow."
        assert result["run_id"] == "test_run_id"
        assert result["model_path"] == "/path/to/model.pkl"
        assert "val_metrics" in result
        assert result["val_metrics"]["mae"] == 0.1
        assert "step_config" in result
        
        # Verify MLflow calls
        mock_set_tracking.assert_called_once()
        mock_get_experiment.assert_called_once()
        mock_start_run.assert_called_once()
        mock_log_param.assert_called_with("step", "arima_training")
        mock_log_input.assert_called_once()

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_input')
    @patch('mlflow.data.from_pandas')
    @patch('mlflow.set_tag')
    @patch('subprocess.run')
    @patch('pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_arima_model')
    def test_train_model_logic_success_without_val_metrics(self, mock_train_arima, mock_file_open, mock_read_csv,
                                                          mock_subprocess, mock_set_tag, mock_from_pandas, 
                                                          mock_log_input, mock_log_metrics, mock_log_param,
                                                          mock_start_run, mock_get_experiment, mock_set_tracking,
                                                          mock_makedirs, mock_getsize, mock_exists):
        """
        Given a valid request
        When train_model_logic executes successfully but returns partial result without val_metrics
        Then it should return success response with empty metrics
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
        
        # Return result without val_metrics
        mock_train_arima.return_value = {
            "model_path": "/path/to/model.pkl"
        }
        
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        # Act
        result = self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

        # Assert
        assert result["status"] == "Modelo registrado correctamente en MLflow."
        assert result["val_metrics"] == {}
        assert "step_config" in result

    # Error Handling Tests
    @pytest.mark.slow
    @patch('os.path.exists')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.end_run')
    @patch('mlflow.set_tag')
    @patch('os.makedirs')
    @patch('builtins.open', side_effect=IOError("File write error"))
    def test_train_model_logic_runtime_error(self, mock_file_open, mock_makedirs, mock_set_tag,
                                            mock_end_run, mock_start_run, mock_get_experiment,
                                            mock_set_tracking, mock_exists):
        """
        Given a valid request
        When train_model_logic encounters a runtime error during execution
        Then it should raise RuntimeError
        """
        # Arrange
        mock_exists.return_value = True
        
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Act & Assert
        with pytest.raises(RuntimeError, match="Error en el proceso de entrenamiento"):
            self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

    # Algorithm-specific Tests
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_input')
    @patch('mlflow.data.from_pandas')
    @patch('mlflow.set_tag')
    @patch('subprocess.run')
    @patch('pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_arima_model')
    def test_train_model_logic_arima_algorithm(self, mock_train_arima, mock_file_open, mock_read_csv,
                                              mock_subprocess, mock_set_tag, mock_from_pandas, mock_log_input,
                                              mock_log_param, mock_start_run, mock_get_experiment,
                                              mock_set_tracking, mock_makedirs, mock_getsize, mock_exists):
        """
        Given a valid request with algorithm="arima" and problem_type="ts_forecasting"
        When train_model_logic is called
        Then it should proceed with time series forecasting workflow
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
        
        mock_train_arima.return_value = {
            "model_path": "/path/to/arima_model.pkl",
            "val_metrics": {"mae": 0.1}
        }
        
        mock_df = pd.DataFrame({'date': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
        mock_read_csv.return_value = mock_df

        data = self.valid_data.copy()
        data["algorithm"] = "arima"
        data["problem_type"] = "ts_forecasting"

        # Act
        result = self.service.train_model_logic(self.mock_dataset_file, data)

        # Assert
        assert result["status"] == "Modelo registrado correctamente en MLflow."
        mock_train_arima.assert_called_once()
        mock_log_param.assert_called_with("step", "arima_training")

    # Edge Case Tests
    @patch('os.path.exists')
    def test_train_model_logic_missing_algorithm_defaults_to_logistic(self, mock_exists):
        """
        Given a request without algorithm parameter
        When train_model_logic is called
        Then it should default to 'logistic' algorithm
        """
        # Arrange
        mock_exists.return_value = True
        data = self.valid_data.copy()
        del data["algorithm"]

        # Act & Assert
        with pytest.raises(ValueError, match="Algoritmo no soportado. Use: logistic, mlp, xgboost, arima"):
            self.service.train_model_logic(self.mock_dataset_file, data)

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_input')
    @patch('mlflow.data.from_pandas')
    @patch('subprocess.run')
    @patch('pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('apiTimeSeries.services.train_arima_model')
    def test_train_model_logic_success_without_model_path(self, mock_train_arima, mock_file_open, mock_read_csv,
                                                         mock_subprocess, mock_from_pandas, mock_log_input,
                                                         mock_log_param, mock_start_run, mock_get_experiment,
                                                         mock_set_tracking, mock_makedirs, mock_getsize, mock_exists):
        """
        Given a valid request
        When train_model_logic executes successfully but returns result without model_path
        Then it should return success response with empty model_path
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
        
        # Return result without model_path
        mock_train_arima.return_value = {
            "val_metrics": {"mae": 0.1}
        }
        
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        # Act
        result = self.service.train_model_logic(self.mock_dataset_file, self.valid_data)

        # Assert
        assert result["status"] == "Modelo registrado correctamente en MLflow."
        assert result["model_path"] is None
        assert "step_config" in result
        assert result["step_config"]["model_path"] is None