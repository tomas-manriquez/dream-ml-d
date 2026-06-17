"""
Test suite for Bayesian Search hyperparameter optimization in classification models.

This test suite covers:
1. Unit tests for convert_frontend_bayesian_params()
2. Bayesian search tests for Logistic Regression, XGBoost, and MLP
3. Validation tests for invalid configurations
4. Reproducibility tests
5. Custom parameter range tests
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import shutil

from api.train import (
    train_logistic_regression_model,
    train_xgboost_model,
    train_mlp_model,
    convert_frontend_bayesian_params
)


class TestConvertFrontendBayesianParams:
    """Unit tests for convert_frontend_bayesian_params helper function"""

    def test_frontend_format_real_log_uniform(self):
        """
        Given frontend format with real type and log-uniform distribution
        When convert_frontend_bayesian_params is called
        Then it should convert to backend format with type=float and log=True
        """
        # Arrange
        frontend_params = {
            "C": {
                "type": "real",
                "distribution": "log-uniform",
                "low": 0.001,
                "high": 100.0
            }
        }

        # Act
        result = convert_frontend_bayesian_params(frontend_params)

        # Assert
        assert result["C"]["type"] == "float"
        assert result["C"]["low"] == 0.001
        assert result["C"]["high"] == 100.0
        assert result["C"]["log"] is True

    def test_frontend_format_integer_uniform(self):
        """
        Given frontend format with integer type and uniform distribution
        When convert_frontend_bayesian_params is called
        Then it should convert to backend format with type=int and log=False
        """
        # Arrange
        frontend_params = {
            "max_iter": {
                "type": "integer",
                "distribution": "uniform",
                "low": 100,
                "high": 1000
            }
        }

        # Act
        result = convert_frontend_bayesian_params(frontend_params)

        # Assert
        assert result["max_iter"]["type"] == "int"
        assert result["max_iter"]["low"] == 100
        assert result["max_iter"]["high"] == 1000
        assert result["max_iter"]["log"] is False

    def test_frontend_format_categorical(self):
        """
        Given frontend format with categorical type
        When convert_frontend_bayesian_params is called
        Then it should preserve type and choices
        """
        # Arrange
        frontend_params = {
            "solver": {
                "type": "categorical",
                "choices": ["lbfgs", "liblinear", "saga"]
            }
        }

        # Act
        result = convert_frontend_bayesian_params(frontend_params)

        # Assert
        assert result["solver"]["type"] == "categorical"
        assert result["solver"]["choices"] == ["lbfgs", "liblinear", "saga"]

    def test_backend_format_already_converted(self):
        """
        Given parameters already in backend format
        When convert_frontend_bayesian_params is called
        Then it should preserve them as-is
        """
        # Arrange
        backend_params = {
            "C": {
                "type": "float",
                "low": 0.001,
                "high": 100.0,
                "log": True
            }
        }

        # Act
        result = convert_frontend_bayesian_params(backend_params)

        # Assert
        assert result["C"]["type"] == "float"
        assert result["C"]["low"] == 0.001
        assert result["C"]["high"] == 100.0
        assert result["C"]["log"] is True

    def test_mixed_frontend_backend_formats(self):
        """
        Given mix of frontend and backend format parameters
        When convert_frontend_bayesian_params is called
        Then it should convert frontend params and preserve backend params
        """
        # Arrange
        mixed_params = {
            "C": {
                "type": "real",
                "distribution": "log-uniform",
                "low": 0.001,
                "high": 100.0
            },
            "max_iter": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "log": False
            }
        }

        # Act
        result = convert_frontend_bayesian_params(mixed_params)

        # Assert
        assert result["C"]["type"] == "float"
        assert result["C"]["log"] is True
        assert result["max_iter"]["type"] == "int"
        assert result["max_iter"]["log"] is False

    def test_log_uniform_validation_negative_values(self):
        """
        Given log-uniform distribution with negative values
        When convert_frontend_bayesian_params is called
        Then it should raise ValueError
        """
        # Arrange
        frontend_params = {
            "C": {
                "type": "real",
                "distribution": "log-uniform",
                "low": -1.0,
                "high": 100.0
            }
        }

        # Act & Assert
        with pytest.raises(ValueError, match="log-uniform.*non-positive values"):
            convert_frontend_bayesian_params(frontend_params)


class TestBayesianSearchLogisticRegression:
    """Test cases for Bayesian Search in Logistic Regression"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create synthetic binary classification dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        # Save to temporary CSV
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.test_csv_path, index=False)

        self.experiment_dir = os.path.join(self.temp_dir, "test_experiment")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.base_data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestLogisticBayesian"
        }

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_logistic_bayesian_basic(self, mock_log_artifact, mock_log_metrics,
                                     mock_log_params, mock_active_run):
        """
        Given valid Bayesian config with n_trials=5
        When train_logistic_regression_model is called
        Then it should complete optimization and return best params
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2,
            "timeout_seconds": None
        }
        data["bayesian_search_params"] = {}

        # Act
        result = train_logistic_regression_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "model_path" in result

        # Verify MLflow logging was called
        assert mock_log_params.called
        assert mock_log_metrics.called

        # Verify pipeline_config.json was created
        pipeline_config_path = os.path.join(self.experiment_dir, "pipeline_config.json")
        assert os.path.exists(pipeline_config_path)

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_logistic_bayesian_custom_ranges_backend_format(self, mock_log_artifact,
                                                             mock_log_metrics, mock_log_params,
                                                             mock_active_run):
        """
        Given custom parameter ranges in backend format
        When train_logistic_regression_model is called
        Then it should use custom ranges and complete successfully
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 300}
        }

        # Act
        result = train_logistic_regression_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_logistic_bayesian_custom_ranges_frontend_format(self, mock_log_artifact,
                                                              mock_log_metrics, mock_log_params,
                                                              mock_active_run):
        """
        Given custom parameter ranges in frontend format
        When train_logistic_regression_model is called
        Then it should convert and use custom ranges successfully
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {
            "C": {
                "type": "real",
                "distribution": "log-uniform",
                "low": 0.1,
                "high": 10.0
            },
            "max_iter": {
                "type": "integer",
                "distribution": "uniform",
                "low": 100,
                "high": 300
            }
        }

        # Act
        result = train_logistic_regression_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result


class TestBayesianSearchXGBoost:
    """Test cases for Bayesian Search in XGBoost"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create synthetic binary classification dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.test_csv_path, index=False)

        self.experiment_dir = os.path.join(self.temp_dir, "test_experiment")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.base_data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "problem_type": "binary",
            "hyperparameter_search_strategy": "bayesian",
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestXGBoostBayesian"
        }

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_xgboost_bayesian_basic(self, mock_log_artifact, mock_log_metrics,
                                    mock_log_params, mock_active_run):
        """
        Given valid Bayesian config with n_trials=5
        When train_xgboost_model is called
        Then it should complete optimization with early stopping
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {}

        # Act
        result = train_xgboost_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result
        assert "test_metrics" in result

        # Verify MLflow logging was called
        assert mock_log_params.called
        assert mock_log_metrics.called

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_xgboost_bayesian_custom_ranges(self, mock_log_artifact, mock_log_metrics,
                                            mock_log_params, mock_active_run):
        """
        Given custom parameter ranges for XGBoost
        When train_xgboost_model is called
        Then it should use custom ranges successfully
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True}
        }

        # Act
        result = train_xgboost_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result


class TestBayesianSearchMLP:
    """Test cases for Bayesian Search in MLP"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create synthetic binary classification dataset
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.test_csv_path, index=False)

        self.experiment_dir = os.path.join(self.temp_dir, "test_experiment")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.base_data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "problem_type": "binary",
            "hyperparameter_search_strategy": "bayesian",
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestMLPBayesian"
        }

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_mlp_bayesian_basic(self, mock_log_artifact, mock_log_metrics,
                                mock_log_params, mock_active_run):
        """
        Given valid Bayesian config with n_trials=5
        When train_mlp_model is called
        Then it should complete optimization successfully
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {}

        # Act
        result = train_mlp_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result
        assert "test_metrics" in result

        # Verify MLflow logging was called
        assert mock_log_params.called
        assert mock_log_metrics.called

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_mlp_bayesian_custom_ranges(self, mock_log_artifact, mock_log_metrics,
                                        mock_log_params, mock_active_run):
        """
        Given custom parameter ranges for MLP
        When train_mlp_model is called
        Then it should use custom ranges successfully
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 5,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {
            "hidden_layer_sizes": {
                "type": "categorical",
                "choices": [(10,), (50,), (100,)]
            },
            "learning_rate_init": {"type": "float", "low": 0.001, "high": 0.01, "log": True}
        }

        # Act
        result = train_mlp_model(
            self.test_csv_path, data, self.experiment_dir
        )

        # Assert
        assert result is not None
        assert "val_metrics" in result


class TestBayesianSearchValidation:
    """Test cases for validation of Bayesian Search configurations"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.test_csv_path, index=False)

        self.experiment_dir = os.path.join(self.temp_dir, "test_experiment")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.base_data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "model_name": "TestValidation"
        }

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    def test_validation_n_trials_too_small(self, mock_active_run):
        """
        Given Bayesian config with n_trials < 1
        When train_logistic_regression_model is called
        Then it should raise ValueError
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 0,
            "n_initial_points": 2
        }
        data["bayesian_search_params"] = {}

        # Act & Assert
        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )

    @patch('mlflow.active_run')
    def test_validation_n_initial_points_too_large(self, mock_active_run):
        """
        Given Bayesian config with n_initial_points >= n_trials
        When train_logistic_regression_model is called
        Then it should raise ValueError
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 10,
            "n_initial_points": 10  # Equal to n_trials
        }
        data["bayesian_search_params"] = {}

        # Act & Assert
        with pytest.raises(ValueError, match="n_initial_points .* must be less than n_trials"):
            train_logistic_regression_model(
                self.test_csv_path, data, self.experiment_dir
            )


class TestBayesianSearchReproducibility:
    """Test cases for reproducibility of Bayesian Search"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y

        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(self.test_csv_path, index=False)

        self.base_data = {
            "input_features": [f"feature_{i}" for i in range(5)],
            "target_variable": "target",
            "hyperparameter_search_strategy": "bayesian",
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        }

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_artifact')
    def test_reproducibility_same_seed_same_results(self, mock_log_artifact,
                                                     mock_log_metrics, mock_log_params,
                                                     mock_active_run):
        """
        Given same Bayesian config and same seed
        When train_logistic_regression_model is called twice
        Then it should produce identical results
        """
        # Arrange
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_active_run.return_value = mock_run

        data = self.base_data.copy()
        data["bayesian_config"] = {
            "n_trials": 10,
            "n_initial_points": 3
        }
        data["bayesian_search_params"] = {}

        # First run
        experiment_dir_1 = os.path.join(self.temp_dir, "exp1")
        os.makedirs(experiment_dir_1, exist_ok=True)
        data["model_name"] = "TestRepro1"
        result1 = train_logistic_regression_model(
            self.test_csv_path, data, experiment_dir_1
        )

        # Second run
        experiment_dir_2 = os.path.join(self.temp_dir, "exp2")
        os.makedirs(experiment_dir_2, exist_ok=True)
        data["model_name"] = "TestRepro2"
        result2 = train_logistic_regression_model(
            self.test_csv_path, data, experiment_dir_2
        )

        # Assert - Results should be identical (within floating point tolerance)
        assert abs(result1["val_metrics"]["val_accuracy"] -
                  result2["val_metrics"]["val_accuracy"]) < 1e-4
