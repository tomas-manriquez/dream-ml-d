"""api_tests specific fixtures."""
import pytest
from unittest.mock import MagicMock, patch
from django.test import RequestFactory


@pytest.fixture(scope="function")
def request_factory():
    """Django RequestFactory for view testing.

    Scope: function - creates fresh factory for each test
    """
    return RequestFactory()


@pytest.fixture(scope="function")
def mock_subprocess_success():
    """Mock successful subprocess.run execution.

    Scope: function - creates fresh mock for each test
    """
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    return mock_result


@pytest.fixture(scope="function")
def mock_dvc_initialized():
    """Mock DVC initialization checks.

    Scope: function - patch context for each test
    """
    with patch('os.path.exists') as mock_exists, \
         patch('os.path.isdir') as mock_isdir:
        mock_exists.return_value = True
        mock_isdir.return_value = True
        yield (mock_exists, mock_isdir)


@pytest.fixture(scope="function")
def mock_mlflow_tracking():
    """Mock MLflow tracking URI and experiment setup.

    Scope: function - patch context for each test
    """
    with patch('api.views.mlflow.set_tracking_uri') as mock_uri, \
         patch('api.views.mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('api.views.mlflow.create_experiment') as mock_create:
        mock_get_exp.return_value = None  # Experiment doesn't exist
        mock_create.return_value = "test-exp-id"
        yield {
            'set_tracking_uri': mock_uri,
            'get_experiment_by_name': mock_get_exp,
            'create_experiment': mock_create
        }


# ============================================================================
# Phase 9 Stage 2 Fixtures - services.py unit testing
# Created: 2026-01-07
# Purpose: Reusable mocks for comprehensive services.py orchestration testing
# ============================================================================


@pytest.fixture(scope="function")
def mock_csv_file(tmp_path):
    """Create mock CSV file with chunks() method for Django file uploads.

    Returns:
        MagicMock: Mock file object with .name and .chunks() method

    Usage:
        test_func(mock_csv_file):
            result = upload_and_clean_csv_logic(csv_file=mock_csv_file, ...)
    """
    mock_file = MagicMock()
    mock_file.name = "test_data.csv"
    # Simulate Django InMemoryUploadedFile chunks
    mock_file.chunks.return_value = [
        b"col1,col2,col3\n",
        b"1,2,3\n",
        b"4,5,6\n"
    ]
    return mock_file


@pytest.fixture(scope="function")
def mock_limpiar_datos_result():
    """Standard return value for limpiar_datos() function.

    Returns realistic cleaning report matching actual data_cleaning module output.

    Returns:
        dict: Cleaning report with all expected keys
    """
    return {
        "filas_totales": 1000,
        "filas_eliminadas": 50,
        "filas_restantes": 950,
        "columnas_eliminadas": ["col_with_nulls", "duplicate_col"],
        "valores_imputados": {
            "numeric_col": 25,
            "another_col": 10
        },
        "duplicados_eliminados": 15,
        "outliers_tratados": 8,
        "reporte_limpieza": "Limpieza completada exitosamente"
    }


@pytest.fixture(scope="function")
def mock_codificar_datos_result():
    """Standard return value for codificar_datos() function.

    Returns realistic encoding info matching actual data_encoding module output.

    Returns:
        dict: Encoding information with transformation details
    """
    return {
        "columnas_codificadas": ["category1", "category2"],
        "encoding_type": "OneHotEncoding",
        "nuevas_columnas": [
            "category1_A", "category1_B",
            "category2_X", "category2_Y"
        ],
        "filas_procesadas": 950,
        "encoding_info": "Codificación aplicada correctamente"
    }


@pytest.fixture(scope="function")
def mock_train_logistic_result():
    """Standard return value for train_logistic_regression_model().

    Returns realistic training result with metrics and model path.

    Returns:
        dict: Training result with validation/test metrics
    """
    return {
        "model_path": "/tmp/experiment/trained/logistic_model.pkl",
        "val_metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "roc_auc": 0.89
        },
        "test_metrics": {
            "accuracy": 0.84,
            "precision": 0.82,
            "recall": 0.86,
            "f1_score": 0.84,
            "roc_auc": 0.88
        },
        "training_time": 12.5,
        "model_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs"
        }
    }


@pytest.fixture(scope="function")
def mock_train_mlp_result():
    """Standard return value for train_mlp_model().

    Returns realistic MLP training result with metrics.

    Returns:
        dict: Training result for MLP model
    """
    return {
        "model_path": "/tmp/experiment/trained/mlp_model.pkl",
        "val_metrics": {
            "accuracy": 0.88,
            "precision": 0.86,
            "recall": 0.90,
            "f1_score": 0.88,
            "roc_auc": 0.92
        },
        "test_metrics": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89,
            "f1_score": 0.87,
            "roc_auc": 0.91
        },
        "training_time": 45.2,
        "model_params": {
            "hidden_layer_sizes": [100, 50],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500
        }
    }


@pytest.fixture(scope="function")
def mock_train_xgboost_result():
    """Standard return value for train_xgboost_model().

    Returns realistic XGBoost training result with metrics.

    Returns:
        dict: Training result for XGBoost model
    """
    return {
        "model_path": "/tmp/experiment/trained/xgboost_model.pkl",
        "val_metrics": {
            "accuracy": 0.91,
            "precision": 0.89,
            "recall": 0.93,
            "f1_score": 0.91,
            "roc_auc": 0.95
        },
        "test_metrics": {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
            "roc_auc": 0.94
        },
        "training_time": 28.7,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8
        }
    }


@pytest.fixture(scope="function")
def mock_mlflow_experiment():
    """Mock MLflow experiment object.

    Returns:
        MagicMock: Mock with experiment_id, artifact_location, name
    """
    mock_exp = MagicMock()
    mock_exp.experiment_id = "test_exp_123"
    mock_exp.name = "Test_Experiment"
    mock_exp.artifact_location = "file:///tmp/mlruns/test_exp_123"
    return mock_exp


@pytest.fixture(scope="function")
def mock_mlflow_run():
    """Mock MLflow run object.

    Returns:
        MagicMock: Mock with info.run_id
    """
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_456"
    return mock_run


@pytest.fixture(scope="function")
def mock_channel_layer():
    """Mock channel layer for WebSocket testing.

    Returns:
        MagicMock: Mock with group_send method
    """
    mock_layer = MagicMock()
    # group_send is async, but async_to_sync handles it
    mock_layer.group_send = MagicMock()
    return mock_layer


@pytest.fixture(scope="function")
def mock_emissions_tracker():
    """Mock CodeCarbon EmissionsTracker.

    Returns:
        MagicMock: Mock tracker with start/stop methods and energy metrics
    """
    mock_tracker = MagicMock()
    mock_tracker.start.return_value = None
    mock_tracker.stop.return_value = None
    # Realistic energy values
    mock_tracker._total_energy = 0.0025  # 2.5 Wh
    mock_tracker.final_emissions = 0.0012  # 1.2 gCO2
    return mock_tracker


# ============================================================================
# Phase 10 Fixtures - Added 2026-01-08
# Purpose: Complex mock setups for services.py testing
# ============================================================================


@pytest.fixture(scope="function")
def mock_subprocess_dvc_success():
    """Mock successful DVC subprocess operations.

    Returns:
        MagicMock: Configured for dvc add, dvc push, git add, git commit
    """
    mock_subprocess = MagicMock()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_subprocess.return_value = mock_result
    return mock_subprocess


@pytest.fixture(scope="function")
def mock_mlflow_client_with_experiment(mock_mlflow_experiment):
    """Mock MlflowClient with get_experiment_by_name returning an experiment.

    Args:
        mock_mlflow_experiment: Fixture providing experiment mock

    Returns:
        MagicMock: Client with experiment lookup capability
    """
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = mock_mlflow_experiment
    return mock_client


@pytest.fixture(scope="function")
def mock_subprocess_dvc_failure():
    """Mock failed DVC subprocess operation.

    Returns:
        MagicMock: Raises CalledProcessError for DVC operations
    """
    import subprocess

    def side_effect_dvc_failure(cmd, *args, **kwargs):
        if 'dvc' in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr="DVC error")
        return MagicMock(returncode=0)

    mock_subprocess = MagicMock()
    mock_subprocess.side_effect = side_effect_dvc_failure
    return mock_subprocess


@pytest.fixture(scope="function")
def mock_file_operations_for_csv():
    """Mock file operations for CSV upload testing.

    Returns:
        dict: Dictionary with mock os.path functions
    """
    return {
        'exists': MagicMock(return_value=False),
        'getsize': MagicMock(return_value=1024),
        'isdir': MagicMock(return_value=True),
        'makedirs': MagicMock()
    }


@pytest.fixture(scope="function")
def mock_mlflow_deep(mock_mlflow_experiment):
    """Deep MLflow mocking that prevents database initialization.

    This fixture patches MLflow at the lowest level to prevent TrackingServiceClient
    from initializing SQLite database and requiring alembic.ini. This is the
    solution to the Phase 9 MLflow blocker.

    The fixture mocks:
    - mlflow.tracking._tracking_service.utils._get_store (prevents DB init)
    - mlflow.tracking.MlflowClient (prevents client instantiation issues)
    - mlflow module functions (set_tracking_uri, get_experiment_by_name, etc.)

    Returns:
        dict: Dictionary with all MLflow mocks for test assertions

    Usage:
        def test_something(mock_mlflow_deep):
            # MLflow operations won't trigger DB initialization
            result = some_function_that_uses_mlflow()
            assert mock_mlflow_deep['get_experiment'].called
    """
    with patch('mlflow.tracking._tracking_service.utils._get_store') as mock_get_store, \
         patch('mlflow.tracking.MlflowClient') as mock_client_class, \
         patch('api.services.set_tracking_uri') as mock_set_uri, \
         patch('api.services.get_experiment_by_name') as mock_get_exp, \
         patch('api.services.start_run') as mock_start_run, \
         patch('api.services.log_param') as mock_log_param, \
         patch('api.services.log_metric') as mock_log_metric, \
         patch('api.services.log_artifact') as mock_log_artifact, \
         patch('api.services.mlflow.log_input') as mock_log_input, \
         patch('api.services.mlflow.data.from_pandas') as mock_from_pandas:

        # Configure mock store
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.get_experiment_by_name.return_value = mock_mlflow_experiment
        mock_client_class.return_value = mock_client_instance

        # Configure mock experiment lookup
        mock_get_exp.return_value = mock_mlflow_experiment

        # Configure mock run context manager
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        # Configure mock dataset
        mock_dataset = MagicMock()
        mock_from_pandas.return_value = mock_dataset

        yield {
            'get_store': mock_get_store,
            'client_class': mock_client_class,
            'client_instance': mock_client_instance,
            'set_tracking_uri': mock_set_uri,
            'get_experiment': mock_get_exp,
            'start_run': mock_start_run,
            'run': mock_run,
            'log_param': mock_log_param,
            'log_metric': mock_log_metric,
            'log_artifact': mock_log_artifact,
            'log_input': mock_log_input,
            'from_pandas': mock_from_pandas,
            'dataset': mock_dataset
        }
