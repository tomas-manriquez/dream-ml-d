"""Root conftest.py for shared test fixtures across all test packages."""
import pytest
import tempfile
import shutil
import os
from unittest.mock import MagicMock
import pandas as pd
import numpy as np


@pytest.fixture(scope="function")
def temp_experiment_dir():
    """Create a temporary experiment directory for tests.

    Yields the directory path and automatically cleans up after test.

    Scope: function - creates fresh directory for each test
    """
    temp_dir = tempfile.mkdtemp(prefix="test_exp_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_mlflow_experiment():
    """Mock MLflow experiment object.

    Scope: session - reusable across all tests
    """
    mock_exp = MagicMock()
    mock_exp.experiment_id = "test-exp-123"
    mock_exp.name = "test-experiment"
    return mock_exp


@pytest.fixture(scope="function")
def mock_mlflow_run():
    """Mock MLflow run object for context manager usage.

    Scope: function - creates fresh mock for each test
    """
    mock_run = MagicMock()
    mock_run.info.run_id = 'test-run-id-456'
    mock_run.info.experiment_id = 'test-exp-123'
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    return mock_run


@pytest.fixture(scope="session")
def sample_dataframe():
    """Sample classification DataFrame for testing.

    Scope: session - expensive to create, reusable across tests
    """
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture(scope="session")
def sample_csv_content():
    """Sample CSV content as string.

    Scope: session - immutable, reusable across tests
    """
    return "feature1,feature2,feature3,target\n1.0,2.0,A,0\n1.5,2.5,B,1\n2.0,3.0,C,0"


@pytest.fixture(scope="session")
def sample_config():
    """Sample experiment configuration.

    Scope: session - immutable dict, reusable across tests
    """
    return {
        'algorithm': 'logistic_regression',
        'model_name': 'test-model',
        'split_ratios': [0.7, 0.15, 0.15],
        'random_state': 42,
        'search_method': 'grid',
        'n_trials': 2
    }


@pytest.fixture(scope="function")
def set_global_seed():
    """Fixture to set global random seeds for reproducibility.

    Scope: function - allows different tests to use different seeds
    """
    def _set_seed(seed=42):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

    return _set_seed
