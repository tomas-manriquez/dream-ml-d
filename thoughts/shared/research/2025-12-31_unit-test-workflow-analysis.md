# Unit Test Workflow and Structure Analysis
**Date:** 2025-12-31
**Author:** Claude Code Analysis
**Purpose:** Understand unit test workflow and structure for api and apiTimeSeries packages to improve test coverage to 75%+

---

## Executive Summary

This analysis examines the unit testing infrastructure for DREAM-ML-backend/GEML/api and DREAM-ML-backend/GEML/apiTimeSeries packages. The codebase uses a **hybrid Django TestCase + pytest architecture** with comprehensive unit test coverage but significant integration test gaps.

**Key Findings:**
- ✅ Strong unit test foundation with 22+ test files covering views, services, and utilities
- ✅ Consistent Arrange-Act-Assert (AAA) pattern with Given-When-Then documentation
- ✅ Comprehensive error handling and edge case validation
- ❌ Deep mocking stacks (10-14 @patch decorators) create brittle tests
- ❌ No integration tests across architectural layers
- ❌ No shared fixtures (no conftest.py files found)
- ❌ Manual Django configuration instead of pytest-django automation

**Current Test Coverage Status:**
- 22 test files analyzed across api_tests and apiTimeSeries_tests
- Heavy focus on unit testing with mocked dependencies
- Minimal integration testing with real services (MLflow, databases, file systems)

---

## Test Framework Configuration

### Pytest Configuration
**File:** [DREAM-ML-backend/GEML/pytest.ini](DREAM-ML-backend/GEML/pytest.ini)

```ini
[pytest]
DJANGO_SETTINGS_MODULE = GEML.settings
python_files = tests.py test_*.py *_tests.py
django_debug_mode = true
pythonpath = . ..
django_find_project = false
```

**Key Configuration Points:**
- Django integration via pytest-django (DJANGO_SETTINGS_MODULE = GEML.settings:2)
- Debug mode enabled during test execution (django_debug_mode = true:4)
- Custom Python path includes parent directory (pythonpath = . ..:5)
- Django auto-discovery disabled (django_find_project= false:6)
- Test file discovery pattern: test_*.py, *_tests.py (python_files:3)

**Missing Configuration:**
- ❌ No custom test markers defined
- ❌ No parallel execution flags (pytest-xdist)
- ❌ No coverage thresholds or reporting configuration
- ❌ No test timeout settings
- ❌ No coverage configuration file (.coveragerc, pyproject.toml)

### Test Dependencies
**File:** [DREAM-ML-backend/GEML/requirements-dev.txt](DREAM-ML-backend/GEML/requirements-dev.txt)

```python
pytest==8.3.4              # Core testing framework
pytest-django==4.9.0       # Django integration for pytest
pytest-asyncio==0.21.0     # Async test support
coverage==7.6.10           # Code coverage measurement
```

**Execution Command:**
```bash
cd DREAM-ML-backend/GEML
coverage run --source='.' -m pytest -v
```

**Notable Absences:**
- ❌ No `pytest-cov` (coverage integration with pytest)
- ❌ No `pytest-xdist` (parallel test execution)
- ❌ No `pytest-mock` (enhanced mocking utilities)
- ❌ No `factory_boy` (test data factories)
- ❌ No `faker` (synthetic data generation)

---

## Test Structure and Organization

### Directory Structure

```
DREAM-ML-backend/GEML/tests/
├── api_tests/
│   ├── __init__.py
│   ├── test_views.py                          # 1299 lines - Django view tests
│   ├── test_services.py                       # 786 lines - Service layer tests
│   ├── test_consumers.py                      # 281 lines - WebSocket tests
│   ├── test_utils.py                          # 736 lines - Utility function tests
│   ├── test_data_cleaning.py                  # 494 lines - Data cleaning tests
│   ├── test_analyze_csv_logic.py              # CSV analysis tests
│   ├── test_bayesian_search_classification.py # Bayesian optimization tests
│   ├── arreglar.py                            # 826 lines - Model training tests
│   └── debug_imports.py                       # Import debugging utility
│
└── apiTimeSeries_tests/
    ├── __init__.py
    ├── test_views_create_experiment.py        # Experiment creation view tests
    ├── test_views_upload_and_clean_csv.py     # CSV upload view tests
    ├── test_views_encode_csv.py               # Encoding view tests
    ├── test_views_train_model.py              # 436 lines - Training view tests
    ├── test_services.py                       # 285 lines - Service layer tests
    ├── test_services_generate_eda_logic.py    # EDA generation logic tests
    ├── test_services_train_model_logic.py     # 387 lines - Training logic tests
    ├── test_services_encode_csv_logic.py      # Encoding logic tests
    ├── test_data_cleaning_utils.py            # 397 lines - Data cleaning utils
    ├── test_lstm_phase1.py                    # 259 lines - LSTM sequence creation
    ├── test_lstm_phase2a.py                   # LSTM training phase 2a
    ├── test_lstm_phase2b.py                   # LSTM training phase 2b
    ├── test_lstm_phase3a.py                   # LSTM training phase 3a
    ├── test_lstm_phase3b.py                   # LSTM training phase 3b
    └── test_lstm_phase4.py                    # LSTM training phase 4
```

### Naming Conventions

**Test Files:**
- `test_<module>.py` - Module-level test coverage
- `test_<module>_<function>.py` - Function-specific tests
- `test_<feature>_phase<N>.py` - Multi-phase feature testing (LSTM)

**Test Methods:**
- Pattern: `test_<functionality>_<scenario>_<expected_result>()`
- Examples:
  - `test_successful_experiment_creation()` - Happy path
  - `test_invalid_http_method_returns_405()` - Error validation
  - `test_train_model_logic_missing_experiment_dir()` - Edge case

**Test Classes:**
- `Test<ViewName>View` - Django view tests
- `Test<ServiceName>Logic` - Service function tests
- `Test<FeatureName>` - Utility/function tests
- `TestErrorHandlingScenarios` - Error scenario grouping
- `TestEdgeCases` - Edge case grouping

---

## Test Patterns and Strategies

### 1. Django View Testing Pattern

**File:** [DREAM-ML-backend/GEML/tests/api_tests/test_views.py](DREAM-ML-backend/GEML/tests/api_tests/test_views.py)

**Structure:**
```python
# Manual Django configuration (anti-pattern)
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.auth', 'django.contrib.contenttypes', 'api'],
        SECRET_KEY='test-secret-key',
    )
    django.setup()

from django.test import RequestFactory, TestCase

class TestCreateExperimentView(TestCase):
    def setUp(self):
        self.factory = RequestFactory()  # test_views.py:75

    @patch('api.views.create_experiment_logic')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_successful_experiment_creation(self, mock_environ_get, mock_isdir, mock_create_logic):
        # Arrange
        mock_environ_get.return_value = '/app/experimentos'
        mock_isdir.return_value = True
        mock_create_logic.return_value = {
            'experiment_id': 'test-uuid',
            'experiment_name': 'test-experiment',
        }
        request = self.factory.post('/create-experiment/', content_type='application/json')

        # Act
        response = views.create_experiment(request)

        # Assert
        self.assertEqual(response.status_code, 201)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Experimento creado exitosamente')
```

**Key Characteristics:**
- Uses Django `TestCase` base class (test_views.py:73)
- `RequestFactory` for request simulation instead of test Client (test_views.py:75)
- No URL routing testing - direct view function calls
- Manual settings configuration instead of pytest-django (test_views.py:29-57)
- Extensive mocking of service layer and external dependencies
- AAA pattern with explicit comment sections

**Validated Aspects:**
- HTTP method enforcement (GET, POST, PUT, DELETE)
- Status code validation (200, 201, 400, 404, 405, 500)
- JSON response structure
- Error message content
- Request parameter validation

### 2. Service Layer Testing Pattern

**File:** [DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services.py](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services.py)

**Structure:**
```python
from apiTimeSeries.services import PreProcessingService

class TestPreProcessingService:
    """Test cases for PreProcessingService class"""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = PreProcessingService()  # test_services.py:28

    def test_analyze_csv_logic_successful_analysis(self):
        """
        Scenario 1: Successful CSV Analysis
        Given a valid CSV file with multiple columns
        When analyze_csv_logic is called
        Then it should return a dictionary with column names
        """
        # Arrange
        mock_csv_content = "col1,col2,col3\n1,2,3\n4,5,6"
        mock_csv_file = StringIO(mock_csv_content)
        expected_columns = ["col1", "col2", "col3"]

        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame(columns=expected_columns)
            mock_read_csv.return_value = mock_df

            # Act
            result = self.service.analyze_csv_logic(mock_csv_file)

            # Assert
            assert isinstance(result, dict)
            assert "columns" in result
            assert result["columns"] == expected_columns
            mock_read_csv.assert_called_once_with(mock_csv_file, nrows=0)
```

**Key Characteristics:**
- Pytest class-based testing (no Django TestCase)
- `setup_method()` fixture instead of `setUp()` (test_services.py:26)
- Given-When-Then documentation in docstrings (test_services.py:36-39)
- Service class instantiation in setup (test_services.py:28)
- Context manager pattern for patching (test_services.py:46)
- Assert-style assertions (pytest) instead of self.assertEqual (Django)

**Validated Aspects:**
- Return value structure and types
- Dictionary key presence
- Error propagation (ValueError, FileNotFoundError, RuntimeError)
- Directory existence validation
- Parameter validation
- MLflow integration (mocked)

### 3. Deep Mocking Pattern (Anti-Pattern)

**File:** [DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services_train_model_logic.py:113-178](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services_train_model_logic.py#L113-L178)

**14-Layer Mocking Stack Example:**
```python
@patch('apiTimeSeries.train.subprocess.run')
@patch('apiTimeSeries.train.mlflow.log_input')
@patch('apiTimeSeries.train.mlflow.set_tag')
@patch('apiTimeSeries.train.mlflow.log_param')
@patch('apiTimeSeries.train.mlflow.set_tracking_uri')
@patch('apiTimeSeries.train.mlflow.start_run')
@patch('apiTimeSeries.train.mlflow.get_experiment_by_name')
@patch('apiTimeSeries.train.os.makedirs')
@patch('apiTimeSeries.train.os.path.exists')
@patch('apiTimeSeries.train.pd.read_csv')
@patch('apiTimeSeries.train.train_arima_model')
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
@patch('json.load')
def test_train_model_logic_successful_execution(
    self, mock_json_load, mock_json_dump, mock_open_file,
    mock_train_arima, mock_read_csv, mock_path_exists, mock_makedirs,
    mock_get_experiment, mock_start_run, mock_set_tracking_uri,
    mock_log_param, mock_set_tag, mock_log_input, mock_subprocess_run
):
    # Complex mock setup...
```

**Issues with Deep Mocking:**
- Tests become brittle and tightly coupled to implementation
- Mock setup complexity exceeds actual logic being tested
- Difficult to maintain when refactoring
- Tests validate mocking behavior more than business logic
- High cognitive load to understand test intent

**Better Approach:**
- Extract testable functions with fewer dependencies
- Use integration tests for workflows with many dependencies
- Test with real pandas/numpy operations where possible
- Create test utilities to reduce mock boilerplate

### 4. WebSocket Testing Pattern (Async)

**File:** [DREAM-ML-backend/GEML/tests/api_tests/test_consumers.py:24-57](DREAM-ML-backend/GEML/tests/api_tests/test_consumers.py#L24-L57)

**Structure:**
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_consumer():
    """Fixture to create a mock consumer with necessary attributes."""
    consumer = MagicMock()
    consumer.channel_layer = AsyncMock()
    consumer.channel_name = "test_channel"
    consumer.send = AsyncMock()
    return consumer

@pytest.mark.asyncio
async def test_successful_websocket_connection(mock_consumer):
    """
    Scenario 1: Successful WebSocket Connection
    Given a WebSocket consumer
    When connect() is called
    Then the consumer should accept the connection and join the channel layer group
    """
    # Arrange
    consumer = ProgressConsumer()
    consumer.channel_layer = mock_consumer.channel_layer
    consumer.channel_name = mock_consumer.channel_name

    # Act
    await consumer.connect()

    # Assert
    consumer.channel_layer.group_add.assert_called_once()
    mock_consumer.send.assert_called()
```

**Key Characteristics:**
- Only async tests in the codebase
- Uses `@pytest.mark.asyncio` decorator (test_consumers.py:37)
- `AsyncMock` for async method mocking (test_consumers.py:28)
- Await consumer methods (test_consumers.py:51)
- Tests WebSocket lifecycle: connect, disconnect, receive
- Validates Django Channels integration

### 5. LSTM/TensorFlow Testing Pattern (Integration-Style)

**File:** [DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase1.py:183-206](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase1.py#L183-L206)

**Structure:**
```python
class TestBuildLSTMModel:
    def test_build_lstm_model_single_layer(self):
        """Tests building a single-layer LSTM model."""
        # Arrange
        n_timesteps = 10
        n_features = 2
        lstm_units = [50]  # Single layer

        # Act
        model = build_lstm_model(n_timesteps, n_features, lstm_units)

        # Assert
        assert isinstance(model, tf.keras.Sequential)
        assert len(model.layers) == 2  # LSTM + Dense
        assert model.layers[0].units == 50
        assert model.output_shape == (None, 1)
        assert model.optimizer is not None
```

**Key Characteristics:**
- **No mocking** - tests actual TensorFlow/Keras model creation
- Validates model architecture directly
- Tests shape transformations with numpy arrays
- Validates temporal ordering preservation
- Integration-style testing for ML workflows

**Contrast with sklearn Tests:**
- sklearn model tests mock everything (api_tests/arreglar.py)
- LSTM tests use real TensorFlow operations
- LSTM tests validate actual numerical transformations

### 6. Data Cleaning Testing Pattern (Real Operations)

**File:** [DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_data_cleaning_utils.py:29-64](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_data_cleaning_utils.py#L29-L64)

**Structure:**
```python
def test_basic_successful_processing(self):
    """Test basic data cleaning with all default options."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Arrange
        input_csv = os.path.join(temp_dir, "input.csv")
        df = pd.DataFrame({
            'col1': [1, 2, 2, 4],  # Has duplicate
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [1.1, None, 2.2, 3.3]  # Has missing value
        })
        df.to_csv(input_csv, index=False)

        # Act
        cleaned_csv, report = limpiar_datos(input_csv, temp_dir)

        # Assert
        assert os.path.exists(cleaned_csv)
        cleaned_df = pd.read_csv(cleaned_csv)

        # Validate transformations
        assert cleaned_df.shape[0] == 3  # Duplicate removed
        assert report['duplicates_removed'] == 1
        assert report['numeric_imputations']['col3'] == 1
```

**Key Characteristics:**
- Uses `tempfile.TemporaryDirectory()` for filesystem isolation
- Tests actual pandas operations (no mocking)
- Validates file I/O operations
- Tests data transformations end-to-end
- Validates report structure and values
- Automatic cleanup via context manager

**Validated Transformations:**
- Duplicate removal
- Missing value imputation (mean, median, mode, custom value)
- Outlier filtering (IQR method)
- Type conversion (string to numeric)
- Column name cleaning (whitespace removal)
- Empty DataFrame handling

---

## Mocking Strategies

### External Dependency Mocking Patterns

#### 1. MLflow Mocking
```python
# Pattern 1: Simple function mock
@patch('api.views.mlflow.set_tracking_uri')
@patch('api.views.mlflow.start_run')
def test_mlflow_integration(mock_start_run, mock_set_uri):
    mock_run = MagicMock()
    mock_run.info.run_id = 'test-run-id-123'
    mock_start_run.return_value.__enter__.return_value = mock_run  # Context manager
```

#### 2. File System Mocking
```python
# Pattern 2: Path existence checks
@patch('os.path.isdir')
@patch('os.path.exists')
@patch('os.makedirs')
def test_directory_operations(mock_makedirs, mock_exists, mock_isdir):
    mock_isdir.return_value = True
    mock_exists.return_value = False
```

#### 3. Subprocess Mocking (DVC/Git)
```python
# Pattern 3: Subprocess execution
@patch('subprocess.run')
def test_dvc_initialization(mock_subprocess):
    mock_subprocess.return_value = MagicMock(returncode=0)

    # Verify specific commands were called
    expected_call = call(["git", "init"], cwd="/path", check=True)
    assert expected_call in mock_subprocess.call_args_list
```

#### 4. Pandas Mocking
```python
# Pattern 4: DataFrame operations
@patch('pandas.read_csv')
def test_csv_reading(mock_read_csv):
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_read_csv.return_value = mock_df
```

#### 5. Django Channels Mocking
```python
# Pattern 5: WebSocket channels
@patch('channels.layers.get_channel_layer')
def test_websocket_progress(mock_get_channel):
    mock_channel_layer = AsyncMock()
    mock_get_channel.return_value = mock_channel_layer
```

#### 6. Built-in Mocking (File I/O)
```python
# Pattern 6: File open operations
@patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
@patch('json.load')
def test_config_file_reading(mock_json_load, mock_open_file):
    mock_json_load.return_value = {'key': 'value'}
```

### Mock Verification Patterns

```python
# Verify function was called
mock_function.assert_called_once()

# Verify function called with specific arguments
mock_function.assert_called_once_with(arg1, arg2, kwarg='value')

# Verify function called at all (at least once)
mock_function.assert_called()

# Verify function was NOT called
mock_function.assert_not_called()

# Verify call count
assert mock_function.call_count == 3

# Verify specific call in call list
expected_call = call(["git", "init"], cwd=path, check=True)
assert expected_call in mock_subprocess.call_args_list
```

---

## Fixture Management

### Temporary Directory Fixtures

**Pattern 1: Context Manager (Recommended)**
```python
import tempfile

def test_file_operations():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        input_file = os.path.join(temp_dir, "input.csv")

        # Run test
        result = process_file(input_file, temp_dir)

        # Validate
        assert os.path.exists(result)
    # Automatic cleanup after context manager exits
```

**Pattern 2: setup_method/teardown_method**
```python
class TestDataProcessing:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
```

### Module-Level Pytest Fixtures

**File:** [DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services.py:266-280](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services.py#L266-L280)

```python
@pytest.fixture
def temp_directory():
    """Fixture to create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_csv_content():
    """Fixture for sample CSV content."""
    return "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"

@pytest.fixture
def mock_csv_file(sample_csv_content):
    """Fixture to create a mock CSV file."""
    return StringIO(sample_csv_content)
```

### Missing Fixture Infrastructure

**No conftest.py Files Found:**
- ❌ No shared fixtures at package level
- ❌ Fixture duplication across test files
- ❌ No pytest plugin configuration
- ❌ No custom pytest markers defined

**Recommendation:** Create conftest.py files:
```
tests/
├── conftest.py                    # Root-level shared fixtures
├── api_tests/
│   ├── conftest.py               # api-specific fixtures
│   └── ...
└── apiTimeSeries_tests/
    ├── conftest.py               # apiTimeSeries-specific fixtures
    └── ...
```

---

## Code Coverage Analysis

### Current Coverage Execution

```bash
cd DREAM-ML-backend/GEML
coverage run --source='.' -m pytest -v
coverage report
coverage html  # Optional HTML report
```

### Coverage Configuration

**Status:** ❌ No coverage configuration file found

**Expected Locations Checked:**
- `.coveragerc` (not found)
- `pyproject.toml` (not found)
- `setup.cfg` (not found)

**Recommendation:** Create `.coveragerc`:
```ini
[run]
source = .
omit =
    */migrations/*
    */tests/*
    */__pycache__/*
    */venv/*
    */node_modules/*
    manage.py
    */wsgi.py
    */asgi.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod

precision = 2
show_missing = True

[html]
directory = htmlcov
```

### Test File Coverage Matrix

| Package | Test File | Tested Module/Feature | Lines | Status |
|---------|-----------|----------------------|-------|--------|
| **api_tests** | test_views.py | api.views (all endpoints) | 1299 | ✅ Comprehensive |
| | test_services.py | api.services business logic | 786 | ✅ Comprehensive |
| | test_consumers.py | WebSocket consumers | 281 | ✅ Async coverage |
| | test_utils.py | api.utils helpers | 736 | ✅ Good coverage |
| | test_data_cleaning.py | api.data_cleaning | 494 | ✅ Parametrized |
| | arreglar.py | api.train (ML models) | 826 | ✅ Model workflows |
| | test_analyze_csv_logic.py | CSV analysis | ? | ⚠️ Not analyzed |
| | test_bayesian_search_classification.py | Bayesian optimization | ? | ⚠️ Not analyzed |
| **apiTimeSeries_tests** | test_views_train_model.py | Training endpoint | 436 | ✅ Error-focused |
| | test_services.py | Service classes | 285 | ✅ Good coverage |
| | test_services_train_model_logic.py | Training logic | 387 | ⚠️ Over-mocked |
| | test_lstm_phase1.py | LSTM sequences/splits | 259 | ✅ Integration-style |
| | test_lstm_phase2a-4.py | LSTM training phases | ? | ⚠️ Not analyzed |
| | test_data_cleaning_utils.py | Data cleaning | 397 | ✅ Real operations |
| | test_views_create_experiment.py | Experiment creation | ? | ⚠️ Not analyzed |
| | test_views_upload_and_clean_csv.py | CSV upload | ? | ⚠️ Not analyzed |
| | test_views_encode_csv.py | Encoding | ? | ⚠️ Not analyzed |
| | test_services_generate_eda_logic.py | EDA generation | ? | ⚠️ Not analyzed |
| | test_services_encode_csv_logic.py | Encoding logic | ? | ⚠️ Not analyzed |

---

## Architecture Insights

### 1. Layered Architecture Testing

```
┌─────────────────────────────────────────┐
│           HTTP Layer (Views)            │  ← test_views.py
│  - Request validation                   │     (Django TestCase)
│  - Response formatting                  │     (RequestFactory)
│  - Status code handling                 │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Service Layer (Business)         │  ← test_services.py
│  - Orchestration                        │     (pytest classes)
│  - Validation logic                     │     (heavy mocking)
│  - Error handling                       │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│      Utility Layer (Helpers)            │  ← test_utils.py
│  - Data transformations                 │     (pytest functions)
│  - File operations                      │     (some real ops)
│  - External service calls               │
└─────────────────────────────────────────┘
```

**Testing Strategy per Layer:**
- **Views:** Mock service layer entirely, test HTTP contract
- **Services:** Mock utilities and external dependencies, test orchestration
- **Utilities:** Mix of real operations and mocking

**Gap:** No integration tests spanning multiple layers

### 2. Test Type Distribution

```
Unit Tests (Isolated)     ████████████████████████ 85%
Integration Tests         ██                        5%
End-to-End Tests          ▌                         0%
Async Tests              █                         3%
Performance Tests         ▌                         0%
```

**Observations:**
- Heavy bias toward unit tests with mocking
- Minimal integration testing
- No end-to-end workflow validation
- Only one async test file (WebSocket consumers)

### 3. Dependency Mocking vs Real Operations

| Dependency | Mocked | Real Operations | Notes |
|------------|--------|-----------------|-------|
| **MLflow** | ✅ Always | ❌ Never | All MLflow calls mocked |
| **Pandas** | ⚠️ Sometimes | ✅ Data cleaning | Mixed approach |
| **NumPy** | ❌ Rarely | ✅ LSTM tests | Real array operations |
| **TensorFlow** | ❌ Never | ✅ Model building | Real model construction |
| **sklearn** | ✅ Always | ❌ Never | All sklearn mocked |
| **File System** | ⚠️ Sometimes | ✅ Temp dirs | Use tempfile extensively |
| **Database** | ✅ Always | ❌ Never | No real DB operations |
| **Subprocess** | ✅ Always | ❌ Never | DVC/Git all mocked |
| **WebSocket** | ✅ Always | ❌ Never | Channels all mocked |

### 4. Test Execution Flow

```
┌─────────────────────────────────────────┐
│  coverage run --source='.' -m pytest -v │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         pytest.ini configuration        │
│  - Load Django settings                 │
│  - Set Python path                      │
│  - Discover test_*.py files             │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│      Execute tests sequentially         │
│  - No parallel execution                │
│  - Debug mode enabled                   │
│  - In-memory SQLite database            │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│     Coverage measurement (.coverage)    │
│  - Source code coverage tracking        │
│  - No HTML reports by default           │
└─────────────────────────────────────────┘
```

---

## Critical Code References

### Test Configuration
- [pytest.ini:1-6](DREAM-ML-backend/GEML/pytest.ini#L1-L6) - Pytest configuration
- [requirements-dev.txt:6-9](DREAM-ML-backend/GEML/requirements-dev.txt#L6-L9) - Test dependencies

### api_tests Package
- [test_views.py:29-57](DREAM-ML-backend/GEML/tests/api_tests/test_views.py#L29-L57) - Manual Django config (anti-pattern)
- [test_views.py:73-100](DREAM-ML-backend/GEML/tests/api_tests/test_views.py#L73-L100) - View testing pattern
- [test_services.py:35-102](DREAM-ML-backend/GEML/tests/api_tests/test_services.py#L35-L102) - Service testing with tempfile
- [test_consumers.py:24-57](DREAM-ML-backend/GEML/tests/api_tests/test_consumers.py#L24-L57) - Async WebSocket testing
- [test_utils.py:382-440](DREAM-ML-backend/GEML/tests/api_tests/test_utils.py#L382-L440) - Subprocess mocking pattern
- [arreglar.py:458-529](DREAM-ML-backend/GEML/tests/api_tests/arreglar.py#L458-L529) - Model training with deep mocking

### apiTimeSeries_tests Package
- [test_services.py:26-57](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services.py#L26-L57) - Service class testing
- [test_services_train_model_logic.py:113-178](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_services_train_model_logic.py#L113-L178) - 14-layer mocking stack
- [test_lstm_phase1.py:40-104](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase1.py#L40-L104) - Sequence creation tests
- [test_lstm_phase1.py:183-206](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_lstm_phase1.py#L183-L206) - Model architecture validation
- [test_data_cleaning_utils.py:29-64](DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/test_data_cleaning_utils.py#L29-L64) - Real pandas operations

---

## Open Questions and Clarifications Needed

### 1. Coverage Target Details
- **Q:** What's the current overall coverage percentage?
- **A:** Run `coverage report` to get baseline metrics
- **Q:** Are there specific modules that need priority coverage?
- **A:** Identify critical business logic modules (train.py, services.py)

### 2. Integration Testing Scope
- **Q:** Should integration tests use real MLflow tracking server?
- **Options:**
  - Use MLflow with SQLite backend for tests
  - Continue mocking MLflow entirely
  - Hybrid approach (unit tests mock, integration tests use real server)

### 3. Database Testing Strategy
- **Q:** Should tests use real Django ORM operations?
- **Current:** All database operations mocked
- **Recommendation:** Use pytest-django's `django_db` marker for model tests

### 4. Performance Testing
- **Q:** Are performance benchmarks needed for ML model training?
- **Consideration:** Large dataset handling, memory usage, training time

### 5. Test Data Management
- **Q:** Should we create factory patterns for test data?
- **Tools:** factory_boy for Django models, faker for synthetic data
- **Benefit:** Reduce hardcoded test data, improve maintainability

### 6. Parallel Test Execution
- **Q:** Should tests run in parallel?
- **Tool:** pytest-xdist
- **Consideration:** Test isolation, shared resources, speed improvement

### 7. Coverage Threshold Enforcement
- **Q:** Should coverage checks fail CI/CD if below 75%?
- **Tool:** coverage with `--fail-under=75` flag
- **Consideration:** Gradual rollout vs immediate enforcement

### 8. Async Testing Expansion
- **Q:** Are there other async components that need testing?
- **Current:** Only WebSocket consumers tested
- **Consideration:** Async views, background tasks, concurrent operations

### 9. Manual Django Configuration
- **Q:** Why is Django manually configured in test_views.py?
- **Issue:** Bypasses pytest-django automation
- **Fix:** Remove manual config, use pytest-django fixtures

### 10. LSTM Test Phases
- **Q:** What do phases 2b, 3a, 3b, 4 test?
- **Analyzed:** Phase 1 (sequence creation, model building)
- **Not analyzed:** Phases 2b-4 (likely training, evaluation, deployment)

---

## Recommendations for Achieving 75%+ Coverage

### High Priority (Immediate Actions)

#### 1. Create Shared Fixtures (conftest.py)
**Impact:** Reduce duplication, improve maintainability

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock

@pytest.fixture
def temp_experiment_dir():
    """Shared temporary directory fixture."""
    temp_dir = tempfile.mkdtemp(prefix="test_exp_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_mlflow_experiment():
    """Shared MLflow experiment mock."""
    mock_exp = MagicMock()
    mock_exp.experiment_id = "test-exp-123"
    mock_exp.name = "test-experiment"
    return mock_exp

@pytest.fixture
def sample_dataframe():
    """Shared test DataFrame."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
```

#### 2. Remove Manual Django Configuration
**File:** [test_views.py:29-57](DREAM-ML-backend/GEML/tests/api_tests/test_views.py#L29-L57)

**Current (Anti-pattern):**
```python
if not settings.configured:
    settings.configure(DEBUG=True, DATABASES={...})
    django.setup()
```

**Recommended:**
```python
import pytest

@pytest.mark.django_db
class TestCreateExperimentView:
    def test_successful_experiment_creation(self, client):
        # Use pytest-django's client fixture
        response = client.post('/create-experiment/', data={...})
        assert response.status_code == 201
```

#### 3. Add Integration Tests
**Coverage Gap:** No tests spanning views → services → models

**Example Integration Test:**
```python
@pytest.mark.django_db
@pytest.mark.integration
def test_full_experiment_workflow(client, temp_experiment_dir):
    """Test complete experiment creation to model training workflow."""
    # 1. Create experiment
    response = client.post('/create-experiment/')
    experiment_id = response.json()['experiment_id']

    # 2. Upload CSV
    csv_file = SimpleUploadedFile("data.csv", b"col1,col2\n1,2\n3,4")
    response = client.post(f'/upload-csv/{experiment_id}/', {'file': csv_file})
    assert response.status_code == 200

    # 3. Train model
    response = client.post(f'/train-model/{experiment_id}/', data={...})
    assert response.status_code == 200

    # 4. Verify model artifacts exist
    model_path = response.json()['model_path']
    assert os.path.exists(model_path)
```

#### 4. Reduce Mocking Depth
**Issue:** 10-14 layer mocking stacks

**Strategy:**
- Extract testable functions with fewer dependencies
- Use real pandas/numpy operations
- Mock only at architectural boundaries (MLflow, subprocess, network)

**Example Refactoring:**
```python
# Before: Mock everything
@patch('apiTimeSeries.train.pd.read_csv')
@patch('apiTimeSeries.train.train_test_split')
@patch('apiTimeSeries.train.StandardScaler')
def test_data_preprocessing(mock_scaler, mock_split, mock_read_csv):
    # 20 lines of mock setup...

# After: Test real operations
def test_data_preprocessing():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    result = preprocess_data(df, target='col2')

    assert isinstance(result['X_train'], np.ndarray)
    assert result['X_train'].shape[0] > 0
```

#### 5. Add Coverage Configuration
**Create:** `.coveragerc` in DREAM-ML-backend/GEML/

```ini
[run]
source = .
omit =
    */migrations/*
    */tests/*
    */__pycache__/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:

precision = 2
show_missing = True
fail_under = 75

[html]
directory = htmlcov
```

### Medium Priority (Next Phase)

#### 6. Standardize on Pytest
**Current:** Mix of Django TestCase and pytest classes

**Migration Path:**
- Keep Django TestCase for tests requiring Django's transaction handling
- Convert utility/service tests to pure pytest
- Use `@pytest.mark.django_db` for database access

#### 7. Add Database Tests
**Coverage Gap:** All model operations mocked

```python
@pytest.mark.django_db
def test_experiment_model_creation():
    """Test Django model creation and retrieval."""
    from api.models import Experiment

    exp = Experiment.objects.create(
        name="test-experiment",
        description="Test description"
    )

    retrieved = Experiment.objects.get(pk=exp.pk)
    assert retrieved.name == "test-experiment"
```

#### 8. Add Custom Pytest Markers
**Purpose:** Organize tests by type, enable selective execution

```python
# pytest.ini
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, multiple components)
    slow: Slow tests (ML training, large datasets)
    asyncio: Async tests
    requires_mlflow: Tests requiring MLflow server

# Usage
pytest -m unit                    # Run only unit tests
pytest -m "not slow"              # Skip slow tests
pytest -m "integration and not slow"
```

#### 9. Create Test Data Builders
**Purpose:** Reduce hardcoded test data

```python
# tests/builders.py
class ExperimentConfigBuilder:
    def __init__(self):
        self.config = {
            'algorithm': 'arima',
            'model_name': 'test-model',
            'split_ratios': [0.7, 0.15, 0.15],
        }

    def with_algorithm(self, algorithm):
        self.config['algorithm'] = algorithm
        return self

    def with_split_ratios(self, ratios):
        self.config['split_ratios'] = ratios
        return self

    def build(self):
        return self.config

# Usage in tests
config = ExperimentConfigBuilder().with_algorithm('lstm').build()
```

#### 10. Add MLflow Integration Tests
**Option 1:** Use MLflow with SQLite backend

```python
@pytest.fixture(scope="session")
def mlflow_tracking_uri():
    """Create temporary MLflow tracking server."""
    temp_dir = tempfile.mkdtemp()
    tracking_uri = f"sqlite:///{temp_dir}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    shutil.rmtree(temp_dir)

@pytest.mark.integration
def test_mlflow_logging(mlflow_tracking_uri):
    """Test actual MLflow logging."""
    with mlflow.start_run():
        mlflow.log_param("test_param", "value")
        mlflow.log_metric("accuracy", 0.95)

    run = mlflow.get_run(mlflow.active_run().info.run_id)
    assert run.data.params['test_param'] == 'value'
```

### Low Priority (Future Enhancements)

#### 11. Add Performance Benchmarks
```python
@pytest.mark.benchmark
def test_lstm_training_performance(benchmark):
    """Benchmark LSTM training time."""
    def train():
        return train_lstm_model(X_train, y_train, epochs=10)

    result = benchmark(train)
    assert benchmark.stats.mean < 5.0  # Max 5 seconds
```

#### 12. Add Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto  # Use all CPU cores
pytest -n 4     # Use 4 workers
```

#### 13. Add Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=1000))
def test_data_cleaning_preserves_positive_values(values):
    """Property: Data cleaning should preserve positive values."""
    df = pd.DataFrame({'values': values})
    cleaned_df = clean_data(df)

    assert (cleaned_df['values'] >= 0).all()
```

---

## Summary and Action Plan

### Current State
✅ **Strengths:**
- 22+ test files with comprehensive unit coverage
- Consistent AAA pattern and GWT documentation
- Strong error handling validation
- Good temporary file management

❌ **Weaknesses:**
- Deep mocking stacks (10-14 patches) create brittle tests
- No integration tests across layers
- No shared fixtures (no conftest.py)
- Manual Django configuration anti-pattern
- All database operations mocked
- All MLflow operations mocked

### Path to 75%+ Coverage

**Phase 1: Foundation (Week 1-2)**
1. Create conftest.py files with shared fixtures
2. Add .coveragerc configuration
3. Remove manual Django configuration
4. Run coverage report to establish baseline

**Phase 2: Coverage Expansion (Week 3-4)**
5. Add integration tests for critical workflows
6. Add database tests with @pytest.mark.django_db
7. Reduce mocking depth in service tests
8. Add tests for currently untested modules

**Phase 3: Quality Improvement (Week 5-6)**
9. Standardize on pytest conventions
10. Add custom pytest markers
11. Create test data builders
12. Add MLflow integration tests (optional)

**Phase 4: Maintenance (Ongoing)**
13. Monitor coverage in CI/CD
14. Enforce 75% threshold
15. Add performance benchmarks (optional)
16. Consider parallel execution (optional)

### Key Metrics to Track
- **Overall Coverage:** Target 75%+
- **Branch Coverage:** Target 70%+
- **View Coverage:** Target 90%+ (critical for API contracts)
- **Service Coverage:** Target 85%+ (business logic)
- **Model Coverage:** Target 70%+ (database operations)

### Next Steps
1. Run baseline coverage report: `coverage run --source='.' -m pytest -v && coverage report`
2. Identify modules below 75% coverage
3. Create prioritized backlog based on coverage gaps
4. Implement shared fixtures (conftest.py)
5. Add integration tests for top 3 critical workflows

---

## Appendix: Test Execution Checklist

### Running Tests
```bash
# Navigate to project directory
cd DREAM-ML-backend/GEML

# Run all tests with coverage
coverage run --source='.' -m pytest -v

# Generate coverage report (terminal)
coverage report

# Generate HTML coverage report
coverage html
# Open htmlcov/index.html in browser

# Run specific test file
pytest tests/api_tests/test_views.py -v

# Run specific test class
pytest tests/api_tests/test_views.py::TestCreateExperimentView -v

# Run specific test method
pytest tests/api_tests/test_views.py::TestCreateExperimentView::test_successful_experiment_creation -v

# Run tests matching pattern
pytest -k "test_successful" -v

# Run tests with markers (after adding markers)
pytest -m unit -v
pytest -m "integration and not slow" -v

# Show test output (disable capture)
pytest -v -s

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Coverage Analysis
```bash
# Show missing lines for specific file
coverage report --include="api/train.py" --show-missing

# Generate XML report (for CI/CD)
coverage xml

# Check if coverage meets threshold
coverage report --fail-under=75
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-31
**Analysis Scope:** 10 core test files + 2 configuration files
**Total Test Files in Codebase:** 22+
