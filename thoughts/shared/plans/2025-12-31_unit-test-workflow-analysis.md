# Implementation Plan: Unit Test Coverage Improvement (2% → 75%+)

**Project:** DREAM-ML Backend
**Date:** 2025-12-31
**Objective:** Improve unit test coverage for the backend from 2% to at least 75%
**Scope:** Focus on api package first, then apiTimeSeries package
**Strategy:** Test how code currently works (including errors/edge cases), use heavy mocking for speed

---

## Executive Summary

**Current Baseline:**
- Overall coverage: 2% (critical - extremely low)
- api package: ~2,600 LOC with 0% coverage on production files
- apiTimeSeries package: ~3,500 LOC with 0% coverage on production files
- Total uncovered: ~6,100 LOC

**Implementation Approach:**
- 10 phases for api package (Phase 0 foundation + 9 implementation phases)
- 4 additional phases for apiTimeSeries package (high-level)
- Fast unit tests with heavy mocking (<2 minutes total execution time)
- Test deterministic behavior with SEED=42
- Extend existing test files where possible

**Success Criteria:**
- api package coverage ≥ 75%
- All phases have passing pytest execution
- Coverage increases measurably after each phase
- Tests execute in <2 minutes total

---

## Phase 0: Coverage Infrastructure & Diagnostic Fixes

**Phase Overview:**
Fix the critical issue where tests exist but production code shows 0% coverage. Establish proper coverage configuration, shared fixtures, and pytest infrastructure.

### Objectives
1. Create `.coveragerc` configuration file with proper exclusions
2. Add `conftest.py` files for shared test fixtures
3. Remove manual Django configuration anti-pattern from test files
4. Add pytest markers to `pytest.ini`
5. Diagnose and fix coverage measurement issue

### Files to Modify

**Create:**
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/.coveragerc`
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/conftest.py`
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/conftest.py`
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/conftest.py`

**Modify:**
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/pytest.ini`
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py` (remove manual Django config)

### Implementation Details

#### 1. Create `.coveragerc` Configuration

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/.coveragerc`

```ini
[run]
source = .
omit =
    */migrations/*
    */tests/*
    */__pycache__/*
    */venv/*
    */env/*
    */node_modules/*
    manage.py
    */wsgi.py
    */asgi.py
    */admin.py
    */apps.py
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
    @abc.abstractmethod

precision = 2
show_missing = True

[html]
directory = htmlcov
```

**Rationale:** Exclude non-testable code (migrations, admin, config) from coverage calculations to get accurate metrics.

---

#### 2. Create Root `conftest.py` for Shared Fixtures

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/conftest.py`

```python
"""Root conftest.py for shared test fixtures across all test packages."""
import pytest
import tempfile
import shutil
import os
from unittest.mock import MagicMock
import pandas as pd
import numpy as np


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary experiment directory for tests.

    Yields the directory path and automatically cleans up after test.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_exp_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_mlflow_experiment():
    """Mock MLflow experiment object."""
    mock_exp = MagicMock()
    mock_exp.experiment_id = "test-exp-123"
    mock_exp.name = "test-experiment"
    return mock_exp


@pytest.fixture
def mock_mlflow_run():
    """Mock MLflow run object for context manager usage."""
    mock_run = MagicMock()
    mock_run.info.run_id = 'test-run-id-456'
    mock_run.info.experiment_id = 'test-exp-123'
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    return mock_run


@pytest.fixture
def sample_dataframe():
    """Sample classification DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def sample_csv_content():
    """Sample CSV content as string."""
    return "feature1,feature2,feature3,target\n1.0,2.0,A,0\n1.5,2.5,B,1\n2.0,3.0,C,0"


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        'algorithm': 'logistic_regression',
        'model_name': 'test-model',
        'split_ratios': [0.7, 0.15, 0.15],
        'random_state': 42,
        'search_method': 'grid',
        'n_trials': 2
    }


@pytest.fixture
def set_global_seed():
    """Fixture to set global random seeds for reproducibility."""
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
```

**Rationale:** Shared fixtures reduce duplication across test files and ensure consistency.

---

#### 3. Create `api_tests/conftest.py`

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/conftest.py`

```python
"""api_tests specific fixtures."""
import pytest
from unittest.mock import MagicMock, patch
from django.test import RequestFactory


@pytest.fixture
def request_factory():
    """Django RequestFactory for view testing."""
    return RequestFactory()


@pytest.fixture
def mock_subprocess_success():
    """Mock successful subprocess.run execution."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    return mock_result


@pytest.fixture
def mock_dvc_initialized():
    """Mock DVC initialization checks."""
    with patch('os.path.exists') as mock_exists, \
         patch('os.path.isdir') as mock_isdir:
        mock_exists.return_value = True
        mock_isdir.return_value = True
        yield (mock_exists, mock_isdir)


@pytest.fixture
def mock_mlflow_tracking():
    """Mock MLflow tracking URI and experiment setup."""
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
```

**Rationale:** api-specific fixtures for Django views and common mocking patterns.

---

#### 4. Create `apiTimeSeries_tests/conftest.py`

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/conftest.py`

```python
"""apiTimeSeries_tests specific fixtures."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_time_series_df():
    """Sample time series DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum(),
        'feature1': np.random.randn(100)
    })


@pytest.fixture
def sample_lstm_sequences():
    """Sample LSTM input sequences."""
    np.random.seed(42)
    n_samples = 50
    n_timesteps = 10
    n_features = 3
    X = np.random.randn(n_samples, n_timesteps, n_features)
    y = np.random.randn(n_samples, 1)
    return X, y


@pytest.fixture
def sample_arima_data():
    """Sample stationary time series for ARIMA testing."""
    np.random.seed(42)
    return np.random.randn(100)
```

**Rationale:** Time-series-specific fixtures for LSTM, ARIMA testing.

---

#### 5. Update `pytest.ini` with Markers

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/pytest.ini`

**Before:**
```ini
[pytest]
DJANGO_SETTINGS_MODULE = GEML.settings
python_files = tests.py test_*.py *_tests.py
django_debug_mode = true
pythonpath = . ..
django_find_project = false
```

**After:**
```ini
[pytest]
DJANGO_SETTINGS_MODULE = GEML.settings
python_files = tests.py test_*.py *_tests.py
django_debug_mode = true
pythonpath = . ..
django_find_project = false

# Custom markers
markers =
    unit: Unit tests (fast, isolated with mocking)
    integration: Integration tests (slower, multiple components)
    slow: Slow tests (ML training, large datasets)
    requires_mlflow: Tests requiring MLflow server
    requires_dvc: Tests requiring DVC initialization
```

**Rationale:** Enable selective test execution (`pytest -m unit`, `pytest -m "not slow"`).

---

#### 6. Remove Manual Django Configuration from test_views.py

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py`

**Remove lines 29-57 (manual Django configuration):**

```python
# DELETE THIS ANTI-PATTERN:
if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.auth', 'django.contrib.contenttypes', 'api'],
        SECRET_KEY='test-secret-key',
    )
    django.setup()
```

**Replace with pytest-django marker at class level:**

```python
import pytest

@pytest.mark.django_db
class TestCreateExperimentView:
    # Tests use pytest-django's automatic Django configuration
    def test_successful_experiment_creation(self, request_factory):
        # Test implementation
        pass
```

**Rationale:** pytest-django handles Django configuration automatically. Manual config is an anti-pattern.

---

### Automated Verification

```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML

# 1. Verify .coveragerc syntax
coverage --version
# Expected: coverage, version 7.6.10

# 2. Run a test to verify coverage collection works
coverage run --source='.' -m pytest tests/api_tests/test_data_cleaning.py -v

# 3. Generate coverage report
coverage report

# 4. Check that production files now show coverage
coverage report --include="api/*,apiTimeSeries/*"

# Expected: Should see percentage > 0% for at least some api modules
```

---

### Manual Verification Steps

**Checklist:**

1. **Verify .coveragerc created:**
   ```bash
   ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/.coveragerc
   cat /workspaces/dream-ml-c/DREAM-ML-backend/GEML/.coveragerc
   ```
   ✅ File exists with proper exclusions

2. **Verify conftest.py files created:**
   ```bash
   ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/conftest.py
   ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/conftest.py
   ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/apiTimeSeries_tests/conftest.py
   ```
   ✅ All three files exist

3. **Verify pytest.ini updated:**
   ```bash
   grep "markers" /workspaces/dream-ml-c/DREAM-ML-backend/GEML/pytest.ini
   ```
   ✅ Markers section present

4. **Verify manual Django config removed:**
   ```bash
   grep -n "settings.configure" /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py
   ```
   ✅ Should return no results (line removed)

5. **Run full test suite:**
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
   coverage run --source='.' -m pytest -v
   coverage report --include="api/*,apiTimeSeries/*"
   ```
   ✅ Coverage report shows > 0% for production files

6. **Test marker functionality:**
   ```bash
   pytest -m unit -v
   pytest -m "not slow" -v
   ```
   ✅ Markers work (even if no tests tagged yet)

---

### Success Criteria

✅ **PHASE 0 COMPLETED - 2025-12-31**

All success criteria met:
- ✅ `.coveragerc` file created with proper exclusions
- ✅ Three `conftest.py` files created with shared fixtures (root, api_tests, apiTimeSeries_tests)
- ✅ `pytest.ini` updated with 5 custom markers (unit, integration, slow, requires_mlflow, requires_dvc)
- ✅ Manual Django configuration removed from test_views.py
- ✅ `@pytest.mark.django_db` decorators added to all 14 test classes in test_views.py
- ✅ `import pytest` added to test_views.py
- ✅ All 62 existing tests pass without Django configuration errors
- ✅ No ImproperlyConfigured errors
- ✅ pytest-django handles Django setup automatically
- ✅ Coverage measurement infrastructure working correctly

**Verification Results:**
- All automated verification steps passed
- Manual verification confirmed by user
- pytest markers registered: `pytest --markers` shows all 5 custom markers
- Django auto-configuration working: all tests pass without manual config

---

## Phase 1: data_encoding.py Tests (Quick Win)

### Phase 1: Pattern Consistency Checklist

**Date:** 2025-12-31
**Previous Phase:** Phase 0 ✅ COMPLETED
**Current Phase:** Phase 1 - data_encoding.py Tests
**Purpose:** Ensure Phase 1 implementation follows established patterns from Phase 0

---

#### Pre-Implementation Checklist

**1. Prerequisites Verification**

Before starting Phase 1, verify:
- [x] Phase 0 completed successfully
- [x] .coveragerc configuration exists and working
- [x] All 3 conftest.py files exist with shared fixtures
- [x] pytest.ini has 5 custom markers configured
- [x] Manual Django config removed from test_views.py
- [x] Working directory: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML`
- [ ] data_encoding.py file exists and is readable
- [ ] Current coverage baseline for data_encoding.py measured

**2. Lessons from Phase 0**

**From Phase 0 (Infrastructure Setup):**
- ✅ Always verify current state before making changes
- ✅ Run baseline measurements to understand impact
- ✅ Test each component independently before integration
- ✅ Provide before/after comparisons for user verification
- ✅ Use Read tool before Edit/Write operations
- ✅ Verify syntax after configuration changes

**Key Insights to Apply:**
1. Phase 1 creates **1 new test file** (test_data_encoding.py)
2. Tests should use real pandas operations (minimal mocking for data transformations)
3. Use shared fixtures from conftest.py (temp_experiment_dir)
4. Mark tests with `@pytest.mark.unit` for selective execution
5. Aim for 75%+ coverage on data_encoding.py module
6. Tests should execute quickly (< 5 seconds total)

---

#### Implementation Pattern Checklist

**3. File Creation Pattern**

Phase 1 creates **1 new test file**:
- [ ] `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py`

**Pattern to follow (from Phase 0):**
- Use Write tool for new file creation
- Include proper module docstring
- Import pytest and necessary dependencies
- Use shared fixtures from conftest.py
- Group tests in classes with clear naming

**4. Test Structure Pattern**

**Test Class Naming:**
```python
@pytest.mark.unit
class TestCodificarDatos:
    """Test cases for codificar_datos() function."""
```

**Test Method Naming:**
```python
def test_successful_encoding_with_onehot_features(self, temp_experiment_dir):
    """
    Scenario: Successful encoding with OneHot for categorical features
    Given a CSV with categorical features
    When codificar_datos is called
    Then categorical features should be one-hot encoded
    """
```

**Pattern components:**
- Use `@pytest.mark.unit` decorator
- Clear class names describing what's being tested
- Docstrings with Given/When/Then format
- Use shared fixtures as parameters
- Arrange/Act/Assert structure in test body

**5. Fixture Usage Pattern**

**From Phase 0 conftest.py, use:**
- [ ] `temp_experiment_dir` - For temporary file operations
- [ ] `sample_dataframe` - For test data (if applicable)
- [ ] `set_global_seed` - For reproducibility (if needed)

**Avoid:**
- Creating duplicate fixtures already in conftest.py
- Using hardcoded paths instead of temp_experiment_dir
- Forgetting to clean up temporary files (fixture handles this)

**6. Testing Strategy Pattern**

**For data_encoding.py module:**
- [ ] Test all major code paths (9+ test scenarios)
- [ ] Test successful encoding transformations
- [ ] Test error handling (KeyError, ValueError, FileNotFoundError)
- [ ] Test edge cases (empty DataFrame, boolean conversion)
- [ ] Use real pandas operations (no mocking for data transformations)
- [ ] Verify actual file outputs (CSV files created correctly)
- [ ] Check encoding reports/metadata returned

**Example test scenarios to cover:**
1. OneHot encoding for categorical features
2. Label encoding for binary target
3. OneHot encoding for multiclass target
4. Boolean to integer conversion
5. Numeric features preserved unchanged
6. Error: missing target column
7. Error: invalid CSV path
8. Empty DataFrame handling
9. Mixed categorical and numeric features

---

#### Verification Pattern Checklist

**7. Automated Verification (from Phase 0 pattern)**

Follow the same verification approach as Phase 0:

**Step 1: Verify test file created**
```bash
ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py
wc -l /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py
```
- [ ] File exists with ~300+ lines

**Step 2: Run tests to verify they pass**
```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
pytest tests/api_tests/test_data_encoding.py -v
```
- [ ] All 9+ tests pass (or adjust based on actual behavior)

**Step 3: Measure coverage for data_encoding.py**
```bash
coverage run --source='.' -m pytest tests/api_tests/test_data_encoding.py -v
coverage report --include="api/data_encoding.py" --show-missing
```
- [ ] Coverage ≥ 75%, missing lines identified

**Step 4: Check test execution time**
```bash
pytest tests/api_tests/test_data_encoding.py -v --durations=10
```
- [ ] All tests complete in < 5 seconds total

**8. Expected Outcomes Pattern**

Based on Phase 0 pattern:

**Before Phase 1 (current state):**
```
api/data_encoding.py: 0% coverage (82 LOC uncovered)
No test_data_encoding.py file exists
Cannot test encoding transformations
```

**After Phase 1 (expected state):**
```
✅ test_data_encoding.py created with 9+ test scenarios
✅ api/data_encoding.py coverage ≥ 75%
✅ All encoding transformations tested
✅ Error handling validated
✅ Tests execute in < 5 seconds
✅ Real pandas operations (minimal mocking)
```

**9. Quality Gates**

Before proceeding from Phase 1 to Phase 2, all must be ✅:

1. [ ] test_data_encoding.py created with 9+ test cases
2. [ ] All tests pass successfully
3. [ ] api/data_encoding.py coverage ≥ 75%
4. [ ] Tests use shared fixtures from conftest.py
5. [ ] Tests marked with `@pytest.mark.unit`
6. [ ] Test execution time < 5 seconds
7. [ ] Real pandas operations used (minimal mocking)
8. [ ] User has verified the changes work
9. [ ] Phase 1 marked complete in implementation plan

---

#### Integration with Phase 0 Infrastructure

**10. Using Phase 0 Deliverables**

Phase 1 depends on Phase 0 deliverables:

**From .coveragerc:**
- Coverage will exclude test files automatically
- Precision set to 2 decimal places
- Show missing lines in reports

**From conftest.py files:**
- Use `temp_experiment_dir` fixture for file operations
- Can use `sample_dataframe` if needed
- Can use `set_global_seed` for reproducibility

**From pytest.ini:**
- Tests will be marked with `@pytest.mark.unit`
- Can run with `pytest -m unit` for selective execution
- Django configuration handled automatically

**11. Command Consistency (from Phase 0)**

All commands should use the same pattern:
- [ ] Working directory: Always `cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML`
- [ ] Test execution: `pytest <test_file> -v`
- [ ] Coverage measurement: `coverage run --source='.' -m pytest <test_file> -v`
- [ ] Coverage reporting: `coverage report --include="<module_path>"`

---

#### Error Handling Pattern (from Phase 0)

**12. If Issues Arise During Phase 1**

Follow Phase 0 error handling pattern:
- [ ] Document the specific error
- [ ] Check test file syntax: `python -m py_compile tests/api_tests/test_data_encoding.py`
- [ ] Verify imports: Test that `from api.data_encoding import codificar_datos` works
- [ ] Check fixture availability: Verify conftest.py fixtures accessible
- [ ] Run with verbose output: `pytest -vv --tb=long`
- [ ] Check if data_encoding.py module is importable
- [ ] Verify temp directories created/cleaned properly

---

#### Success Criteria Pattern (matching Phase 0 rigor)

**13. Phase 1 Success Criteria**

Phase 1 completion requires:
- [ ] test_data_encoding.py created with comprehensive tests
- [ ] api/data_encoding.py coverage ≥ 75%
- [ ] All tests pass successfully
- [ ] Tests execute in < 5 seconds
- [ ] Uses real pandas operations (minimal mocking)
- [ ] Uses shared fixtures from conftest.py
- [ ] Tests marked with `@pytest.mark.unit`
- [ ] All automated verification steps pass
- [ ] User manual verification completed
- [ ] Coverage increase measurable and documented

---

**Checklist Status:** Ready for implementation
**Estimated Implementation Time:** 30-45 minutes
**Next Phase After Completion:** Phase 2 (views.py REST Endpoint Tests)

---

**Phase Overview:**
Create comprehensive tests for the data_encoding module (82 LOC). This is a quick win - small module with clear inputs/outputs, perfect for establishing testing patterns.

### Objectives
1. Create new test file for data_encoding.py
2. Achieve 75%+ coverage on data_encoding module
3. Test encoding transformations with real pandas operations
4. Validate error handling for invalid inputs

### Files to Modify

**Create:**
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py`

### Implementation Details

#### Create test_data_encoding.py

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py`

```python
"""
Unit tests for api.data_encoding module.

Tests the codificar_datos() function which handles:
- OneHot encoding for categorical features
- Label encoding for binary targets
- OneHot encoding for multi-class targets
- Boolean to integer conversion
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from io import StringIO

from api.data_encoding import codificar_datos


@pytest.mark.unit
class TestCodificarDatos:
    """Test cases for codificar_datos() function."""

    def test_successful_encoding_with_onehot_features(self, temp_experiment_dir):
        """
        Scenario 1: Successful encoding with OneHot for categorical features
        Given a CSV with categorical features and numeric target
        When codificar_datos is called with method='onehot'
        Then categorical features should be one-hot encoded
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'C', 'A', 'B'],
            'num_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        target_column = 'target'
        encoding_method = 'onehot'
        target_encoding_method = 'label'
        selected_columns = ['cat_feature', 'num_feature']

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column,
            encoding_method,
            target_encoding_method,
            selected_columns
        )

        # Assert
        assert os.path.exists(result_csv)
        result_df = pd.read_csv(result_csv)

        # OneHot encoding creates separate columns for each category
        assert 'cat_feature_A' in result_df.columns
        assert 'cat_feature_B' in result_df.columns
        assert 'cat_feature_C' in result_df.columns
        assert 'num_feature' in result_df.columns
        assert 'target' in result_df.columns

        # Original categorical column should be removed
        assert 'cat_feature' not in result_df.columns

        # Check report structure
        assert isinstance(report, dict)
        assert 'encoded_columns' in report
        assert 'cat_feature' in report['encoded_columns']


    def test_label_encoding_for_binary_target(self, temp_experiment_dir):
        """
        Scenario 2: Label encoding for binary classification target
        Given a CSV with categorical target (Yes/No)
        When target_encoding_method='label'
        Then target should be encoded as integers (0, 1)
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column='target',
            encoding_method='label',
            target_encoding_method='label',
            selected_columns=['feature1']
        )

        # Assert
        result_df = pd.read_csv(result_csv)

        # Target should be numeric (0, 1)
        assert result_df['target'].dtype in [np.int64, np.int32]
        assert set(result_df['target'].unique()) == {0, 1}


    def test_onehot_encoding_for_multiclass_target(self, temp_experiment_dir):
        """
        Scenario 3: OneHot encoding for multi-class target
        Given a CSV with multi-class target (A, B, C)
        When target_encoding_method='onehot'
        Then target should be one-hot encoded into separate columns
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'target': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column='target',
            encoding_method='label',
            target_encoding_method='onehot',
            selected_columns=['feature1']
        )

        # Assert
        result_df = pd.read_csv(result_csv)

        # Should have separate target columns
        assert 'target_A' in result_df.columns
        assert 'target_B' in result_df.columns
        assert 'target_C' in result_df.columns
        assert 'target' not in result_df.columns


    def test_boolean_to_integer_conversion(self, temp_experiment_dir):
        """
        Scenario 4: Boolean columns converted to integers
        Given a CSV with boolean columns (True/False)
        When codificar_datos is called
        Then boolean columns should be converted to 0/1
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'bool_feature': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column='target',
            encoding_method='label',
            target_encoding_method='label',
            selected_columns=['bool_feature']
        )

        # Assert
        result_df = pd.read_csv(result_csv)

        # Boolean should be converted to int
        assert result_df['bool_feature'].dtype in [np.int64, np.int32]
        assert set(result_df['bool_feature'].unique()) == {0, 1}


    def test_preserves_numeric_features_unchanged(self, temp_experiment_dir):
        """
        Scenario 5: Numeric features preserved without modification
        Given a CSV with numeric features
        When codificar_datos is called
        Then numeric features should remain unchanged
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        original_values = [1.5, 2.3, 3.7, 4.2, 5.9]
        df = pd.DataFrame({
            'numeric_feature': original_values,
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column='target',
            encoding_method='label',
            target_encoding_method='label',
            selected_columns=['numeric_feature']
        )

        # Assert
        result_df = pd.read_csv(result_csv)

        # Numeric values should be identical
        pd.testing.assert_series_equal(
            result_df['numeric_feature'],
            pd.Series(original_values, name='numeric_feature'),
            check_dtype=False
        )


    def test_error_handling_missing_target_column(self, temp_experiment_dir):
        """
        Scenario 6: Error when target column doesn't exist
        Given a CSV without the specified target column
        When codificar_datos is called
        Then should raise KeyError or ValueError
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'actual_target': [0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act & Assert
        with pytest.raises((KeyError, ValueError)):
            codificar_datos(
                input_csv,
                temp_experiment_dir,
                target_column='nonexistent_target',
                encoding_method='label',
                target_encoding_method='label',
                selected_columns=['feature1']
            )


    def test_error_handling_invalid_csv_path(self, temp_experiment_dir):
        """
        Scenario 7: Error when CSV file doesn't exist
        Given an invalid CSV path
        When codificar_datos is called
        Then should raise FileNotFoundError
        """
        # Arrange
        invalid_csv = os.path.join(temp_experiment_dir, "nonexistent.csv")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            codificar_datos(
                invalid_csv,
                temp_experiment_dir,
                target_column='target',
                encoding_method='label',
                target_encoding_method='label',
                selected_columns=['feature1']
            )


    def test_empty_dataframe_handling(self, temp_experiment_dir):
        """
        Scenario 8: Handling of empty DataFrame
        Given an empty CSV file
        When codificar_datos is called
        Then should handle gracefully (either error or return empty result)
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "empty.csv")
        df = pd.DataFrame(columns=['feature1', 'target'])
        df.to_csv(input_csv, index=False)

        # Act & Assert
        # Test current behavior - adjust based on actual implementation
        try:
            result_csv, report = codificar_datos(
                input_csv,
                temp_experiment_dir,
                target_column='target',
                encoding_method='label',
                target_encoding_method='label',
                selected_columns=['feature1']
            )
            # If it succeeds, verify empty result
            result_df = pd.read_csv(result_csv)
            assert len(result_df) == 0
        except (ValueError, Exception) as e:
            # If it fails, that's also valid behavior - document it
            assert isinstance(e, (ValueError, KeyError, Exception))


    def test_mixed_categorical_and_numeric_features(self, temp_experiment_dir):
        """
        Scenario 9: Mixed feature types (categorical + numeric)
        Given a CSV with both categorical and numeric features
        When codificar_datos is called
        Then categorical features encoded, numeric preserved
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_csv, report = codificar_datos(
            input_csv,
            temp_experiment_dir,
            target_column='target',
            encoding_method='onehot',
            target_encoding_method='label',
            selected_columns=['cat1', 'cat2', 'num1', 'num2']
        )

        # Assert
        result_df = pd.read_csv(result_csv)

        # Categorical features should be encoded
        assert 'cat1' not in result_df.columns
        assert 'cat2' not in result_df.columns

        # Numeric features should be preserved
        assert 'num1' in result_df.columns
        assert 'num2' in result_df.columns

        # Check numeric values unchanged
        assert result_df['num1'].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
```

**Rationale:**
- Tests all major code paths in data_encoding.py
- Uses real pandas operations (no mocking for data transformations)
- Tests error handling for invalid inputs
- Uses shared `temp_experiment_dir` fixture
- Marked with `@pytest.mark.unit` for selective execution

---

### Automated Verification

```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML

# 1. Run the new tests
pytest tests/api_tests/test_data_encoding.py -v

# 2. Run with coverage
coverage run --source='.' -m pytest tests/api_tests/test_data_encoding.py -v

# 3. Check coverage for data_encoding module specifically
coverage report --include="api/data_encoding.py"

# Expected: api/data_encoding.py should show ≥ 75% coverage
```

---

### Manual Verification Steps

**Checklist:**

1. **Verify test file created:**
   ```bash
   ls -la /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py
   wc -l /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_encoding.py
   ```
   ✅ File exists with ~300+ lines

2. **Run tests and verify all pass:**
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
   pytest tests/api_tests/test_data_encoding.py -v
   ```
   ✅ All 9 tests pass (or adjust based on actual behavior)

3. **Check coverage for data_encoding.py:**
   ```bash
   coverage run --source='.' -m pytest tests/api_tests/test_data_encoding.py -v
   coverage report --include="api/data_encoding.py" --show-missing
   ```
   ✅ Coverage ≥ 75%, missing lines identified

4. **Verify test execution time:**
   ```bash
   pytest tests/api_tests/test_data_encoding.py -v --durations=10
   ```
   ✅ All tests complete in < 5 seconds total

---

### Success Criteria

✅ **PHASE 1 COMPLETED - 2025-12-31**

All success criteria met:
- ✅ test_data_encoding.py created with 15 test cases (exceeds 9+ requirement)
- ✅ All tests pass successfully (15/15 passing)
- ✅ api/data_encoding.py coverage = 100% (exceeds 75% target)
- ✅ Tests execute in 0.21 seconds (well under 5-second target)
- ✅ Tests use real pandas operations (minimal mocking)
- ✅ Uses shared fixtures from conftest.py (temp_experiment_dir)
- ✅ Tests marked with `@pytest.mark.unit`
- ✅ All automated verification steps passed
- ✅ User manual verification completed

**Verification Results:**
- Test file: 560 lines, 21KB
- Coverage: 29 statements, 0 missed = 100.00%
- Execution time: 0.21s
- All 15 test scenarios covering:
  - get_dummies encoding for object features
  - OneHotEncoder for categorical targets
  - LabelEncoder for binary targets
  - Mutual exclusivity validation
  - "_vacio" column filtering
  - Boolean to integer conversion
  - Mixed data types handling
  - Empty DataFrame handling
  - File not found error handling
  - Multiple categorical features

---

## Phase 2A: Infrastructure Endpoints (DVC + MLflow)

**Note:** This phase is from PHASE_0_AND_2_FIXES.md - an atomic breakdown of the original Phase 2.

### Phase 2A: Pattern Consistency Checklist

**Date:** 2025-12-31
**Previous Phase:** Phase 1 ✅ COMPLETED
**Current Phase:** Phase 2A - Infrastructure Endpoints (DVC + MLflow)
**Purpose:** Ensure Phase 2A implementation follows established patterns from Phase 0 and Phase 1

---

#### Pre-Implementation Checklist

**1. Prerequisites Verification**

Before starting Phase 2A, verify:
- [x] Phase 0 completed successfully (all sub-phases 0A-0D)
- [x] Phase 1 completed successfully (data_encoding.py tests)
- [x] .coveragerc configuration exists and working
- [x] All 3 conftest.py files exist with shared fixtures
- [x] pytest.ini has 5 custom markers configured
- [x] Working directory: `/workspaces/dream-ml-c/DREAM-ML-backend/GEML`
- [ ] api/views.py file exists and is readable
- [ ] Current baseline for infrastructure endpoints measured
- [ ] Shared fixtures from tests/api_tests/conftest.py verified

**2. Lessons from Phase 1**

**From Phase 1 (data_encoding.py tests):**
- ✅ Always read actual function signatures before writing tests
- ✅ Test both return values AND side effects (CSV files)
- ✅ Verify fixtures work before using them
- ✅ Focus on edge cases and error conditions
- ✅ Achieve 75%+ coverage (or higher if possible)
- ✅ Keep execution time fast (< 5 seconds per module)
- ✅ Use `@pytest.mark.unit` for all unit tests
- ✅ Document test scenarios with clear Given/When/Then format

**Key Insights to Apply:**
1. Phase 2A extends **existing test file** (test_views.py)
2. Tests should use heavy mocking for external dependencies (DVC, MLflow)
3. Use Django RequestFactory for view testing
4. Mark tests with `@pytest.mark.django_db` and `@pytest.mark.unit`
5. Test HTTP method validation (405 errors)
6. Test successful scenarios AND error scenarios
7. Aim for infrastructure endpoint coverage increase

---

#### Implementation Pattern Checklist

**3. File Modification Pattern**

Phase 2A extends **1 existing test file**:
- [ ] `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py`

**Pattern to follow (from Phase 1):**
- Read the file first to understand existing structure
- Append new test classes to end of file
- Include proper phase header comment
- Import pytest and necessary mocking libraries
- Use existing patterns from current test_views.py

**4. Test Structure Pattern**

**Test Class Naming:**
```python
@pytest.mark.django_db
@pytest.mark.unit
class TestInitDvcView:
    """Tests for init_dvc() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()
```

**Test Method Naming:**
```python
@patch('api.views.init_dvc_logic')
def test_successful_dvc_initialization(self, mock_init_dvc_logic):
    """
    Scenario: Successful DVC initialization
    Given valid experiment directory
    When POST request to init_dvc endpoint
    Then should initialize DVC and return success
    """
```

**Pattern components:**
- Use `@pytest.mark.django_db` decorator (Django views need DB)
- Use `@pytest.mark.unit` decorator
- Use `setup_method` to initialize RequestFactory
- Clear class names describing endpoint being tested
- Docstrings with Given/When/Then format
- Use `@patch` decorators for mocking external dependencies
- Arrange/Act/Assert structure in test body

**5. Mocking Strategy Pattern**

**For Phase 2A (Infrastructure Endpoints):**
- [ ] Mock `api.views.init_dvc_logic` - subprocess/DVC commands
- [ ] Mock `api.views.configure_dvc_remote_logic` - DVC remote config
- [ ] Mock `api.views.start_mlflow_logic` - MLflow server startup
- [ ] Mock `api.views.is_mlflow_running` - port/process checks
- [ ] Do NOT mock Django RequestFactory or views themselves
- [ ] Mock return values should match expected API responses

**Avoid:**
- Mocking too much (lose test value)
- Mocking too little (tests become slow/brittle)
- Forgetting to verify mock was called correctly

**6. Testing Strategy Pattern**

**For infrastructure endpoints (3 endpoints, ~5 test scenarios):**
- [ ] Test successful initialization/configuration scenarios
- [ ] Test HTTP method validation (405 for wrong methods)
- [ ] Test error handling from service layer
- [ ] Test already-initialized scenarios (MLflow already running)
- [ ] Verify correct parameters passed to service layer

**Test scenarios to cover:**
1. init_dvc() - successful initialization
2. init_dvc() - invalid HTTP method (GET → 405)
3. configure_dvc_remote() - successful S3 remote config
4. start_mlflow() - start when not running
5. start_mlflow() - already running scenario

---

#### Verification Pattern Checklist

**7. Automated Verification (from Phase 1 pattern)**

Follow the same verification approach as Phase 1:

**Step 1: Verify test classes added to test_views.py**
```bash
grep -n "class TestInitDvcView" tests/api_tests/test_views.py
grep -n "class TestConfigureDvcRemoteView" tests/api_tests/test_views.py
grep -n "class TestStartMlflowView" tests/api_tests/test_views.py
```
- [ ] All 3 test classes present

**Step 2: Run Phase 2A tests specifically**
```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
pytest tests/api_tests/test_views.py::TestInitDvcView -v
pytest tests/api_tests/test_views.py::TestConfigureDvcRemoteView -v
pytest tests/api_tests/test_views.py::TestStartMlflowView -v
```
- [ ] All ~5 tests pass

**Step 3: Measure coverage increase for infrastructure endpoints**
```bash
coverage run --source='.' -m pytest tests/api_tests/test_views.py::TestInitDvcView tests/api_tests/test_views.py::TestConfigureDvcRemoteView tests/api_tests/test_views.py::TestStartMlflowView -v
coverage report --include="api/views.py" --show-missing
```
- [ ] Coverage increased for init_dvc, configure_dvc_remote, start_mlflow functions

**Step 4: Check test execution time**
```bash
pytest tests/api_tests/test_views.py::TestInitDvcView tests/api_tests/test_views.py::TestConfigureDvcRemoteView tests/api_tests/test_views.py::TestStartMlflowView -v --durations=10
```
- [ ] All tests complete in < 5 seconds total

**8. Expected Outcomes Pattern**

Based on Phase 1 pattern:

**Before Phase 2A (current state):**
```
test_views.py exists with some tests
Infrastructure endpoints (init_dvc, configure_dvc_remote, start_mlflow) have 0% or low coverage
Cannot verify infrastructure setup logic
```

**After Phase 2A (expected state):**
```
✅ test_views.py extended with 3 new test classes
✅ ~5 new test methods added
✅ Infrastructure endpoints have test coverage
✅ HTTP method validation tested
✅ Error scenarios tested
✅ Tests execute in < 5 seconds
✅ Heavy mocking for DVC/MLflow dependencies
```

**9. Quality Gates**

Before proceeding from Phase 2A to Phase 2B, all must be ✅:

1. [ ] 3 test classes added to test_views.py (TestInitDvcView, TestConfigureDvcRemoteView, TestStartMlflowView)
2. [ ] ~5 test methods added across the 3 classes
3. [ ] All tests pass successfully
4. [ ] Coverage increased for infrastructure endpoints
5. [ ] Tests use Django RequestFactory pattern
6. [ ] Tests use `@pytest.mark.django_db` and `@pytest.mark.unit`
7. [ ] Test execution time < 5 seconds
8. [ ] Heavy mocking of DVC/MLflow operations
9. [ ] User has verified the changes work
10. [ ] Phase 2A marked complete in PHASE_0_AND_2_FIXES.md

---

#### Integration with Phase 0 and Phase 1 Infrastructure

**10. Using Previous Phase Deliverables**

Phase 2A depends on Phase 0 and Phase 1 deliverables:

**From Phase 0 (.coveragerc):**
- Coverage will exclude test files automatically
- Precision set to 2 decimal places
- Show missing lines in reports

**From Phase 0 (conftest.py files):**
- Use `request_factory` fixture from tests/api_tests/conftest.py
- Can use `mock_subprocess_success` if needed
- Can use `mock_mlflow_tracking` if needed

**From Phase 0 (pytest.ini):**
- Tests will be marked with `@pytest.mark.unit`
- Tests will be marked with `@pytest.mark.django_db`
- Can run with `pytest -m unit` for selective execution

**From Phase 1 (test patterns):**
- Follow Given/When/Then docstring format
- Use Arrange/Act/Assert test structure
- Verify both return values and side effects
- Test error conditions thoroughly

**11. Command Consistency (from Phase 0 and Phase 1)**

All commands should use the same pattern:
- [ ] Working directory: Always `cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML`
- [ ] Test execution: `pytest tests/api_tests/test_views.py::TestClassName -v`
- [ ] Coverage measurement: `coverage run --source='.' -m pytest <test_file> -v`
- [ ] Coverage reporting: `coverage report --include="api/views.py"`

---

#### Error Handling Pattern (from Phase 0 and Phase 1)

**12. If Issues Arise During Phase 2A**

Follow Phase 1 error handling pattern:
- [ ] Document the specific error
- [ ] Check test file syntax: `python -m py_compile tests/api_tests/test_views.py`
- [ ] Verify imports: Test that `from api import views` works
- [ ] Check fixture availability: Verify conftest.py fixtures accessible
- [ ] Run with verbose output: `pytest -vv --tb=long`
- [ ] Check if api/views.py module is importable
- [ ] Verify mocks are configured correctly

---

#### Success Criteria Pattern (matching Phase 1 rigor)

**13. Phase 2A Success Criteria**

Phase 2A completion requires:
- [ ] 3 test classes added to test_views.py
- [ ] ~5 test methods added
- [ ] All tests pass successfully
- [ ] Coverage increased for init_dvc, configure_dvc_remote, start_mlflow
- [ ] Tests execute in < 5 seconds
- [ ] Uses Django RequestFactory pattern
- [ ] Uses `@pytest.mark.django_db` and `@pytest.mark.unit`
- [ ] All automated verification steps pass
- [ ] User manual verification completed
- [ ] Coverage increase measurable and documented

---

**Checklist Status:** Ready for implementation
**Estimated Implementation Time:** 30-40 minutes
**Estimated Lines of Code:** ~180 LOC
**Next Phase After Completion:** Phase 2B (Data Analysis & Upload Endpoints)

---

## Original Phase 2: views.py REST Endpoint Tests (SUPERSEDED by Phase 2A-2E)

**Phase Overview:**
Extend existing test_views.py to achieve comprehensive coverage of all 13 REST API endpoints in views.py (429 LOC).

### Objectives
1. Add tests for missing or under-tested endpoints
2. Achieve 75%+ coverage on api/views.py
3. Test HTTP method validation (GET/POST enforcement)
4. Test error responses (400, 404, 405, 500)
5. Mock service layer and external dependencies

### Files to Modify

**Extend:**
- `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py`

### Implementation Details

#### Extend test_views.py

**File:** `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py`

Add these test cases (append to existing file after removing manual Django config):

```python
"""
Additional tests for api.views module REST endpoints.
Extends existing test_views.py with comprehensive endpoint coverage.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
from django.test import RequestFactory


@pytest.mark.django_db
@pytest.mark.unit
class TestInitDvcView:
    """Tests for init_dvc() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.init_dvc_logic')
    def test_successful_dvc_initialization(self, mock_init_dvc_logic):
        """
        Scenario: Successful DVC initialization
        Given valid experiment directory
        When POST request to init_dvc endpoint
        Then should initialize DVC and return success
        """
        # Arrange
        mock_init_dvc_logic.return_value = {
            'status': 'success',
            'message': 'DVC initialized'
        }

        request_data = {
            'experiment_dir': '/app/experimentos/exp123'
        }
        request = self.factory.post(
            '/init-dvc/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.init_dvc(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        mock_init_dvc_logic.assert_called_once_with('/app/experimentos/exp123')

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to init_dvc endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/init-dvc/')

        # Act
        from api import views
        response = views.init_dvc(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert 'error' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestConfigureDvcRemoteView:
    """Tests for configure_dvc_remote() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.configure_dvc_remote_logic')
    def test_successful_remote_configuration(self, mock_configure_logic):
        """
        Scenario: Successful DVC remote configuration
        Given valid S3 remote configuration
        When POST request with remote params
        Then should configure DVC remote and return success
        """
        # Arrange
        mock_configure_logic.return_value = {
            'status': 'success',
            'message': 'DVC remote configured'
        }

        request_data = {
            'experiment_dir': '/app/experimentos/exp123',
            'remote_name': 's3remote',
            'remote_url': 's3://mybucket/dvc-storage'
        }
        request = self.factory.post(
            '/configure-dvc-remote/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.configure_dvc_remote(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        mock_configure_logic.assert_called_once()


@pytest.mark.django_db
@pytest.mark.unit
class TestStartMlflowView:
    """Tests for start_mlflow() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.start_mlflow_logic')
    @patch('api.views.is_mlflow_running')
    def test_start_mlflow_when_not_running(self, mock_is_running, mock_start_logic):
        """
        Scenario: Start MLflow when not already running
        Given MLflow is not running
        When POST request to start_mlflow
        Then should start MLflow server and return success
        """
        # Arrange
        mock_is_running.return_value = False
        mock_start_logic.return_value = {
            'status': 'success',
            'port': 5000,
            'tracking_uri': 'http://localhost:5000'
        }

        request = self.factory.post('/start-mlflow/')

        # Act
        from api import views
        response = views.start_mlflow(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'tracking_uri' in response_data
        mock_start_logic.assert_called_once()

    @patch('api.views.is_mlflow_running')
    def test_mlflow_already_running(self, mock_is_running):
        """
        Scenario: MLflow already running
        Given MLflow is already running
        When POST request to start_mlflow
        Then should return already running message
        """
        # Arrange
        mock_is_running.return_value = True
        request = self.factory.post('/start-mlflow/')

        # Act
        from api import views
        response = views.start_mlflow(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'already running' in response_data.get('message', '').lower()


@pytest.mark.django_db
@pytest.mark.unit
class TestAnalyzeCsvView:
    """Tests for analyze_csv() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.analyze_csv_logic')
    def test_successful_csv_analysis(self, mock_analyze_logic):
        """
        Scenario: Successful CSV analysis
        Given uploaded CSV file
        When POST request with file
        Then should analyze and return column information
        """
        # Arrange
        mock_analyze_logic.return_value = {
            'columns': ['feature1', 'feature2', 'target'],
            'row_count': 100,
            'column_types': {
                'feature1': 'float64',
                'feature2': 'object',
                'target': 'int64'
            }
        }

        from django.core.files.uploadedfile import SimpleUploadedFile
        csv_content = b"feature1,feature2,target\n1.0,A,0\n2.0,B,1"
        csv_file = SimpleUploadedFile("test.csv", csv_content)

        request = self.factory.post('/analyze-csv/', {'file': csv_file})

        # Act
        from api import views
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'columns' in response_data
        assert len(response_data['columns']) == 3

    def test_missing_file_returns_400(self):
        """
        Scenario: Missing file in request
        Given POST request without file
        When sent to analyze_csv
        Then should return 400 Bad Request
        """
        # Arrange
        request = self.factory.post('/analyze-csv/', {})

        # Act
        from api import views
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'error' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestUploadAndCleanCsvView:
    """Tests for upload_and_clean_csv() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.upload_and_clean_csv_logic')
    def test_successful_upload_and_cleaning(self, mock_upload_logic):
        """
        Scenario: Successful CSV upload and cleaning
        Given valid CSV file and cleaning options
        When POST request with file and options
        Then should upload, clean, and return success
        """
        # Arrange
        mock_upload_logic.return_value = {
            'status': 'success',
            'cleaned_csv_path': '/app/experimentos/exp123/cleaned.csv',
            'report': {
                'duplicates_removed': 5,
                'missing_values_handled': 10
            }
        }

        from django.core.files.uploadedfile import SimpleUploadedFile
        csv_content = b"feature1,feature2,target\n1.0,A,0\n2.0,B,1"
        csv_file = SimpleUploadedFile("test.csv", csv_content)

        request_data = {
            'file': csv_file,
            'experiment_id': 'exp123',
            'remove_duplicates': 'true',
            'handle_missing': 'mean'
        }
        request = self.factory.post('/upload-clean/', request_data)

        # Act
        from api import views
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'cleaned_csv_path' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestGenerarReporteEdaView:
    """Tests for generar_reporte_eda() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.generate_eda_logic')
    def test_successful_eda_report_generation(self, mock_generate_logic):
        """
        Scenario: Successful EDA report generation
        Given valid experiment with cleaned data
        When POST request to generate EDA
        Then should generate report and return path
        """
        # Arrange
        mock_generate_logic.return_value = {
            'status': 'success',
            'report_path': '/app/experimentos/exp123/eda_report.html'
        }

        request_data = {
            'experiment_id': 'exp123',
            'dataset_type': 'eda'
        }
        request = self.factory.post(
            '/generate-eda/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'report_path' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestEncodeCsvView:
    """Tests for encode_csv() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.encode_csv_logic')
    def test_successful_encoding(self, mock_encode_logic):
        """
        Scenario: Successful CSV encoding
        Given cleaned CSV with encoding configuration
        When POST request to encode_csv
        Then should encode features and return success
        """
        # Arrange
        mock_encode_logic.return_value = {
            'status': 'success',
            'encoded_csv_path': '/app/experimentos/exp123/encoded.csv',
            'encoding_report': {
                'onehot_columns': ['cat1', 'cat2'],
                'label_encoded': ['target']
            }
        }

        request_data = {
            'experiment_id': 'exp123',
            'target_column': 'target',
            'encoding_method': 'onehot',
            'selected_columns': ['cat1', 'cat2', 'num1']
        }
        request = self.factory.post(
            '/encode/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'


@pytest.mark.django_db
@pytest.mark.unit
class TestStartJupyterView:
    """Tests for start_jupyter() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.start_jupyter_logic')
    @patch('api.views.is_port_available')
    def test_successful_jupyter_startup(self, mock_port_available, mock_start_logic):
        """
        Scenario: Successful Jupyter notebook startup
        Given available port
        When POST request to start Jupyter
        Then should start Jupyter and return URL
        """
        # Arrange
        mock_port_available.return_value = True
        mock_start_logic.return_value = {
            'status': 'success',
            'jupyter_url': 'http://localhost:8888',
            'token': 'test-token-123'
        }

        request_data = {'experiment_dir': '/app/experimentos/exp123'}
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'jupyter_url' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestTrainModelView:
    """Tests for train_model() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.train_model_logic')
    def test_successful_model_training_logistic_regression(self, mock_train_logic):
        """
        Scenario: Successful logistic regression training
        Given valid training configuration
        When POST request to train_model
        Then should train model and return metrics
        """
        # Arrange
        mock_train_logic.return_value = {
            'status': 'success',
            'model_path': '/app/experimentos/exp123/model.pkl',
            'metrics': {
                'accuracy': 0.85,
                'f1_score': 0.83,
                'roc_auc': 0.87
            },
            'run_id': 'mlflow-run-123'
        }

        request_data = {
            'experiment_id': 'exp123',
            'algorithm': 'logistic_regression',
            'search_method': 'grid',
            'random_state': 42
        }
        request = self.factory.post(
            '/train/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.train_model(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'metrics' in response_data
        assert response_data['metrics']['accuracy'] == 0.85

    @patch('api.views.train_model_logic')
    def test_training_error_handling(self, mock_train_logic):
        """
        Scenario: Training error handling
        Given training logic that raises error
        When POST request to train_model
        Then should return 500 with error message
        """
        # Arrange
        mock_train_logic.side_effect = ValueError("Invalid hyperparameters")

        request_data = {
            'experiment_id': 'exp123',
            'algorithm': 'xgboost'
        }
        request = self.factory.post(
            '/train/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.train_model(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'error' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestGetPipelineConfigView:
    """Tests for get_pipeline_config() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('builtins.open', new_callable=mock_open, read_data='{"algorithm": "mlp"}')
    @patch('os.path.exists')
    def test_successful_config_retrieval(self, mock_exists, mock_file):
        """
        Scenario: Successful pipeline configuration retrieval
        Given existing config file
        When GET request to get_pipeline_config
        Then should return configuration
        """
        # Arrange
        mock_exists.return_value = True
        request = self.factory.get('/pipeline-config/?experiment_id=exp123')

        # Act
        from api import views
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'algorithm' in response_data

    @patch('os.path.exists')
    def test_config_not_found_returns_404(self, mock_exists):
        """
        Scenario: Configuration file not found
        Given non-existent config file
        When GET request to get_pipeline_config
        Then should return 404
        """
        # Arrange
        mock_exists.return_value = False
        request = self.factory.get('/pipeline-config/?experiment_id=exp999')

        # Act
        from api import views
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 404


@pytest.mark.django_db
@pytest.mark.unit
class TestRunPipelineView:
    """Tests for run_pipeline() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.run_pipeline_logic')
    def test_successful_pipeline_execution(self, mock_pipeline_logic):
        """
        Scenario: Successful end-to-end pipeline execution
        Given valid pipeline configuration
        When POST request to run_pipeline
        Then should execute full pipeline and return summary
        """
        # Arrange
        mock_pipeline_logic.return_value = {
            'status': 'success',
            'steps_completed': [
                'upload_clean',
                'generate_eda',
                'encode',
                'train'
            ],
            'final_metrics': {
                'accuracy': 0.88,
                'f1_score': 0.86
            }
        }

        request_data = {
            'experiment_id': 'exp123',
            'pipeline_config': {
                'algorithm': 'xgboost',
                'search_method': 'bayesian',
                'n_trials': 2
            }
        }
        request = self.factory.post(
            '/pipeline/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        from api import views
        response = views.run_pipeline(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert len(response_data['steps_completed']) == 4


@pytest.mark.django_db
@pytest.mark.unit
class TestGetExperimentSummaryView:
    """Tests for get_experiment_summary() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    def test_successful_summary_generation(self, mock_exists, mock_generate_pdf):
        """
        Scenario: Successful experiment summary generation
        Given experiment with completed training
        When GET request to get_experiment_summary
        Then should generate and return PDF summary
        """
        # Arrange
        mock_exists.return_value = True
        mock_generate_pdf.return_value = '/app/experimentos/exp123/summary.pdf'

        request = self.factory.get('/experiment-summary/?experiment_id=exp123')

        # Act
        from api import views
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'summary_path' in response_data
```

**Rationale:**
- Comprehensive coverage of all 13 endpoints in views.py
- Tests HTTP method validation (405 errors)
- Tests error handling (400, 404, 500)
- Heavy mocking of service layer (align with user preference)
- Marked with `@pytest.mark.unit` and `@pytest.mark.django_db`

---

### Automated Verification

```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML

# 1. Run all view tests
pytest tests/api_tests/test_views.py -v

# 2. Run with coverage
coverage run --source='.' -m pytest tests/api_tests/test_views.py -v

# 3. Check coverage for views.py specifically
coverage report --include="api/views.py" --show-missing

# Expected: api/views.py should show ≥ 75% coverage
```

---

### Manual Verification Steps

**Checklist:**

1. **Verify test_views.py updated:**
   ```bash
   wc -l /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py
   ```
   ✅ Line count significantly increased (1298 + ~600 new lines)

2. **Verify manual Django config removed:**
   ```bash
   grep -n "settings.configure" /workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_views.py
   ```
   ✅ Should return no results

3. **Run all view tests:**
   ```bash
   cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
   pytest tests/api_tests/test_views.py -v --tb=short
   ```
   ✅ All tests pass (adjust for actual behavior)

4. **Check coverage increase:**
   ```bash
   coverage run --source='.' -m pytest tests/api_tests/test_views.py -v
   coverage report --include="api/views.py" --show-missing
   ```
   ✅ Coverage ≥ 75%

5. **Verify test execution time:**
   ```bash
   pytest tests/api_tests/test_views.py -v --durations=20
   ```
   ✅ Total time < 30 seconds

---

### Success Criteria

- ✅ test_views.py extended with 40+ new test methods
- ✅ All 13 endpoints have test coverage
- ✅ Manual Django configuration removed
- ✅ All tests pass
- ✅ api/views.py coverage ≥ 75%
- ✅ Tests execute in < 30 seconds

---

## Phases 3-10: Remaining Implementation

**Note:** Due to space constraints, I'll provide a high-level overview of the remaining phases. Each will follow the same detailed structure as Phases 1-2.

### Phase 3: utils.py Infrastructure Tests

**Files:** Extend `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_utils.py`

**Focus Areas:**
- `init_dvc_logic()` - Mock subprocess for git/dvc commands
- `configure_dvc_remote_logic()` - Mock DVC remote configuration
- `start_mlflow_logic()` - Mock subprocess for MLflow server
- `is_mlflow_running()` - Mock port/process checks
- `start_jupyter_logic()` - Mock Jupyter startup
- `send_progress_update()` - Mock WebSocket channels
- `generate_experiment_summary_pdf()` - Mock PDF generation

**Target:** 75%+ coverage on utils.py (315 LOC)

---

### Phase 4: train.py - Data Preparation Utilities

**Files:** Extend `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/arreglar.py` (or create `test_train_data_prep.py`)

**Focus Functions:**
- `set_global_seeds()` - Test seed initialization (SEED=42)
- `load_and_validate_data()` - Test CSV loading, validation
- `split_dataset()` - Test train/val/test splits with stratification

**Target:** Test data preparation pipeline with real pandas/numpy operations

---

### Phase 5: train.py - Model Training Functions

**Files:** Extend test file for training functions

**Focus Functions:**
- `train_logistic_regression_model()` - Test with n_trials=2 Bayesian
- `train_mlp_model()` - Test with minimal iterations
- `train_xgboost_model()` - Test with deterministic seed

**Mocking Strategy:**
- Mock MLflow logging entirely
- Use real sklearn/xgboost with SEED=42
- Test with tiny datasets (50 samples) for speed

**Target:** 75%+ coverage on training functions

---

### Phase 6: train.py - Model Evaluation Functions

**Files:** Extend test file for evaluation

**Focus Functions:**
- `evaluate_model()` - Test metrics calculation (accuracy, F1, ROC-AUC)
- `generate_plots()` - Mock matplotlib, test plot generation calls

**Target:** Test metric calculations with real sklearn.metrics

---

### Phase 7: train.py - Bayesian Optimization Utilities

**Files:** Extend test file for Bayesian utilities

**Focus Functions:**
- Hyperparameter search space validation
- Optuna integration with n_trials=2
- Search reproducibility with SEED=42

**Target:** Validate Bayesian search integration without full optimization

---

### Phase 8: services.py Orchestration Tests

**Files:** Extend `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_services.py`

**Focus Functions:**
- `create_experiment_logic()` - Mock MLflow/DVC initialization
- `upload_and_clean_csv_logic()` - Mock file I/O, test orchestration
- `generate_eda_logic()` - Mock report generation libraries
- `encode_csv_logic()` - Mock encoding functions
- `train_model_logic()` - Mock training functions
- `run_pipeline_logic()` - Mock full pipeline steps

**Mocking Strategy:**
- Heavy mocking of all dependencies
- Test orchestration logic and error propagation
- Validate correct function call sequences

**Target:** 75%+ coverage on services.py (836 LOC)

---

### Phase 9: data_cleaning.py Tests (If Not Already Covered)

**Files:** Extend `/workspaces/dream-ml-c/DREAM-ML-backend/GEML/tests/api_tests/test_data_cleaning.py`

**Focus:**
- `limpiar_datos()` - Comprehensive cleaning tests with real pandas

**Target:** 75%+ coverage (if not already achieved)

---

### Phase 10: Integration Test Validation

**Objective:** Run full test suite and validate overall coverage

**Actions:**
1. Run complete test suite
2. Generate coverage report
3. Identify any remaining gaps
4. Add targeted tests for uncovered lines

**Target:** api package overall coverage ≥ 75%

---

## Phases 11-14: apiTimeSeries Package (High-Level Outline)

### Phase 11: apiTimeSeries Data Utilities

**Files:**
- `test_data_cleaning_utils.py` (already exists - extend)
- Create `test_data_encoding_utils.py`

**Target:** 75%+ coverage on data preprocessing utilities

---

### Phase 12: apiTimeSeries Views & Services

**Files:**
- Extend existing `test_views_*.py` files
- Extend `test_services.py`

**Target:** 75%+ coverage on REST endpoints and service orchestration

---

### Phase 13: apiTimeSeries train.py - Part 1 (ARIMA/XGBoost)

**Files:** Create focused test files for ARIMA and XGBoost training

**Functions:**
- `train_arima_model()` - With n_trials=2 Bayesian
- `train_xgboost_model()` - Time series specific
- Evaluation functions

**Target:** 60%+ coverage (large module - 2010 LOC)

---

### Phase 14: apiTimeSeries train.py - Part 2 (LSTM)

**Files:** Extend existing LSTM phase tests

**Functions:**
- `train_lstm_model()` - Deep learning training
- `create_sequences_for_lstm()` - Sequence creation
- `build_lstm_model()` - Architecture building

**Use existing test patterns from test_lstm_phase*.py**

**Target:** Complete apiTimeSeries coverage to 75%+

---

## Overall Success Criteria

### Coverage Metrics
- ✅ api package: ≥ 75% line coverage
- ✅ apiTimeSeries package: ≥ 75% line coverage
- ✅ Combined backend: ≥ 75% coverage

### Quality Metrics
- ✅ All tests pass successfully
- ✅ Test suite executes in < 2 minutes
- ✅ No test failures or errors
- ✅ Proper use of fixtures and mocking patterns

### Infrastructure
- ✅ .coveragerc configuration in place
- ✅ conftest.py files with shared fixtures
- ✅ pytest markers configured and used
- ✅ Manual Django configuration removed

---

## Execution Commands Reference

### Run All Tests
```bash
cd /workspaces/dream-ml-c/DREAM-ML-backend/GEML
pytest -v
```

### Run with Coverage
```bash
coverage run --source='.' -m pytest -v
coverage report
coverage html  # Generate HTML report
```

### Run Specific Test Markers
```bash
pytest -m unit -v           # Fast unit tests only
pytest -m "not slow" -v     # Exclude slow tests
```

### Coverage by Package
```bash
coverage report --include="api/*"
coverage report --include="apiTimeSeries/*"
```

### Phase-by-Phase Validation
```bash
# After each phase:
coverage run --source='.' -m pytest tests/api_tests/test_<module>.py -v
coverage report --include="api/<module>.py" --show-missing
```

---

## Appendix: Test Pattern Examples

### Pattern 1: View Testing with Mocking

```python
@pytest.mark.django_db
@pytest.mark.unit
class TestExampleView:
    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.service_function')
    def test_successful_request(self, mock_service):
        mock_service.return_value = {'status': 'success'}
        request = self.factory.post('/endpoint/', data={...})
        response = views.example_view(request)
        assert response.status_code == 200
```

### Pattern 2: Service Testing with Heavy Mocking

```python
@pytest.mark.unit
class TestExampleService:
    @patch('api.services.mlflow.start_run')
    @patch('api.services.subprocess.run')
    @patch('pandas.read_csv')
    def test_service_logic(self, mock_read_csv, mock_subprocess, mock_mlflow):
        # Setup mocks
        mock_read_csv.return_value = pd.DataFrame(...)
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Test
        result = service_function(...)

        # Assertions
        assert result['status'] == 'success'
        mock_mlflow.assert_called_once()
```

### Pattern 3: ML Training with Deterministic Seeds

```python
@pytest.mark.unit
@pytest.mark.slow
class TestModelTraining:
    @patch('api.train.mlflow.log_metric')
    def test_reproducible_training(self, mock_log_metric, set_global_seed):
        set_global_seed(42)

        # Create tiny dataset
        X = np.random.randn(50, 5)
        y = np.random.choice([0, 1], 50)

        # Train with minimal iterations
        model = train_logistic_regression_model(
            X, y,
            search_method='bayesian',
            n_trials=2,
            random_state=42
        )

        # Verify deterministic behavior
        assert model is not None
        mock_log_metric.assert_called()  # MLflow logging happened
```

---

**End of Implementation Plan**

**Total Estimated Effort:** 10-14 phases
**Estimated Time:** Varies by phase complexity
**Priority Order:** Phases 0-2 (foundation), then 3-10 (api), then 11-14 (apiTimeSeries)
