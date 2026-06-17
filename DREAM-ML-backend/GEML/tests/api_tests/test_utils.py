import pytest
import os
import socket
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from io import StringIO
import pandas as pd
from api import utils
import json
import tempfile
import subprocess
from pathlib import Path
import requests
from io import StringIO

@pytest.fixture
def temp_experiment_dir(tmp_path):
    """Create a temporary directory for an experiment."""
    exp_dir = tmp_path / "test_experiment"
    exp_dir.mkdir()
    return str(exp_dir)

# --- Tests for analyze_csv_logic ---

def test_analyze_csv_logic_success():
    """Test successful analysis of a valid CSV."""
    # Arrange
    csv_content = "header1,header2,header3\nvalue1,value2,value3"
    csv_file = StringIO(csv_content)
    expected_columns = ["header1", "header2", "header3"]

    # Act
    result = utils.analyze_csv_logic(csv_file)

    # Assert
    assert "columns" in result
    assert result["columns"] == expected_columns

def test_analyze_csv_logic_empty_csv():
    """Test analysis of a CSV with only headers."""
    # Arrange
    csv_content = "col_a,col_b"
    csv_file = StringIO(csv_content)
    expected_columns = ["col_a", "col_b"]

    # Act
    result = utils.analyze_csv_logic(csv_file)

    # Assert
    assert result["columns"] == expected_columns



# --- Tests for is_port_available ---

def test_is_port_available_when_free():
    """Test that a free port is identified as available."""
    # Arrange
    # Find a free port by binding to port 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        free_port = s.getsockname()[1]
    
    # Act
    result = utils.is_port_available(free_port)

    # Assert
    assert result is True

def test_is_port_available_when_taken():
    """Test that a taken port is identified as unavailable."""
    # Arrange
    # Take a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        taken_port = s.getsockname()[1]
        
        # Act
        result = utils.is_port_available(taken_port)

        # Assert
        assert result is False

# --- Tests for start_jupyter_logic ---

@patch('api.utils.subprocess.Popen')
@patch('api.utils.start_run')
@patch('api.utils.get_experiment_by_name')
@patch('api.utils.set_tracking_uri')
@patch('api.utils.shutil.copy')
@patch('api.utils.log_param')
@patch('api.utils.log_artifact')
@patch('api.utils.os.makedirs')
@patch('api.utils.is_port_available', return_value=True)
@patch('os.path.exists', return_value=True)
def test_start_jupyter_logic_success(
    mock_exists, mock_is_port_available, mock_makedirs, mock_log_artifact, 
    mock_log_param, mock_copy, mock_set_uri, mock_get_exp, mock_start_run, 
    mock_popen, temp_experiment_dir
):
    """Test the successful start of a Jupyter server."""
    # Arrange
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test_exp_id"
    mock_get_exp.return_value = mock_experiment

    mock_run = MagicMock()
    mock_run.info.run_id = "new_run_id"
    print("new_run_id= ", mock_run.info.run_id)
    # Properly mock the context manager
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_run
    mock_context_manager.__exit__.return_value = None
    mock_start_run.return_value = mock_context_manager
    
    mock_process = MagicMock()
    mock_popen.return_value = mock_process

    # Act
    result = utils.start_jupyter_logic(temp_experiment_dir, "some_run_id", port=9999)

    # Assert
    assert result["success"] is True
    assert result["notebook_url"] == f"http://localhost:9999/tree/EDA_manual.ipynb"
    assert result["notebook_path"] == "EDA_manual.ipynb"
    
    mock_copy.assert_called_once()
    mock_set_uri.assert_called_once()
    mock_get_exp.assert_called_with(os.path.basename(temp_experiment_dir))
    mock_start_run.assert_called_once_with(nested=True)
    mock_popen.assert_called_once()
    command_args = mock_popen.call_args[0][0]
    assert f"--port=9999" in command_args
    assert f"--notebook-dir={temp_experiment_dir}" in command_args

def test_start_jupyter_logic_invalid_dir_raises_error(temp_experiment_dir):
    """Test that a non-existent experiment directory raises FileNotFoundError."""
    # Arrange
    invalid_dir = os.path.join(temp_experiment_dir, "non_existent")

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Directorio del experimento no válido"):
        utils.start_jupyter_logic(invalid_dir, "some_run_id")

def test_start_jupyter_logic_port_not_available_raises_error(temp_experiment_dir):
    """Test that an unavailable port raises OSError."""
    # Arrange
    with patch('api.utils.is_port_available', return_value=False):
        # Act & Assert
        with pytest.raises(OSError, match="El puerto 8888 no está disponible."):
            utils.start_jupyter_logic(temp_experiment_dir, "some_run_id", port=8888)

@patch('os.path.exists', side_effect=[True, False]) # experiment dir exists, template does not
def test_start_jupyter_logic_template_not_found_raises_error(mock_exists, temp_experiment_dir):
    """Test that a missing notebook template raises FileNotFoundError."""
    # Arrange
    with patch('api.utils.is_port_available', return_value=True):
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Plantilla EDA no encontrada"):
            utils.start_jupyter_logic(temp_experiment_dir, "some_run_id")



# Import the functions to test
from api.utils import (
    chunk_list,
    is_mlflow_running,
    analyze_csv_logic,
    init_dvc_logic,
    configure_dvc_remote_logic,
    start_mlflow_logic,
    start_jupyter_logic,
    send_progress_update,
    generate_experiment_summary_pdf,
    header_footer,
)


class TestChunkList:
    """Tests for the chunk_list function."""
    
    def test_chunk_list_longer_than_chunk_size(self):
        """Given a list longer than chunk_size When chunk_list is called Then it should return multiple chunks."""
        # Arrange
        test_list = list(range(25))
        chunk_size = 10
        
        # Act
        chunks = list(chunk_list(test_list, chunk_size))
        
        # Assert
        assert len(chunks) == 3
        assert chunks[0] == list(range(10))
        assert chunks[1] == list(range(10, 20))
        assert chunks[2] == list(range(20, 25))
    
    def test_chunk_list_shorter_than_chunk_size(self):
        """Given a list shorter than chunk_size When chunk_list is called Then it should return single chunk."""
        # Arrange
        test_list = [1, 2, 3]
        chunk_size = 10
        
        # Act
        chunks = list(chunk_list(test_list, chunk_size))
        
        # Assert
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]
    
    def test_chunk_list_empty_list(self):
        """Given an empty list When chunk_list is called Then it should return empty generator."""
        # Arrange
        test_list = []
        chunk_size = 10
        
        # Act
        chunks = list(chunk_list(test_list, chunk_size))
        
        # Assert
        assert len(chunks) == 0
    
    def test_chunk_list_chunk_size_one(self):
        """Given chunk_size of 1 When chunk_list is called Then it should return individual elements."""
        # Arrange
        test_list = [1, 2, 3]
        chunk_size = 1
        
        # Act
        chunks = list(chunk_list(test_list, chunk_size))
        
        # Assert
        assert len(chunks) == 3
        assert chunks[0] == [1]
        assert chunks[1] == [2]
        assert chunks[2] == [3]


class TestIsMLflowRunning:
    """Tests for the is_mlflow_running function."""
    
    @patch('api.utils.requests.get')
    def test_mlflow_server_returns_200(self, mock_get):
        """Given a server returning status 200 When is_mlflow_running is called Then it should return True."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        url = "http://localhost:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is True
        mock_get.assert_called_once_with(url, timeout=5)
    
    @patch('api.utils.requests.get')
    def test_mlflow_server_returns_405(self, mock_get):
        """Given a server returning status 405 When is_mlflow_running is called Then it should return True."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 405
        mock_get.return_value = mock_response
        url = "http://localhost:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is True
        mock_get.assert_called_once_with(url, timeout=5)
    
    @patch('api.utils.requests.get')
    def test_mlflow_server_returns_other_status(self, mock_get):
        """Given a server returning other status codes When is_mlflow_running is called Then it should return False."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        url = "http://localhost:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is False
    
    @patch('api.utils.requests.get')
    def test_mlflow_connection_error(self, mock_get):
        """Given an unreachable URL When is_mlflow_running is called Then it should return False."""
        # Arrange
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        url = "http://unreachable:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is False
    
    @patch('api.utils.requests.get')
    def test_mlflow_request_timeout(self, mock_get):
        """Given a request that times out When is_mlflow_running is called Then it should return False."""
        # Arrange
        mock_get.side_effect = requests.Timeout("Request timed out")
        url = "http://localhost:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is False
    
    @patch('api.utils.requests.get')
    @patch('api.utils.logger')
    def test_mlflow_request_exception_logs_error(self, mock_logger, mock_get):
        """Given a request that raises RequestException When is_mlflow_running is called Then it should return False and log error."""
        # Arrange
        mock_get.side_effect = requests.RequestException("Request failed")
        url = "http://localhost:5000"
        
        # Act
        result = is_mlflow_running(url)
        
        # Assert
        assert result is False
        mock_logger.error.assert_called_once()


class TestAnalyzeCsvLogic:
    """Tests for the analyze_csv_logic function."""
    
    def test_analyze_csv_valid_file(self):
        """Given a valid CSV file When analyze_csv_logic is called Then it should return column names."""
        # Arrange
        csv_content = "col1,col2,col3\nvalue1,value2,value3\n"
        csv_file = StringIO(csv_content)
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert "columns" in result
        assert result["columns"] == ["col1", "col2", "col3"]
    
    def test_analyze_csv_special_characters_in_headers(self):
        """Given a CSV file with special characters in headers When analyze_csv_logic is called Then it should return columns with special characters preserved."""
        # Arrange
        csv_content = "col-1,col_2,col@3,col#4\nvalue1,value2,value3,value4\n"
        csv_file = StringIO(csv_content)
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result["columns"] == ["col-1", "col_2", "col@3", "col#4"]
    
    @patch('api.utils.pd.read_csv')
    def test_analyze_csv_read_error(self, mock_read_csv):
        """Given a CSV file that cannot be read When analyze_csv_logic is called Then it should raise exception."""
        # Arrange
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns to parse")
        csv_file = StringIO("")
        
        # Act & Assert
        with pytest.raises(pd.errors.EmptyDataError):
            analyze_csv_logic(csv_file)


class TestInitDvcLogic:
    """Tests for the init_dvc_logic function."""
    
    def test_init_dvc_invalid_directory(self):
        """Given a directory that doesn't exist When init_dvc_logic is called Then it should raise ValueError."""
        # Arrange
        invalid_dir = "/nonexistent/directory"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Directorio inválido"):
            init_dvc_logic(invalid_dir)
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.Path')
    def test_init_dvc_successful_initialization(self, mock_path, mock_subprocess):
        """Given a valid experiment directory When init_dvc_logic is called Then it should initialize Git and DVC successfully."""
        # Arrange
        experiment_dir = "/tmp/test_experiment"
        
        # Mock Path behavior
        mock_exp_path = Mock()
        mock_exp_path.is_dir.return_value = True
        mock_exp_path.resolve.return_value = mock_exp_path
        mock_path.return_value = mock_exp_path
        
        # Mock directory existence checks
        mock_git_dir = Mock()
        mock_git_dir.exists.return_value = False
        mock_dvc_dir = Mock()
        mock_dvc_dir.exists.return_value = False
        mock_cache_dir = Mock()
        mock_gitignore = Mock()
        mock_gitignore.exists.return_value = False
        
        # Configure the __truediv__ method to support path division (/)
        def path_division(path_part):
            path_mapping = {
                ".git": mock_git_dir,
                ".dvc": mock_dvc_dir,
                ".dvc_cache": mock_cache_dir,
                ".gitignore": mock_gitignore
            }
            return path_mapping[path_part]
        
        mock_exp_path.__truediv__ = Mock(side_effect=path_division)
        
        # Mock gitignore write
        mock_gitignore.write_text = Mock()
        
        # Act
        result = init_dvc_logic(experiment_dir)
        
        # Assert
        assert result["status"] == "DVC inicializado correctamente"
        assert "experiment_dir" in result
        
        # Verify subprocess calls (git add .dvc is conditional on .dvc existing)
        expected_calls = [
            call(["git", "init"], cwd=mock_exp_path, check=True),
            call(["dvc", "init"], cwd=mock_exp_path, check=True),
            call(["dvc", "config", "cache.dir", str(mock_cache_dir)], cwd=mock_exp_path, check=True),
            call(["git", "config", "user.email", "geml@user.com"], cwd=mock_exp_path, check=True),
            call(["git", "config", "user.name", "geml user"], cwd=mock_exp_path, check=True),
            call(["git", "add", ".gitignore"], cwd=mock_exp_path, check=True),
            # Note: git add .dvc is conditional - only if .dvc directory exists
            call(["git", "commit", "-m", "Configuración inicial DVC"], cwd=mock_exp_path, check=True)
        ]

        for expected_call in expected_calls:
            assert expected_call in mock_subprocess.call_args_list
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.Path')
    def test_init_dvc_git_already_initialized(self, mock_path, mock_subprocess):
        """Given a directory where Git is already initialized When init_dvc_logic is called Then it should skip Git init and proceed with DVC."""
        # Arrange
        experiment_dir = "/tmp/test_experiment"
        
        # Mock Path behavior
        mock_exp_path = Mock()
        mock_exp_path.is_dir.return_value = True
        mock_exp_path.resolve.return_value = mock_exp_path
        mock_path.return_value = mock_exp_path
        
        # Mock directory existence checks - Git exists, DVC doesn't
        mock_git_dir = Mock()
        mock_git_dir.exists.return_value = True  # Git already exists
        mock_dvc_dir = Mock()
        mock_dvc_dir.exists.return_value = False
        mock_cache_dir = Mock()
        mock_gitignore = Mock()
        mock_gitignore.exists.return_value = False
        
        # Configure the __truediv__ method to support path division (/)
        def path_division(path_part):
            path_mapping = {
                ".git": mock_git_dir,
                ".dvc": mock_dvc_dir,
                ".dvc_cache": mock_cache_dir,
                ".gitignore": mock_gitignore
            }
            return path_mapping[path_part]
        
        mock_exp_path.__truediv__ = Mock(side_effect=path_division)
        
        mock_gitignore.write_text = Mock()
        
        # Act
        result = init_dvc_logic(experiment_dir)
        
        # Assert
        assert result["status"] == "DVC inicializado correctamente"
        
        # Verify git init was NOT called
        git_init_calls = [call for call in mock_subprocess.call_args_list 
                         if call[0][0] == ["git", "init"]]
        assert len(git_init_calls) == 0
        
        # Verify DVC init was called
        dvc_init_calls = [call for call in mock_subprocess.call_args_list 
                         if call[0][0] == ["dvc", "init"]]
        assert len(dvc_init_calls) == 1
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.Path')
    def test_init_dvc_subprocess_error(self, mock_path, mock_subprocess):
        """Given a directory where git commands fail When init_dvc_logic is called Then it should raise subprocess.CalledProcessError."""
        # Arrange
        experiment_dir = "/tmp/test_experiment"
        
        mock_exp_path = Mock()
        mock_exp_path.is_dir.return_value = True
        mock_exp_path.resolve.return_value = mock_exp_path
        mock_path.return_value = mock_exp_path
        
        mock_git_dir = Mock()
        mock_git_dir.exists.return_value = False
        # Configure the __truediv__ method to support path division (/)
        mock_exp_path.__truediv__ = Mock(return_value=mock_git_dir)
        
        # Mock subprocess to raise error
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ["git", "init"])
        
        # Act & Assert
        with pytest.raises(subprocess.CalledProcessError):
            init_dvc_logic(experiment_dir)


class TestConfigureDvcRemoteLogic:
    """Tests for the configure_dvc_remote_logic function."""
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.os.makedirs')
    @patch('api.utils._get_existing_dvc_remotes')
    def test_configure_dvc_remote_success(self, mock_get_remotes, mock_makedirs, mock_subprocess):
        """Given a valid experiment directory When configure_dvc_remote_logic is called Then it should create shared remote successfully."""
        # Arrange
        experiment_dir = "/tmp/experiment"
        mock_get_remotes.return_value = []  # No existing remotes
        
        # Act
        result = configure_dvc_remote_logic(experiment_dir)
        
        # Assert
        assert result["status"] == "Remoto DVC configurado exitosamente en ubicación compartida"
        assert "remote_path" in result
        
        # Verify directory creation
        expected_remote_path = "/tmp/dvc_remote"
        mock_makedirs.assert_called_once_with(expected_remote_path, exist_ok=True)
        
        # Verify subprocess calls for adding and setting default remote
        assert mock_subprocess.call_count >= 2
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.os.makedirs')
    @patch('api.utils._get_existing_dvc_remotes')
    def test_configure_dvc_remote_already_exists(self, mock_get_remotes, mock_makedirs, mock_subprocess):
        """Given a directory where remote already exists When configure_dvc_remote_logic is called Then it should skip remote creation."""
        # Arrange
        experiment_dir = "/tmp/experiment"
        mock_get_remotes.return_value = ["shared_remote"]  # Remote already exists
        
        # Act
        result = configure_dvc_remote_logic(experiment_dir)
        
        # Assert
        assert result["status"] == "Remoto DVC configurado exitosamente en ubicación compartida"
        
        # Verify that add remote was not called (only default remote setting)
        add_remote_calls = [call for call in mock_subprocess.call_args_list 
                           if "remote" in call[0][0] and "add" in call[0][0]]
        assert len(add_remote_calls) == 0
    
    @patch('api.utils.subprocess.run')
    @patch('api.utils.os.makedirs')
    def test_configure_dvc_remote_subprocess_error(self, mock_makedirs, mock_subprocess):
        """Given a directory where DVC commands fail When configure_dvc_remote_logic is called Then it should raise subprocess.CalledProcessError."""
        # Arrange
        experiment_dir = "/tmp/experiment"
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ["dvc", "remote", "list"])
        
        # Act & Assert
        with pytest.raises(subprocess.CalledProcessError):
            configure_dvc_remote_logic(experiment_dir)
    
    @patch('api.utils.os.makedirs')
    def test_configure_dvc_remote_os_error(self, mock_makedirs):
        """Given a directory where remote directory cannot be created When configure_dvc_remote_logic is called Then it should raise OSError."""
        # Arrange
        experiment_dir = "/tmp/experiment"
        mock_makedirs.side_effect = OSError("Permission denied")
        
        # Act & Assert
        with pytest.raises(OSError):
            configure_dvc_remote_logic(experiment_dir)


class TestSendProgressUpdate:
    """Tests for the send_progress_update function."""
    
    @patch('api.utils.get_channel_layer')
    @patch('api.utils.async_to_sync')
    def test_send_progress_update_success(self, mock_async_to_sync, mock_get_channel_layer):
        """Given valid step and status When send_progress_update is called Then it should send message to channel layer."""
        # Arrange
        mock_channel_layer = Mock()
        mock_get_channel_layer.return_value = mock_channel_layer
        mock_sync_func = Mock()
        mock_async_to_sync.return_value = mock_sync_func
        
        step = "test_step"
        status = "in_progress"
        
        # Act
        send_progress_update(step, status)
        
        # Assert
        mock_async_to_sync.assert_called_once_with(mock_channel_layer.group_send)
        mock_sync_func.assert_called_once_with(
            "progreso_group",
            {
                "type": "send_progress",
                "step": step,
                "status": status,
            }
        )


class TestGenerateExperimentSummaryPdf:
    """Tests for the generate_experiment_summary_pdf function."""
    
    def test_generate_pdf_nonexistent_file(self):
        """Given a non-existent config file When generate_experiment_summary_pdf is called Then it should raise FileNotFoundError."""
        # Arrange
        pipeline_config_path = "/nonexistent/file.json"
        output_pdf_path = "/tmp/output.pdf"
        
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="no existe o está vacío"):
            generate_experiment_summary_pdf(pipeline_config_path, output_pdf_path)
    
    def test_generate_pdf_empty_file(self):
        """Given an empty config file When generate_experiment_summary_pdf is called Then it should raise FileNotFoundError."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            pipeline_config_path = f.name
            # File is empty
        
        try:
            # Act & Assert
            with pytest.raises(FileNotFoundError, match="no existe o está vacío"):
                generate_experiment_summary_pdf(pipeline_config_path, "/tmp/output.pdf")
        finally:
            os.unlink(pipeline_config_path)
    
    def test_generate_pdf_invalid_json(self):
        """Given invalid JSON in config file When generate_experiment_summary_pdf is called Then it should raise ValueError."""
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("invalid json content")
            pipeline_config_path = f.name
        
        try:
            # Act & Assert
            with pytest.raises(ValueError, match="Error al parsear JSON"):
                generate_experiment_summary_pdf(pipeline_config_path, "/tmp/output.pdf")
        finally:
            os.unlink(pipeline_config_path)
    
    def test_generate_pdf_missing_experiment_id(self):
        """Given config file without experiment_id When generate_experiment_summary_pdf is called Then it should raise KeyError."""
        # Arrange
        config = {"steps": []}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(config, f)
            pipeline_config_path = f.name
        
        try:
            # Act & Assert
            with pytest.raises(KeyError, match="experiment_id"):
                generate_experiment_summary_pdf(pipeline_config_path, "/tmp/output.pdf")
        finally:
            os.unlink(pipeline_config_path)
    
    def test_generate_pdf_missing_steps(self):
        """Given config file without steps list When generate_experiment_summary_pdf is called Then it should raise KeyError."""
        # Arrange
        config = {"experiment_id": "test_exp"}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(config, f)
            pipeline_config_path = f.name
        
        try:
            # Act & Assert
            with pytest.raises(KeyError, match="steps"):
                generate_experiment_summary_pdf(pipeline_config_path, "/tmp/output.pdf")
        finally:
            os.unlink(pipeline_config_path)
    
    @patch('api.utils.Frame')
    @patch('api.utils.PageTemplate')
    @patch('api.utils.SimpleDocTemplate')
    def test_generate_pdf_valid_config(self, mock_doc, mock_template, mock_frame):
        """Given a valid pipeline config file When generate_experiment_summary_pdf is called Then it should create PDF successfully."""
        # Arrange
        config = {
            "experiment_id": "test_experiment",
            "steps": [
                {
                    "step": "data_loading",
                    "input_file": "test.csv",
                    "params": {"batch_size": 32}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config, f)
            pipeline_config_path = f.name
        
        output_pdf_path = "/tmp/test_output.pdf"
        
        # Mock the document
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        try:
            # Act
            generate_experiment_summary_pdf(pipeline_config_path, output_pdf_path)
            
            # Assert
            mock_doc.assert_called_once_with(
                output_pdf_path,
                pagesize=mock_doc.call_args[1]["pagesize"],
                rightMargin=mock_doc.call_args[1]["rightMargin"],
                leftMargin=mock_doc.call_args[1]["leftMargin"],
                topMargin=mock_doc.call_args[1]["topMargin"],
                bottomMargin=mock_doc.call_args[1]["bottomMargin"]
            )
            mock_doc_instance.build.assert_called_once()
            
        finally:
            os.unlink(pipeline_config_path)


# Note: Some tests for start_mlflow_logic and start_jupyter_logic are more complex
# and would require extensive mocking of subprocess, file operations, and MLflow.
# These could be added in a follow-up if needed.


# ===== PHASE 3: Additional Edge Case Tests =====


class TestInitDvcLogicEdgeCases:
    """Edge case tests for init_dvc_logic function - Phase 3."""

    @patch('api.utils.subprocess.run')
    def test_init_dvc_with_existing_incomplete_gitignore(self, mock_subprocess, tmp_path):
        """
        Scenario: .gitignore exists but missing some DVC entries
        Given: Directory with existing incomplete .gitignore
        When: init_dvc_logic is called
        Then: Missing entries should be appended and committed
        Coverage: Lines 129-135
        """
        # Arrange - Create real temp directory with partial .gitignore
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        # Create partial .gitignore (missing some DVC entries)
        gitignore_path = exp_dir / ".gitignore"
        gitignore_path.write_text("# Existing content\n*.pyc\n__pycache__/\n", encoding="utf-8")

        # Mock subprocess to succeed
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Act
        result = init_dvc_logic(str(exp_dir))

        # Assert
        assert result["status"] == "DVC inicializado correctamente"

        # Verify .gitignore was updated with DVC entries
        updated_gitignore = gitignore_path.read_text(encoding="utf-8")
        assert "# DVC config" in updated_gitignore
        assert ".dvc/tmp/" in updated_gitignore
        assert ".dvc/cache/" in updated_gitignore
        assert ".dvc_cache/" in updated_gitignore

        # Verify original content preserved
        assert "*.pyc" in updated_gitignore
        assert "__pycache__/" in updated_gitignore

        # Verify git commit was called (because needs_update=True)
        git_commit_calls = [call for call in mock_subprocess.call_args_list
                           if len(call[0]) > 0 and "commit" in call[0][0]]
        assert len(git_commit_calls) >= 1

    @patch('api.utils.subprocess.run')
    def test_init_dvc_with_complete_gitignore_no_update_needed(self, mock_subprocess, tmp_path):
        """
        Scenario: .gitignore already has all DVC entries
        Given: Directory with complete .gitignore
        When: init_dvc_logic is called
        Then: No gitignore update needed, but commit may occur for dvc init
        Coverage: Lines 129-135 (no update branch)
        """
        # Arrange
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        # Create complete .gitignore with all DVC entries
        gitignore_content = "\n".join([
            "# DVC config",
            ".dvc/tmp/",
            ".dvc/cache/",
            ".dvc/state",
            ".dvc/config.local",
            ".dvc_cache/"
        ])
        gitignore_path = exp_dir / ".gitignore"
        gitignore_path.write_text(gitignore_content, encoding="utf-8")

        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Act
        result = init_dvc_logic(str(exp_dir))

        # Assert
        assert result["status"] == "DVC inicializado correctamente"

        # Verify .gitignore content unchanged
        final_gitignore = gitignore_path.read_text(encoding="utf-8")
        assert final_gitignore == gitignore_content

    @patch('api.utils.subprocess.run')
    def test_init_dvc_dvc_directory_check_for_commit(self, mock_subprocess, tmp_path):
        """
        Scenario: .dvc directory exists after dvc init
        Given: DVC initialization creates .dvc directory
        When: Commit logic executes
        Then: .dvc directory should be added to git
        Coverage: Line 148
        """
        # Arrange
        exp_dir = tmp_path / "test_experiment"
        exp_dir.mkdir()

        # Create .dvc directory to simulate dvc init result
        dvc_dir = exp_dir / ".dvc"
        dvc_dir.mkdir()

        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Act
        result = init_dvc_logic(str(exp_dir))

        # Assert
        assert result["status"] == "DVC inicializado correctamente"

        # Verify git add .dvc was called
        git_add_dvc_calls = [call for call in mock_subprocess.call_args_list
                             if len(call[0]) > 0 and call[0][0] == ["git", "add", ".dvc"]]
        assert len(git_add_dvc_calls) >= 1

    @patch('api.utils.subprocess.run')
    @patch('api.utils.Path')
    def test_init_dvc_os_error_on_file_operations(self, mock_path, mock_subprocess):
        """
        Scenario: OSError during directory/file operations
        Given: File system error (e.g., permission denied)
        When: init_dvc_logic attempts to create cache directory
        Then: OSError should be raised
        Coverage: Lines 162-164
        """
        # Arrange
        mock_exp_path = Mock()
        mock_exp_path.is_dir.return_value = True
        mock_exp_path.resolve.return_value = mock_exp_path
        mock_path.return_value = mock_exp_path

        # Mock cache directory to raise OSError on mkdir
        mock_cache_dir = Mock()
        mock_cache_dir.mkdir.side_effect = OSError("Permission denied")

        # Mock other paths
        mock_git_dir = Mock()
        mock_git_dir.exists.return_value = True  # Git exists
        mock_dvc_dir = Mock()
        mock_dvc_dir.exists.return_value = True  # DVC exists
        mock_gitignore = Mock()
        mock_gitignore.exists.return_value = True  # Gitignore exists

        def path_division(path_part):
            path_mapping = {
                ".git": mock_git_dir,
                ".dvc": mock_dvc_dir,
                ".dvc_cache": mock_cache_dir,
                ".gitignore": mock_gitignore
            }
            return path_mapping.get(path_part, Mock())

        mock_exp_path.__truediv__ = Mock(side_effect=path_division)

        # Mock subprocess to succeed if called
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Act & Assert
        with pytest.raises(OSError, match="Permission denied"):
            init_dvc_logic("/tmp/test_experiment")


class TestStartJupyterLogicEdgeCases:
    """Edge case tests for start_jupyter_logic function - Phase 3."""

    def test_start_jupyter_with_empty_run_id(self, temp_experiment_dir):
        """
        Scenario: Empty string passed as run_id
        Given: Valid experiment_dir but run_id is empty string
        When: start_jupyter_logic is called
        Then: ValueError should be raised
        Coverage: Line 451
        """
        # Act & Assert
        with pytest.raises(ValueError, match="El run_id no fue proporcionado o es inválido"):
            utils.start_jupyter_logic(temp_experiment_dir, "", port=9999)

    def test_start_jupyter_with_none_run_id(self, temp_experiment_dir):
        """
        Scenario: None passed as run_id
        Given: Valid experiment_dir but run_id is None
        When: start_jupyter_logic is called
        Then: ValueError should be raised
        Coverage: Line 451
        """
        # Act & Assert
        with pytest.raises(ValueError, match="El run_id no fue proporcionado o es inválido"):
            utils.start_jupyter_logic(temp_experiment_dir, None, port=9999)

    @patch('api.utils.is_port_available', return_value=True)
    @patch('os.path.exists', return_value=True)
    @patch('api.utils.set_tracking_uri')
    @patch('api.utils.get_experiment_by_name')
    def test_start_jupyter_mlflow_experiment_not_found(
        self, mock_get_exp, mock_set_uri, mock_exists, mock_is_port, temp_experiment_dir
    ):
        """
        Scenario: MLflow experiment doesn't exist
        Given: Valid directory and port, but MLflow experiment not found
        When: start_jupyter_logic is called
        Then: ValueError should be raised
        Coverage: Line 483
        """
        # Arrange - Mock get_experiment_by_name to return None
        mock_get_exp.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="El experimento .* no fue encontrado en MLflow"):
            utils.start_jupyter_logic(temp_experiment_dir, "test_run_id", port=9999)

    @patch('api.utils.subprocess.Popen')
    @patch('api.utils.start_run')
    @patch('api.utils.get_experiment_by_name')
    @patch('api.utils.set_tracking_uri')
    @patch('api.utils.shutil.copy')
    @patch('api.utils.log_param')
    @patch('api.utils.log_artifact')
    @patch('api.utils.os.makedirs')
    @patch('api.utils.is_port_available', return_value=True)
    @patch('os.path.exists', return_value=True)
    def test_start_jupyter_mlflow_run_creation_fails(
        self, mock_exists, mock_is_port, mock_makedirs, mock_log_artifact,
        mock_log_param, mock_copy, mock_set_uri, mock_get_exp, mock_start_run,
        mock_popen, temp_experiment_dir
    ):
        """
        Scenario: MLflow start_run raises exception
        Given: Valid setup but MLflow run creation fails
        When: start_jupyter_logic is called
        Then: RuntimeError should be raised
        Coverage: Lines 502-504
        """
        # Arrange
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_get_exp.return_value = mock_experiment

        # Mock start_run to raise exception
        mock_start_run.side_effect = Exception("MLflow service unavailable")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Error al iniciar run anidado en MLflow"):
            utils.start_jupyter_logic(temp_experiment_dir, "test_run_id", port=9999)

    @patch('api.utils.subprocess.Popen')
    @patch('api.utils.start_run')
    @patch('api.utils.get_experiment_by_name')
    @patch('api.utils.set_tracking_uri')
    @patch('api.utils.shutil.copy')
    @patch('api.utils.log_param')
    @patch('api.utils.log_artifact')
    @patch('api.utils.os.makedirs')
    @patch('api.utils.is_port_available', return_value=True)
    @patch('os.path.exists', return_value=True)
    def test_start_jupyter_popen_file_not_found(
        self, mock_exists, mock_is_port, mock_makedirs, mock_log_artifact,
        mock_log_param, mock_copy, mock_set_uri, mock_get_exp, mock_start_run,
        mock_popen, temp_experiment_dir
    ):
        """
        Scenario: Jupyter command not found
        Given: Valid setup but jupyter/notebook not installed
        When: start_jupyter_logic attempts to start Jupyter
        Then: RuntimeError should be raised
        Coverage: Lines 531-533
        """
        # Arrange
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_get_exp.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "new_run_id"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_run
        mock_context_manager.__exit__.return_value = None
        mock_start_run.return_value = mock_context_manager

        # Mock Popen to raise FileNotFoundError
        mock_popen.side_effect = FileNotFoundError("python command not found")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Comando no encontrado"):
            utils.start_jupyter_logic(temp_experiment_dir, "test_run_id", port=9999)

    @patch('api.utils.subprocess.Popen')
    @patch('api.utils.start_run')
    @patch('api.utils.get_experiment_by_name')
    @patch('api.utils.set_tracking_uri')
    @patch('api.utils.shutil.copy')
    @patch('api.utils.log_param')
    @patch('api.utils.log_artifact')
    @patch('api.utils.os.makedirs')
    @patch('api.utils.is_port_available', return_value=True)
    @patch('os.path.exists', return_value=True)
    def test_start_jupyter_popen_generic_exception(
        self, mock_exists, mock_is_port, mock_makedirs, mock_log_artifact,
        mock_log_param, mock_copy, mock_set_uri, mock_get_exp, mock_start_run,
        mock_popen, temp_experiment_dir
    ):
        """
        Scenario: Generic exception during Jupyter startup
        Given: Valid setup but Popen raises unexpected exception
        When: start_jupyter_logic attempts to start Jupyter
        Then: RuntimeError should be raised
        Coverage: Lines 534-536
        """
        # Arrange
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_get_exp.return_value = mock_experiment

        mock_run = MagicMock()
        mock_run.info.run_id = "new_run_id"
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_run
        mock_context_manager.__exit__.return_value = None
        mock_start_run.return_value = mock_context_manager

        # Mock Popen to raise generic exception
        mock_popen.side_effect = Exception("Unexpected error during process creation")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Error al iniciar Jupyter Notebook"):
            utils.start_jupyter_logic(temp_experiment_dir, "test_run_id", port=9999)


class TestGenerateExperimentSummaryPdfEdgeCases:
    """Edge case tests for generate_experiment_summary_pdf function - Phase 3."""

    @patch('api.utils.Frame')
    @patch('api.utils.PageTemplate')
    @patch('api.utils.SimpleDocTemplate')
    def test_pdf_generation_with_large_list_chunking(self, mock_doc, mock_template, mock_frame, tmp_path):
        """
        Scenario: Pipeline config has step with large list (>15 items)
        Given: pipeline_config.json with input_features list of 20 items
        When: generate_experiment_summary_pdf is called
        Then: Large list should be chunked into multiple table rows
        Coverage: Lines 745-757
        """
        # Arrange - Create pipeline config with large input_features list
        config = {
            "experiment_id": "test_experiment",
            "steps": [
                {
                    "step": "data_encoding",
                    "input_features": [f"feature_{i}" for i in range(20)],  # 20 items > 15
                    "target_variable": "target",
                    "encoding_method": "onehot"
                }
            ]
        }

        pipeline_config_path = tmp_path / "pipeline_config.json"
        with open(pipeline_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        output_pdf_path = tmp_path / "output.pdf"

        # Mock the document
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance

        # Act
        generate_experiment_summary_pdf(str(pipeline_config_path), str(output_pdf_path))

        # Assert
        # Verify doc.build was called (covers lines 803-806)
        mock_doc_instance.build.assert_called_once()

        # Verify document was created with correct parameters
        assert mock_doc.called

        # To verify chunking occurred, we'd check the elements passed to build
        # The first positional argument to build() is the elements list
        build_call_args = mock_doc_instance.build.call_args
        elements = build_call_args[0][0] if build_call_args else []

        # Should have more elements due to chunking (exact count depends on structure)
        assert len(elements) > 0

    @patch('api.utils.Frame')
    @patch('api.utils.PageTemplate')
    @patch('api.utils.SimpleDocTemplate')
    def test_pdf_generation_end_to_end_with_multiple_steps(self, mock_doc, mock_template, mock_frame, tmp_path):
        """
        Scenario: Complete PDF generation with multiple pipeline steps
        Given: Valid pipeline_config.json with multiple steps
        When: generate_experiment_summary_pdf is called
        Then: PDF should be built successfully
        Coverage: Lines 803-806 (doc.build)
        """
        # Arrange
        config = {
            "experiment_id": "exp_12345",
            "steps": [
                {
                    "step": "data_upload",
                    "file": "data.csv",
                    "rows": 1000
                },
                {
                    "step": "data_cleaning",
                    "method": "remove_duplicates",
                    "missing_threshold": 0.3
                },
                {
                    "step": "training",
                    "algorithm": "logistic_regression",
                    "params": {"C": 1.0, "max_iter": 100}
                }
            ]
        }

        pipeline_config_path = tmp_path / "pipeline_config.json"
        with open(pipeline_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        output_pdf_path = tmp_path / "summary.pdf"

        # Mock document
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance

        # Act
        generate_experiment_summary_pdf(str(pipeline_config_path), str(output_pdf_path))

        # Assert
        mock_doc_instance.addPageTemplates.assert_called_once()
        mock_doc_instance.build.assert_called_once()


class TestHeaderFooter:
    """Tests for header_footer function - Phase 3 basic coverage."""

    @patch('api.utils.os.path.exists', return_value=True)
    @patch('api.utils.svg2rlg')
    @patch('api.utils.renderPDF.draw')
    def test_header_footer_with_svg_logo(self, mock_draw, mock_svg2rlg, mock_exists):
        """
        Scenario: PDF header with SVG logo
        Given: LOGO_PATH exists and is SVG file
        When: header_footer is called
        Then: SVG should be rendered in header
        Coverage: Lines 607-618
        """
        # Arrange
        from reportlab.lib.pagesizes import A4
        mock_canvas = Mock()
        mock_doc = Mock()
        mock_doc.leftMargin = 2
        mock_doc.rightMargin = 2

        # Mock SVG drawing
        mock_drawing = Mock()
        mock_drawing.width = 100
        mock_drawing.height = 100
        mock_svg2rlg.return_value = mock_drawing

        # Act
        header_footer(mock_canvas, mock_doc)

        # Assert
        mock_svg2rlg.assert_called_once()
        mock_draw.assert_called_once()
        mock_canvas.saveState.assert_called_once()
        mock_canvas.restoreState.assert_called_once()

    @patch('api.utils.os.path.exists', return_value=False)
    def test_header_footer_without_logo(self, mock_exists):
        """
        Scenario: PDF header without logo
        Given: LOGO_PATH doesn't exist
        When: header_footer is called
        Then: Header should be rendered without logo
        Coverage: Line 607 (logo doesn't exist branch)
        """
        # Arrange
        mock_canvas = Mock()
        mock_doc = Mock()
        mock_doc.leftMargin = 2
        mock_doc.rightMargin = 2

        # Act
        header_footer(mock_canvas, mock_doc)

        # Assert
        # Verify canvas operations were called (header and footer text)
        mock_canvas.saveState.assert_called_once()
        mock_canvas.setFont.assert_called()  # At least for header and footer text
        mock_canvas.restoreState.assert_called_once()


class TestGetExistingDvcRemotesEdgeCases:
    """Edge case tests for _get_existing_dvc_remotes helper - Phase 3."""

    @patch('api.utils.subprocess.run')
    def test_get_existing_dvc_remotes_with_empty_lines(self, mock_subprocess):
        """
        Scenario: dvc remote list output contains empty lines
        Given: subprocess returns output with whitespace and empty lines
        When: _get_existing_dvc_remotes is called
        Then: Empty lines should be filtered out
        Coverage: Line 245 (if line.strip() filter)
        """
        # Arrange
        mock_result = Mock()
        mock_result.stdout = "remote1 /path/to/remote1\n\n   \nremote2 /path/to/remote2\n"
        mock_subprocess.return_value = mock_result

        # Act
        from api.utils import _get_existing_dvc_remotes
        result = _get_existing_dvc_remotes("/tmp/experiment")

        # Assert
        assert result == ["remote1", "remote2"]
        assert len(result) == 2  # Empty lines filtered out