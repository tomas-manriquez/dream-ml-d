import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from django.test import RequestFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import JsonResponse

# Import the view function to test
from apiTimeSeries.views import train_model


class TestTrainModelView:
    """Test cases for the train_model Django view function."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.factory = RequestFactory()
        self.valid_data = {
            "experiment_dir": "/path/to/experiment",
            "algorithm": "logistic"
        }
        self.valid_file = SimpleUploadedFile(
            "test_dataset.csv", 
            b"col1,col2\n1,2\n3,4", 
            content_type="text/csv"
        )

    def test_invalid_http_method_returns_405(self):
        """
        Scenario 1: Invalid HTTP method
        Given a request is made to the train_model endpoint
        When the HTTP method is not POST (e.g., GET, PUT, DELETE)
        Then the response should return status 405 with error message "Método no permitido. Use POST"
        """
        # Arrange
        request = self.factory.get('/train_model/')
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Método no permitido. Use POST"
        assert response_data["error_code"] == "HTTP_405_METHOD_NOT_ALLOWED"

    def test_missing_file_upload_returns_400(self):
        """
        Scenario 2: Missing file upload
        Given a POST request is made to the train_model endpoint
        When no file is included in the request
        Then the response should return status 400 with error message about missing CSV file
        """
        # Arrange
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        })
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert "No se encontró archivo CSV" in response_data["error_details"]

    def test_missing_configuration_data_returns_400(self):
        """
        Scenario 3: Missing configuration data
        Given a POST request with a file upload
        When the 'data' field is missing from the request
        Then the response should return status 400 with error about missing configuration data
        """
        # Arrange
        request = self.factory.post('/train_model/', data={}, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert "Datos de configuración faltantes" in response_data["error_details"]

    def test_invalid_json_in_configuration_data_returns_400(self):
        """
        Scenario 4: Invalid JSON in configuration data
        Given a POST request with file and 'data' field
        When the 'data' field contains invalid JSON
        Then the response should return status 400 with JSON parsing error details
        """
        # Arrange
        request = self.factory.post('/train_model/', data={
            'data': 'invalid json {'
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Formato JSON inválido en datos de configuración"

    @patch('apiTimeSeries.views.os.path.isdir')
    def test_missing_experiment_directory_returns_404(self, mock_isdir):
        """
        Scenario 5: Missing experiment directory in data
        Given a POST request with valid file and JSON data
        When the 'experiment_dir' field is missing from the data
        Then the response should return status 404 with invalid experiment directory error
        """
        # Arrange
        mock_isdir.return_value = False
        data_without_exp_dir = {"algorithm": "logistic"}
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(data_without_exp_dir)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert "Directorio de experimento inválido" in response_data["error_details"]

    @patch('apiTimeSeries.views.os.path.isdir')
    def test_invalid_experiment_directory_path_returns_404(self, mock_isdir):
        """
        Scenario 6: Invalid experiment directory path
        Given a POST request with valid file and JSON data
        When the 'experiment_dir' points to a non-existent directory
        Then the response should return status 404 with invalid experiment directory error
        """
        # Arrange
        mock_isdir.return_value = False
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert "Directorio de experimento inválido" in response_data["error_details"]

    @patch('apiTimeSeries.views.os.path.isdir')
    def test_empty_experiment_directory_string_returns_404(self, mock_isdir):
        """
        Scenario 17: Empty experiment directory string
        Given a POST request with valid file and JSON data
        When the 'experiment_dir' field is an empty string
        Then the response should return status 404 with invalid experiment directory error
        """
        # Arrange
        mock_isdir.return_value = False
        empty_dir_data = {"experiment_dir": "", "algorithm": "logistic"}
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(empty_dir_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert "Directorio de experimento inválido" in response_data["error_details"]

    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    def test_active_mlflow_run_cleanup(self, mock_basename, mock_dirname, mock_isdir, mock_mlflow):
        """
        Scenario 8: Active MLflow run cleanup
        Given there is an active MLflow run before processing the request
        When the train_model function is called
        Then the active run should be ended before proceeding
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_mlflow.active_run.return_value = MagicMock()  # Simulate active run
        mock_mlflow.get_experiment_by_name.return_value = None  # Will cause ValueError
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        mock_mlflow.end_run.assert_called_once()  # Verify cleanup was called

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    @patch('apiTimeSeries.views.os.environ.get')
    def test_successful_model_training_default_url(self, mock_environ_get, mock_basename, 
                                                  mock_dirname, mock_isdir, mock_mlflow, 
                                                  mock_service):
        """
        Scenario 11: Default MLflow UI URL
        Given a successful training scenario
        When the MLFLOW_UI_URL environment variable is not set
        Then the response should use the default localhost:5000 URL
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_environ_get.return_value = None  # MLFLOW_UI_URL not set
        
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.active_run.return_value = None
        
        mock_service.train_model_logic.return_value = {
            "run_id": "run123",
            "val_metrics": {"accuracy": 0.95},
            "model_path": "/path/to/model"
        }
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data["status"] == "success"
        assert "http://localhost:5000" in response_data["mlflow_ui"]

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    @patch('apiTimeSeries.views.os.environ.get')
    def test_successful_model_training_custom_url(self, mock_environ_get, mock_basename, 
                                                 mock_dirname, mock_isdir, mock_mlflow, 
                                                 mock_service):
        """
        Scenario 10: Custom MLflow UI URL
        Given a successful training scenario
        When the MLFLOW_UI_URL environment variable is set to a custom URL
        Then the response should include the custom URL in the mlflow_ui field
        """
        # Arrange
        custom_url = "https://custom-mlflow.example.com"
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_environ_get.return_value = custom_url
        
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.active_run.return_value = None
        
        mock_service.train_model_logic.return_value = {
            "run_id": "run123",
            "val_metrics": {"accuracy": 0.95},
            "model_path": "/path/to/model"
        }
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data["status"] == "success"
        assert custom_url in response_data["mlflow_ui"]

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    def test_training_service_runtime_error_returns_500(self, mock_basename, mock_dirname, 
                                                       mock_isdir, mock_mlflow, mock_service):
        """
        Scenario 12: Training service runtime error
        Given a POST request with valid inputs
        When trainModelService.train_model_logic raises a RuntimeError
        Then the response should return status 500 with training error details
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_mlflow.active_run.return_value = None
        
        mock_service.train_model_logic.side_effect = RuntimeError("Training failed")
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Error durante el entrenamiento"

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    def test_training_service_value_error_returns_400(self, mock_basename, mock_dirname, 
                                                     mock_isdir, mock_mlflow, mock_service):
        """
        Scenario 13: Training service value error
        Given a POST request with valid inputs
        When trainModelService.train_model_logic raises a ValueError
        Then the response should return status 400 with parameter validation error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_mlflow.active_run.return_value = None
        
        mock_service.train_model_logic.side_effect = ValueError("Invalid parameter")
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Error en parámetros de entrada"

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    def test_training_service_file_not_found_error_returns_404(self, mock_basename, mock_dirname, 
                                                              mock_isdir, mock_mlflow, mock_service):
        """
        Scenario 14: Training service file not found error
        Given a POST request with valid inputs
        When trainModelService.train_model_logic raises a FileNotFoundError
        Then the response should return status 404 with resource not found error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_mlflow.active_run.return_value = None
        
        mock_service.train_model_logic.side_effect = FileNotFoundError("File not found")
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Recurso no encontrado"

    @patch('apiTimeSeries.views.trainModelService')
    @patch('apiTimeSeries.views.mlflow')
    @patch('apiTimeSeries.views.os.path.isdir')
    @patch('apiTimeSeries.views.os.path.dirname')
    @patch('apiTimeSeries.views.os.path.basename')
    def test_unexpected_exception_with_active_mlflow_run_cleanup(self, mock_basename, mock_dirname, 
                                                               mock_isdir, mock_mlflow, mock_service):
        """
        Scenario 15: Unexpected exception with active MLflow run
        Given a POST request being processed
        When an unexpected exception occurs and there's an active MLflow run
        Then the MLflow run should be ended and response should return status 500
        """
        # Arrange
        mock_isdir.return_value = True
        mock_dirname.return_value = "/path/to"
        mock_basename.return_value = "experiment"
        mock_mlflow.active_run.side_effect = [None, MagicMock()]  # No active run initially, then active during exception
        
        mock_service.train_model_logic.side_effect = Exception("Unexpected error")
        
        request = self.factory.post('/train_model/', data={
            'data': json.dumps(self.valid_data)
        }, files={'file': self.valid_file})
        
        # Act
        response = train_model(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data["status"] == "error"
        assert response_data["message"] == "Error interno del servidor"
        # The final exception handler should call end_run
        mock_mlflow.end_run.assert_called()