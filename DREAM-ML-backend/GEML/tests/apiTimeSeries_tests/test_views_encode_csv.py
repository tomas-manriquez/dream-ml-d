import pytest
import json
import subprocess
from unittest.mock import Mock, patch, MagicMock
from django.test import RequestFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import JsonResponse
import mlflow

from apiTimeSeries.views import encode_csv


class TestEncodeCSVView:
    """Test suite for encode_csv Django view function"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.factory = RequestFactory()
        self.csv_content = b"col1,col2,col3\n1,2,3\n4,5,6"
        self.csv_file = SimpleUploadedFile(
            "test.csv", 
            self.csv_content, 
            content_type="text/csv"
        )
    
    # HTTP Method Validation Tests
    
    def test_non_post_request_rejection(self):
        """Scenario 1: Non-POST request rejection"""
        # Arrange
        request = self.factory.get('/encode_csv/')
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data["status"] == "Método no permitido."
    
    # File Upload Validation Tests
    
    def test_missing_csv_file(self):
        """Scenario 2: Missing CSV file"""
        # Arrange
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1,feature2',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "Archivo CSV no recibido."
    
    # Required Parameters Validation Tests
    
    def test_missing_input_features_parameter(self):
        """Scenario 4: Missing input_features parameter"""
        # Arrange
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "Variables de entrada/salida no especificadas."
    
    def test_empty_input_features_parameter(self):
        """Scenario 5: Empty input_features parameter"""
        # Arrange
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': '',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "Variables de entrada/salida no especificadas."
    
    def test_missing_target_variables_parameter(self):
        """Scenario 6: Missing target_variables parameter"""
        # Arrange
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1,feature2',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "Variables de entrada/salida no especificadas."
    
    def test_empty_target_variables_parameter(self):
        """Scenario 7: Empty target_variables parameter"""
        # Arrange
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1,feature2',
            'target_variables': '',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data["status"] == "Variables de entrada/salida no especificadas."
    
    # Parameter Processing Tests
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_comma_separated_input_features_parsing(self, mock_encode_logic, mock_get_run):
        """Scenario 8: Comma-separated input features parsing"""
        # Arrange
        mock_get_run.return_value = Mock()  # Mock active run
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1, feature2, feature3',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        # Check that the service was called with properly parsed and trimmed features
        mock_encode_logic.assert_called_once()
        call_args = mock_encode_logic.call_args
        assert call_args[1]['input_features'] == ['feature1', 'feature2', 'feature3']
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_comma_separated_target_variables_parsing(self, mock_encode_logic, mock_get_run):
        """Scenario 9: Comma-separated target variables parsing"""
        # Arrange
        mock_get_run.return_value = Mock()  # Mock active run
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1, target2',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        # Check that the service was called with properly parsed and trimmed targets
        mock_encode_logic.assert_called_once()
        call_args = mock_encode_logic.call_args
        assert call_args[1]['target_variables'] == ['target1', 'target2']
    
    # Boolean Parameter Processing Tests
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_encode_target_ohe_default_value(self, mock_encode_logic, mock_get_run):
        """Scenario 10: encode_target_ohe default value"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
            # encode_target_ohe not provided
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        assert call_args[1]['apply_target_ohe'] is False
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_encode_target_ohe_true_value(self, mock_encode_logic, mock_get_run):
        """Scenario 11: encode_target_ohe true value"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123',
            'encode_target_ohe': 'true'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        assert call_args[1]['apply_target_ohe'] is True
    
    @pytest.mark.parametrize("ohe_value", ["TRUE", "True", "true"])
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_encode_target_ohe_case_insensitive(self, mock_encode_logic, mock_get_run, ohe_value):
        """Scenario 12: encode_target_ohe case insensitive"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123',
            'encode_target_ohe': ohe_value
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        assert call_args[1]['apply_target_ohe'] is True
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_encode_target_label_default_value(self, mock_encode_logic, mock_get_run):
        """Scenario 13: encode_target_label default value"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
            # encode_target_label not provided
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        assert call_args[1]['apply_target_label'] is False
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_encode_target_label_true_value(self, mock_encode_logic, mock_get_run):
        """Scenario 14: encode_target_label true value"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123',
            'encode_target_label': 'true'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        assert call_args[1]['apply_target_label'] is True
    
    # MLflow Run Validation Tests
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    def test_invalid_run_id(self, mock_get_run):
        """Scenario 15: Invalid run_id"""
        # Arrange
        mock_get_run.return_value = None  # Simulate run not found
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'invalid_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert "Run padre no encontrado o no activo: invalid_run_123" in response_data["status"]
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_valid_run_id(self, mock_encode_logic, mock_get_run):
        """Scenario 16: Valid run_id"""
        # Arrange
        mock_get_run.return_value = Mock()  # Mock active run
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'valid_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        mock_encode_logic.assert_called_once()
    
    # Service Layer Integration Tests
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_successful_encoding_process(self, mock_encode_logic, mock_get_run):
        """Scenario 17: Successful encoding process"""
        # Arrange
        mock_get_run.return_value = Mock()
        expected_result = {
            "status": "Archivo CSV codificado correctamente.",
            "processed_train_path": "processed/processed_train_test.csv",
            "run_id": "test_run_123"
        }
        mock_encode_logic.return_value = expected_result
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data == expected_result
    
    # Exception Handling Tests
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_file_not_found_error_handling(self, mock_encode_logic, mock_get_run):
        """Scenario 18: FileNotFoundError handling"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.side_effect = FileNotFoundError("Test file not found")
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert "Archivo no encontrado: Test file not found" in response_data["status"]
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_subprocess_called_process_error_handling(self, mock_encode_logic, mock_get_run):
        """Scenario 19: subprocess.CalledProcessError handling"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.side_effect = subprocess.CalledProcessError(1, 'dvc', 'DVC command failed')
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert "Error en DVC:" in response_data["status"]
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_value_error_handling(self, mock_encode_logic, mock_get_run):
        """Scenario 20: ValueError handling"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.side_effect = ValueError("Invalid value provided")
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert "Error de valor: Invalid value provided" in response_data["status"]
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_runtime_error_handling(self, mock_encode_logic, mock_get_run):
        """Scenario 21: RuntimeError handling"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.side_effect = RuntimeError("Runtime error occurred")
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert "Error de ejecución: Runtime error occurred" in response_data["status"]
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_generic_exception_handling(self, mock_encode_logic, mock_get_run):
        """Scenario 22: Generic exception handling"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.side_effect = Exception("Unexpected error")
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert "Error inesperado: Unexpected error" in response_data["status"]
    
    # Edge Cases
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    def test_missing_run_id_parameter(self, mock_get_run):
        """Scenario 24: Missing run_id parameter"""
        # Arrange
        mock_get_run.side_effect = Exception("Run ID is None")
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': 'feature1',
            'target_variables': 'target1'
            # run_id not provided
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 500  # Should be handled by generic exception handler
    
    @patch('apiTimeSeries.views.mlflow.get_run')
    @patch('apiTimeSeries.views.dataEncodingService.encode_csv_logic')
    def test_special_characters_in_feature_names(self, mock_encode_logic, mock_get_run):
        """Scenario 25: Special characters in feature names"""
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {"status": "success", "processed_train_path": "test", "run_id": "123"}
        
        request = self.factory.post('/encode_csv/', {
            'experiment_dir': '/path/to/experiment',
            'input_features': ' feature with spaces , feature-with-dashes , feature_with_underscores ',
            'target_variables': ' target with spaces ',
            'run_id': 'test_run_123'
        })
        request.FILES['file'] = self.csv_file
        
        # Act
        response = encode_csv(request)
        
        # Assert
        assert response.status_code == 200
        call_args = mock_encode_logic.call_args
        # Verify that features are properly trimmed
        assert call_args[1]['input_features'] == [
            'feature with spaces', 
            'feature-with-dashes', 
            'feature_with_underscores'
        ]
        assert call_args[1]['target_variables'] == ['target with spaces']