# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.

import os
import json
import tempfile
import subprocess
from io import BytesIO
from unittest.mock import patch, MagicMock, Mock
import pytest
from django.test import RequestFactory
from django.http import JsonResponse
from django.core.files.uploadedfile import SimpleUploadedFile

# Import the view function to test
from apiTimeSeries.views import upload_and_clean_csv


class TestUploadAndCleanCSV:
    """Test class for upload_and_clean_csv Django view."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.factory = RequestFactory()
        self.valid_csv_content = b"col1,col2,col3\n1,2,3\n4,5,6\n"
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_csv_file(self, filename="test.csv", content=None, size_mb=None):
        """Helper method to create mock CSV files."""
        if size_mb:
            # Create file of specific size
            content = b"x" * (size_mb * 1024 * 1024)
        elif content is None:
            content = self.valid_csv_content
        
        return SimpleUploadedFile(filename, content, content_type="text/csv")
    
    # Happy Path Scenarios
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_01_successful_csv_upload_default_params(self, mock_isdir, mock_service):
        """Test successful CSV upload with default parameters."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {
            "status": "success",
            "run_id": "test_run_123",
            "raw_file_path": "/path/to/file.csv"
        }
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'run_id' in response_data
        mock_service.upload_and_clean_csv_logic.assert_called_once()
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_02_successful_upload_custom_cleaning_params(self, mock_isdir, mock_service):
        """Test successful upload with custom cleaning parameters."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'eliminar_duplicados': 'true',
            'filtrar_outliers': 'true',
            'relleno_valores_numericos': 'media'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
        mock_service.upload_and_clean_csv_logic.assert_called_once()
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_03_successful_upload_custom_imputation_value(self, mock_isdir, mock_service):
        """Test successful upload with custom imputation value."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'relleno_valores_numericos': 'valor',
            'valor_imputacion': '5.5'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
        mock_service.upload_and_clean_csv_logic.assert_called_once()
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_04_boolean_parameter_variations(self, mock_isdir, mock_service):
        """Test boolean parameter parsing with mixed case."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'eliminar_duplicados': 'True',
            'filtrar_outliers': 'FALSE'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_06_upload_at_max_file_size(self, mock_isdir, mock_service):
        """Test upload with file exactly at 10MB limit."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        # Create file exactly at 10MB
        csv_file = self.create_csv_file(size_mb=10)
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
    
    # HTTP Method Validation Scenarios
    
    def test_scenario_07_invalid_http_method_get(self):
        """Test GET request returns 405."""
        # Arrange
        request = self.factory.get('/upload-and-clean-csv/')
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Método no permitido'
    
    def test_scenario_08_invalid_http_method_put(self):
        """Test PUT request returns 405."""
        # Arrange
        request = self.factory.put('/upload-and-clean-csv/')
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Método no permitido'
    
    def test_scenario_09_invalid_http_method_delete(self):
        """Test DELETE request returns 405."""
        # Arrange
        request = self.factory.delete('/upload-and-clean-csv/')
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 405
    
    def test_scenario_10_post_without_file_upload(self):
        """Test POST request without file returns 405."""
        # Arrange
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        # Deliberately not adding request.FILES['file']
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 405
    
    # Experiment Directory Validation Scenarios
    
    def test_scenario_12_missing_experiment_directory(self):
        """Test missing experiment_dir parameter."""
        # Arrange
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/')
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'La ruta del experimento no se proporcionó.'
    
    def test_scenario_13_empty_experiment_directory(self):
        """Test empty experiment_dir parameter."""
        # Arrange
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': '',
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'La ruta del experimento no se proporcionó.'
    
    @patch('os.path.isdir')
    def test_scenario_14_non_existent_experiment_directory(self, mock_isdir):
        """Test non-existent experiment directory."""
        # Arrange
        mock_isdir.return_value = False
        non_existent_path = '/non/existent/path'
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': non_existent_path,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        expected_message = f"La ruta '{non_existent_path}' no existe o no es un directorio válido."
        assert response_data['status'] == expected_message
    
    # File Validation Scenarios
    
    @patch('os.path.isdir')
    def test_scenario_16_non_csv_file_txt(self, mock_isdir):
        """Test .txt file returns 400."""
        # Arrange
        mock_isdir.return_value = True
        txt_file = SimpleUploadedFile("test.txt", b"content", content_type="text/plain")
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = txt_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo debe ser un CSV.'
    
    @patch('os.path.isdir')
    def test_scenario_17_non_csv_file_xlsx(self, mock_isdir):
        """Test .xlsx file returns 400."""
        # Arrange
        mock_isdir.return_value = True
        xlsx_file = SimpleUploadedFile("test.xlsx", b"content")
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = xlsx_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo debe ser un CSV.'
    
    @patch('os.path.isdir')
    def test_scenario_18_file_without_extension(self, mock_isdir):
        """Test file without extension returns 400."""
        # Arrange
        mock_isdir.return_value = True
        no_ext_file = SimpleUploadedFile("data", b"content")
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = no_ext_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo debe ser un CSV.'
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_19_file_with_multiple_extensions(self, mock_isdir, mock_service):
        """Test file with multiple extensions (data.txt.csv) is accepted."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        multi_ext_file = SimpleUploadedFile("data.txt.csv", self.valid_csv_content)
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = multi_ext_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
    
    @patch('os.path.isdir')
    def test_scenario_20_case_sensitivity_file_extension(self, mock_isdir):
        """Test uppercase .CSV extension returns 400."""
        # Arrange
        mock_isdir.return_value = True
        csv_file = SimpleUploadedFile("data.CSV", self.valid_csv_content)
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo debe ser un CSV.'
    
    @patch('os.path.isdir')
    def test_scenario_21_file_exceeds_10mb_limit(self, mock_isdir):
        """Test file larger than 10MB returns 400."""
        # Arrange
        mock_isdir.return_value = True
        # Create file slightly over 10MB
        large_file = self.create_csv_file(size_mb=11)
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = large_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo excede el tamaño máximo permitido de 10MB.'
    
    # Parameter Validation Scenarios
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_23_invalid_boolean_eliminar_duplicados(self, mock_isdir, mock_service):
        """Test invalid boolean parameter is treated as false."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'eliminar_duplicados': 'invalid'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200  # Should proceed successfully
    
    @patch('os.path.isdir')
    def test_scenario_25_missing_imputation_value_when_required(self, mock_isdir):
        """Test missing imputation value when required returns 400."""
        # Arrange
        mock_isdir.return_value = True
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'relleno_valores_numericos': 'valor'
            # Missing valor_imputacion
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        expected_message = "Se requiere 'valor_imputacion' cuando 'relleno_valores_numericos' es 'valor'."
        assert response_data['status'] == expected_message
    
    @patch('os.path.isdir')
    def test_scenario_27_invalid_numeric_imputation_value(self, mock_isdir):
        """Test invalid numeric imputation value returns 400."""
        # Arrange
        mock_isdir.return_value = True
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'relleno_valores_numericos': 'valor',
            'valor_imputacion': 'abc'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El valor de imputación debe ser un número.'
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_29_valid_negative_imputation_value(self, mock_isdir, mock_service):
        """Test valid negative imputation value."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'relleno_valores_numericos': 'valor',
            'valor_imputacion': '-3.14'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_30_valid_zero_imputation_value(self, mock_isdir, mock_service):
        """Test valid zero imputation value."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.return_value = {"status": "success"}
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
            'relleno_valores_numericos': 'valor',
            'valor_imputacion': '0'
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 200
    
    # Service Layer Exception Scenarios
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_37_dvc_subprocess_error(self, mock_isdir, mock_service):
        """Test DVC subprocess error returns 500."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.side_effect = subprocess.CalledProcessError(
            1, 'dvc', 'DVC command failed'
        )
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error en DVC:' in response_data['status']
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_38_runtime_error_in_service(self, mock_isdir, mock_service):
        """Test RuntimeError in service layer returns 500."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.side_effect = RuntimeError('Service runtime error')
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error de ejecución:' in response_data['status']
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_39_value_error_in_service(self, mock_isdir, mock_service):
        """Test ValueError in service layer returns 400."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.side_effect = ValueError('Invalid value')
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'Valor inválido:' in response_data['status']
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_40_file_not_found_error_in_service(self, mock_isdir, mock_service):
        """Test FileNotFoundError in service layer returns 500."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.side_effect = FileNotFoundError('File not found')
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Archivo no encontrado:' in response_data['status']
    
    @patch('apiTimeSeries.views.preProcessingService')
    @patch('os.path.isdir')
    def test_scenario_41_generic_exception_in_service(self, mock_isdir, mock_service):
        """Test generic Exception in service layer returns 500."""
        # Arrange
        mock_isdir.return_value = True
        mock_service.upload_and_clean_csv_logic.side_effect = Exception('Unexpected error')
        
        csv_file = self.create_csv_file()
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': self.temp_dir,
        })
        request.FILES['file'] = csv_file
        
        # Act
        response = upload_and_clean_csv(request)
        
        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error inesperado:' in response_data['status']