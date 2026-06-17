# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

import json
import os
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest

from django.test import RequestFactory, TestCase
from django.http import JsonResponse
from django.core.files.uploadedfile import SimpleUploadedFile

# Import views using direct import
try:
    from api import views
except ImportError:
    try:
        from GEML.api import views
    except ImportError:
        import views


@pytest.mark.django_db
class TestCreateExperimentView(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

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
            'experiment_dir': '/app/experimentos/test-experiment'
        }
        
        request = self.factory.post('/create-experiment/', 
                                  content_type='application/json')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 201)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Experimento creado exitosamente')
        self.assertIn('details', response_data)

    def test_method_not_allowed(self):
        # Arrange
        request = self.factory.get('/create-experiment/')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 405)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Método no permitido')

    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_invalid_base_directory(self, mock_environ_get, mock_isdir):
        # Arrange
        mock_environ_get.return_value = '/invalid/path'
        mock_isdir.return_value = False
        
        request = self.factory.post('/create-experiment/',
                                  content_type='application/json')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('Directorio base no encontrado', response_data['status'])

    @patch('api.views.create_experiment_logic')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_unexpected_exception(self, mock_environ_get, mock_isdir, mock_create_logic):
        # Arrange
        mock_environ_get.return_value = '/app/experimentos'
        mock_isdir.return_value = True
        mock_create_logic.side_effect = Exception('Unexpected error')
        
        request = self.factory.post('/create-experiment/',
                                  content_type='application/json')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Error interno al crear experimento')


"""
Phase 2A: Infrastructure Endpoint Tests (DVC + MLflow)
Tests for init_dvc(), configure_dvc_remote(), start_mlflow()
"""


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
        response = views.init_dvc(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert 'error' in response_data or 'status' in response_data


@pytest.mark.django_db
@pytest.mark.unit
class TestConfigureDvcRemoteView:
    """Tests for configure_dvc_remote() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.configure_dvc_remote_logic')
    @patch('os.path.isdir')
    def test_successful_remote_configuration(self, mock_isdir, mock_configure_logic):
        """
        Scenario: Successful DVC remote configuration
        Given valid S3 remote configuration
        When POST request with remote params
        Then should configure DVC remote and return success
        """
        # Arrange
        mock_isdir.return_value = True
        mock_configure_logic.return_value = {
            'status': 'success',
            'message': 'DVC remote configured',
            'remote_path': '/shared/dvc-storage'
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
    @patch('os.path.isdir')
    def test_successful_mlflow_start(self, mock_isdir, mock_start_logic):
        """
        Scenario: Successful MLflow server startup
        Given valid directory path
        When POST request to start_mlflow endpoint
        Then should start MLflow server and return success
        """
        # Arrange
        mock_isdir.return_value = True
        mock_start_logic.return_value = {
            'status': 'Servidor MLflow iniciado exitosamente',
            'backend_store_uri': 'sqlite:///test.db',
            'artifact_store': '/test/artifacts',
            'tracking_uri': 'http://localhost:5000'
        }

        request_data = {
            'directory_path': '/app/experimentos/exp123'
        }
        request = self.factory.post(
            '/start-mlflow/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_mlflow(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'status' in response_data
        mock_start_logic.assert_called_once_with('/app/experimentos/exp123')

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to start_mlflow endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/start-mlflow/')

        # Act
        response = views.start_mlflow(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert 'status' in response_data


"""
Phase 2B: Data Analysis & Upload Endpoint Tests
Tests for analyze_csv(), upload_and_clean_csv()
"""


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

        csv_content = b"feature1,feature2,target\n1.0,A,0\n2.0,B,1"
        csv_file = SimpleUploadedFile("test.csv", csv_content, content_type='text/csv')

        request = self.factory.post('/analyze-csv/', {'file': csv_file})

        # Act
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
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert 'error' in response_data

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to analyze_csv endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/analyze-csv/')

        # Act
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['error'] == 'Método no permitido'

    def test_non_csv_file_type_returns_400(self):
        """
        Scenario: Non-CSV file type submitted
        Given uploaded TXT file instead of CSV
        When POST request with non-CSV file
        Then should return 400 Bad Request with validation error
        """
        # Arrange
        txt_file = SimpleUploadedFile('test.txt', b'not a csv', content_type='text/plain')
        request = self.factory.post('/analyze-csv/', {'file': txt_file})

        # Act
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['error'] == 'El archivo debe ser un CSV.'

    def test_file_size_exceeds_maximum(self):
        """
        Scenario: File size exceeds 10MB limit
        Given CSV file larger than 10MB
        When POST request with oversized file
        Then should return 400 Bad Request with size error
        """
        # Arrange
        large_content = b'a' * (11 * 1024 * 1024)  # 11MB
        large_file = SimpleUploadedFile('large.csv', large_content, content_type='text/csv')
        request = self.factory.post('/analyze-csv/', {'file': large_file})

        # Act
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['error'] == 'El archivo excede el tamaño máximo permitido de 10MB.'

    @patch('api.views.analyze_csv_logic')
    def test_empty_csv_file_returns_400(self, mock_analyze_logic):
        """
        Scenario: Empty CSV file submitted
        Given empty CSV file
        When POST request with empty file
        Then should return 400 Bad Request with empty data error
        """
        # Arrange
        import pandas as pd
        mock_analyze_logic.side_effect = pd.errors.EmptyDataError('No data')

        csv_file = SimpleUploadedFile('empty.csv', b'', content_type='text/csv')
        request = self.factory.post('/analyze-csv/', {'file': csv_file})

        # Act
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['error'] == 'El archivo CSV está vacío.'

    @patch('api.views.analyze_csv_logic')
    def test_malformed_csv_returns_400(self, mock_analyze_logic):
        """
        Scenario: Malformed CSV file submitted
        Given CSV file with parsing errors
        When POST request with malformed CSV
        Then should return 400 Bad Request with parser error
        """
        # Arrange
        import pandas as pd
        mock_analyze_logic.side_effect = pd.errors.ParserError('Parse failed')

        csv_file = SimpleUploadedFile('bad.csv', b'malformed,csv\ndata', content_type='text/csv')
        request = self.factory.post('/analyze-csv/', {'file': csv_file})

        # Act
        response = views.analyze_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'Error al parsear el archivo CSV' in response_data['error']


@pytest.mark.django_db
@pytest.mark.unit
class TestUploadAndCleanCsvView:
    """Tests for upload_and_clean_csv() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.upload_and_clean_csv_logic')
    @patch('os.path.isdir')
    def test_successful_upload_and_cleaning(self, mock_isdir, mock_upload_logic):
        """
        Scenario: Successful CSV upload and cleaning
        Given valid CSV file and cleaning options
        When POST request with file and options
        Then should upload, clean, and return success
        """
        # Arrange
        mock_isdir.return_value = True
        mock_upload_logic.return_value = {
            'status': 'success',
            'cleaned_csv_path': '/app/experimentos/exp123/cleaned.csv',
            'report': {
                'duplicates_removed': 5,
                'missing_values_handled': 10
            }
        }

        csv_content = b"feature1,feature2,target\n1.0,A,0\n2.0,B,1"
        csv_file = SimpleUploadedFile("test.csv", csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'eliminar_duplicados': 'true',
            'filtrar_outliers': 'false',
            'relleno_valores_numericos': 'media'
        }
        request = self.factory.post('/upload-clean/', request_data)

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert 'cleaned_csv_path' in response_data

    def test_missing_experiment_dir_returns_400(self):
        """
        Scenario: Missing experiment directory
        Given CSV file without experiment_dir
        When POST request without experiment directory
        Then should return 400 Bad Request
        """
        # Arrange
        csv_content = b'col1,col2,col3\n1,2,3'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/upload-and-clean-csv/', {'file': csv_file})

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'La ruta del experimento no se proporcionó.'

    @patch('os.path.isdir')
    def test_invalid_experiment_dir_returns_400(self, mock_isdir):
        """
        Scenario: Invalid experiment directory
        Given CSV file with non-existent experiment_dir
        When POST request with invalid directory path
        Then should return 400 Bad Request
        """
        # Arrange
        mock_isdir.return_value = False
        csv_content = b'col1,col2,col3\n1,2,3'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/upload-and-clean-csv/', {
            'file': csv_file,
            'experiment_dir': '/invalid/path'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'no existe o no es un directorio válido' in response_data['status']

    @patch('os.path.isdir')
    def test_missing_imputation_value_when_required(self, mock_isdir):
        """
        Scenario: Missing imputation value when relleno_valores_numericos='valor'
        Given CSV file with 'valor' option but no valor_imputacion
        When POST request without required imputation value
        Then should return 400 Bad Request
        """
        # Arrange
        mock_isdir.return_value = True
        csv_content = b'col1,col2,col3\n1,2,3'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/upload-and-clean-csv/', {
            'file': csv_file,
            'experiment_dir': '/test/experiment',
            'relleno_valores_numericos': 'valor'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'Se requiere \'valor_imputacion\'' in response_data['status']

    def test_missing_file_returns_405(self):
        """
        Scenario: Missing file in request
        Given POST request without file
        When sent to upload_and_clean_csv
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.post('/upload-and-clean-csv/', {
            'experiment_dir': '/test/experiment'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Método no permitido'

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to upload_and_clean_csv endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/upload-and-clean-csv/')

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Método no permitido'

    @patch('os.path.isdir')
    def test_non_csv_file_type_returns_400(self, mock_isdir):
        """
        Scenario: Non-CSV file type submitted
        Given uploaded TXT file instead of CSV
        When POST request with non-CSV file
        Then should return 400 Bad Request with validation error
        """
        # Arrange
        mock_isdir.return_value = True
        txt_file = SimpleUploadedFile('test.txt', b'not a csv', content_type='text/plain')
        request = self.factory.post('/upload-and-clean-csv/', {
            'file': txt_file,
            'experiment_dir': '/test/experiment'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo debe ser un CSV.'

    @patch('os.path.isdir')
    def test_file_size_exceeds_maximum(self, mock_isdir):
        """
        Scenario: File size exceeds 10MB limit
        Given CSV file larger than 10MB
        When POST request with oversized file
        Then should return 400 Bad Request with size error
        """
        # Arrange
        mock_isdir.return_value = True
        large_content = b'a' * (11 * 1024 * 1024)  # 11MB
        large_file = SimpleUploadedFile('large.csv', large_content, content_type='text/csv')
        request = self.factory.post('/upload-and-clean-csv/', {
            'file': large_file,
            'experiment_dir': '/test/experiment'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El archivo excede el tamaño máximo permitido de 10MB.'

    @patch('api.views.upload_and_clean_csv_logic')
    @patch('os.path.isdir')
    def test_invalid_imputation_value_returns_400(self, mock_isdir, mock_upload_logic):
        """
        Scenario: Invalid (non-numeric) imputation value
        Given CSV file with 'valor' option and non-numeric valor_imputacion
        When POST request with invalid imputation value
        Then should return 400 Bad Request with validation error
        """
        # Arrange
        mock_isdir.return_value = True
        csv_content = b'col1,col2,col3\n1,2,3'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/upload-and-clean-csv/', {
            'file': csv_file,
            'experiment_dir': '/test/experiment',
            'relleno_valores_numericos': 'valor',
            'valor_imputacion': 'not_a_number'
        })

        # Act
        response = views.upload_and_clean_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'El valor de imputación debe ser un número.'


"""
Phase 2C: Data Processing Endpoint Tests
Tests for generar_reporte_eda(), encode_csv()
"""


@pytest.mark.django_db
@pytest.mark.unit
class TestGenerarReporteEdaView:
    """Tests for generar_reporte_eda() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.generate_eda_logic')
    @patch('os.path.isdir')
    def test_successful_eda_report_generation(self, mock_isdir, mock_generate_logic):
        """
        Scenario: Successful EDA report generation
        Given valid experiment with cleaned data
        When POST request to generate EDA
        Then should generate report and return path
        """
        # Arrange
        mock_isdir.return_value = True
        mock_generate_logic.return_value = {
            'success': True,
            'report_path': '/app/experimentos/exp123/eda_report.html'
        }

        request_data = {
            'dataset_type': 'eda',
            'experiment_dir': '/app/experimentos/exp123',
            'run_id': 'test-run-id-123'
        }
        request = self.factory.post(
            '/generate-eda/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['success'] == True
        assert 'report_path' in response_data
        mock_generate_logic.assert_called_once_with('eda', '/app/experimentos/exp123', 'test-run-id-123')

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to generar_reporte_eda endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/generate-eda/')

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert response_data['error'] == 'Método no permitido.'

    def test_invalid_dataset_type_returns_400(self):
        """
        Scenario: Invalid dataset_type parameter
        Given request with invalid dataset_type (not 'eda' or 'train')
        When POST request to generate EDA
        Then should return 400 Bad Request
        """
        # Arrange
        request_data = {
            'dataset_type': 'invalid_type',
            'experiment_dir': '/app/experimentos/exp123',
            'run_id': 'test-run-id'
        }
        request = self.factory.post(
            '/generate-eda/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'dataset_type' in response_data['error']

    @patch('os.path.isdir')
    def test_missing_run_id_returns_400(self, mock_isdir):
        """
        Scenario: Missing run_id parameter
        Given request without run_id
        When POST request to generate EDA
        Then should return 400 Bad Request
        """
        # Arrange
        mock_isdir.return_value = True
        request_data = {
            'dataset_type': 'eda',
            'experiment_dir': '/app/experimentos/exp123'
        }
        request = self.factory.post(
            '/generate-eda/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'run_id' in response_data['error']

    @patch('os.path.isdir')
    def test_invalid_experiment_dir_returns_400(self, mock_isdir):
        """
        Scenario: Invalid experiment directory
        Given request with non-existent experiment_dir
        When POST request to generate EDA
        Then should return 400 Bad Request
        """
        # Arrange
        mock_isdir.return_value = False
        request_data = {
            'dataset_type': 'eda',
            'experiment_dir': '/invalid/path',
            'run_id': 'test-run-id'
        }
        request = self.factory.post(
            '/generate-eda/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'no es válida o no existe' in response_data['error']

    def test_invalid_json_returns_400(self):
        """
        Scenario: Invalid JSON in request body
        Given request with malformed JSON
        When POST request to generate EDA
        Then should return 400 Bad Request
        """
        # Arrange
        request = self.factory.post(
            '/generate-eda/',
            data='invalid json {{{',
            content_type='application/json'
        )

        # Act
        response = views.generar_reporte_eda(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert response_data['error'] == 'JSON inválido.'


"""
Phase 2E: Training & Pipeline Endpoint Tests (Part 1/2)
Tests for get_pipeline_config(), run_pipeline()
"""


@pytest.mark.django_db
@pytest.mark.unit
class TestGetPipelineConfigView:
    """Tests for get_pipeline_config() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('builtins.open', new_callable=mock_open, read_data='{"steps": ["step1", "step2"], "parameters": {}}')
    @patch('os.path.exists')
    def test_successful_pipeline_config_retrieval(self, mock_exists, mock_file):
        """
        Scenario: Successful pipeline configuration retrieval
        Given existing pipeline_config.json file
        When GET request to get_pipeline_config with directory_path
        Then should return 200 with configuration content
        """
        # Arrange
        mock_exists.return_value = True
        request = self.factory.get('/get-pipeline-config/?directory_path=/test/experiment')

        # Act
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'steps' in response_data
        assert response_data['steps'] == ['step1', 'step2']
        assert 'parameters' in response_data

        # Verify the file path was checked
        mock_exists.assert_called_once_with('/test/experiment/pipeline_config.json')

    @patch('os.path.exists')
    def test_config_not_found_returns_404(self, mock_exists):
        """
        Scenario: Configuration file not found
        Given non-existent pipeline_config.json
        When GET request to get_pipeline_config
        Then should return 404 with appropriate message
        """
        # Arrange
        mock_exists.return_value = False
        request = self.factory.get('/get-pipeline-config/?directory_path=/test/experiment')

        # Act
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data['status'] == 'No hay configuraciones registradas aún.'

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_file_read_error_returns_500(self, mock_exists, mock_file):
        """
        Scenario: Error reading configuration file
        Given existing file but read error occurs
        When GET request to get_pipeline_config
        Then should return 500 with error message
        """
        # Arrange
        mock_exists.return_value = True
        mock_file.side_effect = IOError("Failed to read file")
        request = self.factory.get('/get-pipeline-config/?directory_path=/test/experiment')

        # Act
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error al leer las configuraciones' in response_data['status']

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json {{{')
    @patch('os.path.exists')
    def test_invalid_json_in_config_returns_500(self, mock_exists, mock_file):
        """
        Scenario: Invalid JSON in configuration file
        Given config file with malformed JSON
        When GET request to get_pipeline_config
        Then should return 500 with JSON parse error
        """
        # Arrange
        mock_exists.return_value = True
        request = self.factory.get('/get-pipeline-config/?directory_path=/test/experiment')

        # Act
        response = views.get_pipeline_config(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error al leer las configuraciones' in response_data['status']


@pytest.mark.django_db
@pytest.mark.unit
class TestRunPipelineView:
    """Tests for run_pipeline() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.run_pipeline_logic')
    def test_successful_pipeline_execution(self, mock_run_logic):
        """
        Scenario: Successful end-to-end pipeline execution
        Given valid pipeline configuration
        When POST request to run_pipeline
        Then should execute pipeline and return success result
        """
        # Arrange
        mock_run_logic.return_value = {
            'success': True,
            'experiment_id': 'test-experiment-id',
            'experiment_dir': '/app/experimentos/exp-123'
        }

        request_data = {'pipeline_config': {'steps': ['step1', 'step2']}}
        request = self.factory.post(
            '/run-pipeline/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.run_pipeline(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['success'] == True
        assert 'experiment_id' in response_data

        # Verify logic was called with correct data
        mock_run_logic.assert_called_once_with(request_data)

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to run_pipeline endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/run-pipeline/')

        # Act
        response = views.run_pipeline(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert response_data['error'] == 'Método no permitido.'

    def test_invalid_json_returns_500(self):
        """
        Scenario: Invalid JSON in request body
        Given POST request with malformed JSON
        When sent to run_pipeline endpoint
        Then should return 500 with JSON error
        """
        # Arrange
        request = self.factory.post(
            '/run-pipeline/',
            data='invalid json {{{',
            content_type='application/json'
        )

        # Act
        response = views.run_pipeline(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'error' in response_data

    @patch('api.views.run_pipeline_logic')
    def test_pipeline_execution_error_returns_500(self, mock_run_logic):
        """
        Scenario: Pipeline execution error
        Given valid request but pipeline logic fails
        When POST request to run_pipeline
        Then should return 500 with error details
        """
        # Arrange
        mock_run_logic.side_effect = Exception('Pipeline execution failed')

        request_data = {'pipeline_config': {'steps': []}}
        request = self.factory.post(
            '/run-pipeline/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.run_pipeline(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'Pipeline execution failed' in response_data['error']


@pytest.mark.django_db
@pytest.mark.unit
class TestGetExperimentSummaryView:
    """Tests for get_experiment_summary() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('builtins.open', new_callable=mock_open, read_data=b'PDF content')
    @patch('os.path.isfile')
    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_successful_experiment_summary_generation(self, mock_isdir, mock_exists,
                                                     mock_generate_pdf, mock_isfile, mock_file):
        """
        Scenario: Successful experiment summary PDF generation
        Given experiment with valid pipeline_config.json
        When GET request to get_experiment_summary
        Then should generate PDF and return it as attachment
        """
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_generate_pdf.return_value = None  # Generates PDF successfully
        mock_isfile.return_value = True

        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 200
        assert response['Content-Type'] == 'application/pdf'
        assert 'Content-Disposition' in response
        assert 'attachment' in response['Content-Disposition']
        assert 'experiment_summary.pdf' in response['Content-Disposition']

        # Verify PDF generation was called
        mock_generate_pdf.assert_called_once_with(
            '/test/experiment/pipeline_config.json',
            '/test/experiment/experiment_summary.pdf'
        )

    @patch('os.path.isdir')
    def test_invalid_directory_path_returns_400(self, mock_isdir):
        """
        Scenario: Invalid experiment directory path
        Given non-existent or invalid directory_path
        When GET request to get_experiment_summary
        Then should return 400 Bad Request
        """
        # Arrange
        mock_isdir.return_value = False
        request = self.factory.get('/get-experiment-summary/?directory_path=/invalid/path')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'La ruta del experimento no es válida.'

    def test_missing_directory_path_returns_400(self):
        """
        Scenario: Missing directory_path parameter
        Given GET request without directory_path
        When sent to get_experiment_summary
        Then should return 400 Bad Request
        """
        # Arrange
        request = self.factory.get('/get-experiment-summary/')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'La ruta del experimento no es válida.'

    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_missing_pipeline_config_returns_404(self, mock_isdir, mock_exists):
        """
        Scenario: Missing pipeline_config.json file
        Given valid directory but no pipeline_config.json
        When GET request to get_experiment_summary
        Then should return 404 Not Found
        """
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = False
        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data['status'] == 'No se encontró el pipeline_config.json.'

    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_invalid_config_content_returns_400(self, mock_isdir, mock_exists, mock_generate_pdf):
        """
        Scenario: Invalid content in pipeline_config.json
        Given pipeline_config.json with invalid structure
        When GET request to get_experiment_summary
        Then should return 400 with validation error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_generate_pdf.side_effect = KeyError('Required key missing')

        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'Error en el contenido de pipeline_config.json' in response_data['status']

    @patch('os.path.isfile')
    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_pdf_generation_failed_returns_500(self, mock_isdir, mock_exists,
                                               mock_generate_pdf, mock_isfile):
        """
        Scenario: PDF file not generated after calling generator
        Given successful PDF generation call but file not created
        When GET request to get_experiment_summary
        Then should return 500 Internal Server Error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_generate_pdf.return_value = None
        mock_isfile.return_value = False  # PDF file not created

        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data['status'] == 'No se generó el PDF de resumen.'

    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_unexpected_error_returns_500(self, mock_isdir, mock_exists, mock_generate_pdf):
        """
        Scenario: Unexpected error during summary generation
        Given valid request but unexpected exception occurs
        When GET request to get_experiment_summary
        Then should return 500 with error details
        """
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_generate_pdf.side_effect = Exception('Unexpected error')

        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')

        # Act
        response = views.get_experiment_summary(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert 'Error interno al generar el resumen' in response_data['status']


@pytest.mark.django_db
@pytest.mark.unit
class TestEncodeCsvView:
    """Tests for encode_csv() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.encode_csv_logic')
    @patch('mlflow.get_run')
    def test_successful_encoding(self, mock_get_run, mock_encode_logic):
        """
        Scenario: Successful CSV encoding
        Given cleaned CSV with encoding configuration
        When POST request to encode_csv
        Then should encode features and return success
        """
        # Arrange
        mock_get_run.return_value = Mock()  # Mock that run exists
        mock_encode_logic.return_value = {
            'status': 'Archivo CSV codificado correctamente.',
            'processed_train_path': 'processed/processed_train_test.csv',
            'run_id': 'nested-run-123'
        }

        csv_content = b'feature1,feature2,target\n1.0,A,0\n2.0,B,1\n3.0,C,0'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'input_features': 'feature1,feature2',
            'target_variables': 'target',
            'run_id': 'test-run-id-123',
            'encode_target_ohe': 'false',
            'encode_target_label': 'true'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert 'status' in response_data
        assert 'Archivo CSV codificado correctamente' in response_data['status']

        # Verify mock was called with correct arguments
        mock_encode_logic.assert_called_once()
        call_args = mock_encode_logic.call_args
        assert call_args.kwargs['experiment_dir'] == '/app/experimentos/exp123'
        assert call_args.kwargs['input_features'] == ['feature1', 'feature2']
        assert call_args.kwargs['target_variables'] == ['target']
        assert call_args.kwargs['apply_target_ohe'] == False
        assert call_args.kwargs['apply_target_label'] == True

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to encode_csv endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/encode/')

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Método no permitido.'

    def test_missing_csv_file_returns_400(self):
        """
        Scenario: Missing CSV file in request
        Given POST request without file
        When sent to encode_csv endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        request_data = {
            'experiment_dir': '/app/experimentos/exp123',
            'input_features': 'feature1,feature2',
            'target_variables': 'target',
            'run_id': 'test-run-id'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Archivo CSV no recibido.'

    def test_missing_input_features_returns_400(self):
        """
        Scenario: Missing input_features parameter
        Given POST request with file but no input_features
        When sent to encode_csv endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        csv_content = b'feature1,feature2,target\n1.0,A,0'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'target_variables': 'target',
            'run_id': 'test-run-id'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Variables de entrada/salida no especificadas.'

    def test_missing_target_variables_returns_400(self):
        """
        Scenario: Missing target_variables parameter
        Given POST request with file but no target_variables
        When sent to encode_csv endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        csv_content = b'feature1,feature2,target\n1.0,A,0'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'input_features': 'feature1,feature2',
            'run_id': 'test-run-id'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'Variables de entrada/salida no especificadas.'

    @patch('mlflow.get_run')
    def test_invalid_run_id_returns_400(self, mock_get_run):
        """
        Scenario: Invalid or non-existent MLflow run_id
        Given POST request with invalid run_id
        When MLflow cannot find the run
        Then should return 400 Bad Request
        """
        # Arrange
        mock_get_run.return_value = None
        csv_content = b'feature1,feature2,target\n1.0,A,0'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'input_features': 'feature1,feature2',
            'target_variables': 'target',
            'run_id': 'invalid-run-id'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert 'Run padre no encontrado' in response_data['status']
        mock_get_run.assert_called_once_with('invalid-run-id')

    @patch('api.views.encode_csv_logic')
    @patch('mlflow.get_run')
    def test_encoding_with_onehot_encoding(self, mock_get_run, mock_encode_logic):
        """
        Scenario: CSV encoding with one-hot encoding
        Given CSV with encode_target_ohe=true
        When POST request to encode_csv
        Then should apply one-hot encoding to target
        """
        # Arrange
        mock_get_run.return_value = Mock()
        mock_encode_logic.return_value = {
            'status': 'Archivo CSV codificado correctamente.',
            'processed_train_path': 'processed/processed_train_test.csv',
            'run_id': 'nested-run-123'
        }

        csv_content = b'feature1,feature2,target\n1.0,A,cat1\n2.0,B,cat2'
        csv_file = SimpleUploadedFile('test.csv', csv_content, content_type='text/csv')

        request_data = {
            'file': csv_file,
            'experiment_dir': '/app/experimentos/exp123',
            'input_features': 'feature1,feature2',
            'target_variables': 'target',
            'run_id': 'test-run-id-123',
            'encode_target_ohe': 'true',
            'encode_target_label': 'false'
        }
        request = self.factory.post('/encode/', request_data)

        # Act
        response = views.encode_csv(request)

        # Assert
        assert response.status_code == 200

        # Verify one-hot encoding flag was set correctly
        call_args = mock_encode_logic.call_args
        assert call_args.kwargs['apply_target_ohe'] == True
        assert call_args.kwargs['apply_target_label'] == False


"""
Phase 2D: Development Environment Endpoint Tests
Tests for start_jupyter()
"""


@pytest.mark.django_db
@pytest.mark.unit
class TestStartJupyterView:
    """Tests for start_jupyter() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.start_jupyter_logic')
    @patch('mlflow.get_run')
    def test_successful_jupyter_startup(self, mock_get_run, mock_start_logic):
        """
        Scenario: Successful Jupyter notebook startup
        Given valid experiment directory and run_id
        When POST request to start Jupyter
        Then should start Jupyter and return URL with token
        """
        # Arrange
        mock_get_run.return_value = Mock()  # Mock that run exists
        mock_start_logic.return_value = {
            'success': True,
            'jupyter_url': 'http://localhost:8888',
            'token': 'test-token-123'
        }

        request_data = {
            'experiment_dir': '/app/experimentos/exp123',
            'run_id': 'test-run-id-123',
            'port': 8888
        }
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['success'] == True
        assert 'jupyter_url' in response_data
        assert response_data['jupyter_url'] == 'http://localhost:8888'
        assert 'token' in response_data
        assert response_data['token'] == 'test-token-123'

        # Verify mocks were called correctly
        mock_get_run.assert_called_once_with('test-run-id-123')
        mock_start_logic.assert_called_once_with(
            experiment_dir='/app/experimentos/exp123',
            run_id='test-run-id-123',
            port=8888
        )

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to start_jupyter endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/jupyter/')

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert response_data['error'] == 'Método no permitido.'

    def test_missing_experiment_dir_returns_400(self):
        """
        Scenario: Missing experiment_dir parameter
        Given POST request without experiment_dir
        When sent to start_jupyter endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        request_data = {
            'run_id': 'test-run-id'
        }
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'experiment_dir' in response_data['error']

    def test_missing_run_id_returns_400(self):
        """
        Scenario: Missing run_id parameter
        Given POST request without run_id
        When sent to start_jupyter endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        request_data = {
            'experiment_dir': '/app/experimentos/exp123'
        }
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'run_id' in response_data['error']

    @patch('mlflow.get_run')
    def test_invalid_run_id_returns_400(self, mock_get_run):
        """
        Scenario: Invalid or non-existent MLflow run_id
        Given POST request with invalid run_id
        When MLflow cannot find the run
        Then should return 400 Bad Request
        """
        # Arrange
        from mlflow.exceptions import RestException
        mock_get_run.side_effect = RestException({
            'error_code': 'RESOURCE_DOES_NOT_EXIST',
            'message': 'Run not found'
        })

        request_data = {
            'experiment_dir': '/app/experimentos/exp123',
            'run_id': 'invalid-run-id'
        }
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'Run padre no encontrada' in response_data['error']
        mock_get_run.assert_called_once_with('invalid-run-id')

    @patch('mlflow.get_run')
    def test_mlflow_exception_returns_500(self, mock_get_run):
        """
        Scenario: MLflow service error
        Given POST request with valid parameters
        When MLflow service encounters an error
        Then should return 500 Internal Server Error
        """
        # Arrange
        from mlflow.exceptions import MlflowException
        mock_get_run.side_effect = MlflowException('MLflow service unavailable')

        request_data = {
            'experiment_dir': '/app/experimentos/exp123',
            'run_id': 'test-run-id'
        }
        request = self.factory.post(
            '/jupyter/',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert 'Error al obtener run padre' in response_data['error']

    def test_invalid_json_returns_400(self):
        """
        Scenario: Invalid JSON in request body
        Given POST request with malformed JSON
        When sent to start_jupyter endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        request = self.factory.post(
            '/jupyter/',
            data='invalid json {{{',
            content_type='application/json'
        )

        # Act
        response = views.start_jupyter(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['success'] == False
        assert response_data['error'] == 'JSON inválido.'


"""
Phase 2E: Training & Pipeline Endpoint Tests (Part 2/2)
Tests for train_model()
"""


@pytest.mark.django_db
@pytest.mark.unit
class TestTrainModelView:
    """Tests for train_model() endpoint."""

    def setup_method(self):
        self.factory = RequestFactory()

    @patch('api.views.train_model_logic')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.end_run')
    @patch('mlflow.active_run')
    @patch('mlflow.set_tracking_uri')
    @patch('os.path.isdir')
    def test_successful_model_training(self, mock_isdir, mock_set_tracking,
                                      mock_active_run, mock_end_run,
                                      mock_get_experiment, mock_train_logic):
        """
        Scenario: Successful model training with file upload
        Given valid CSV file and training configuration
        When POST request to train_model
        Then should train model and return success with metrics
        """
        # Arrange
        mock_isdir.return_value = True
        mock_active_run.return_value = None  # No active run initially
        mock_experiment = Mock()
        mock_experiment.experiment_id = 'exp-123'
        mock_get_experiment.return_value = mock_experiment
        mock_train_logic.return_value = {
            'run_id': 'test-run-id',
            'val_metrics': {'accuracy': 0.95, 'f1_score': 0.93},
            'model_path': '/test/experiment/model.pkl'
        }

        csv_content = b'feature1,feature2,target\n1,2,A\n3,4,B'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        data = {
            'experiment_dir': '/test/experiment',
            'model_type': 'classification',
            'target_column': 'target'
        }

        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert response_data['status'] == 'success'
        assert response_data['run_id'] == 'test-run-id'
        assert 'metrics' in response_data
        assert response_data['metrics']['accuracy'] == 0.95
        assert 'model_path' in response_data
        assert 'mlflow_ui' in response_data

        # Verify MLflow tracking was set up correctly
        mock_set_tracking.assert_called_once()
        mock_train_logic.assert_called_once()

    def test_invalid_http_method_returns_405(self):
        """
        Scenario: Invalid HTTP method (GET instead of POST)
        Given GET request
        When sent to train_model endpoint
        Then should return 405 Method Not Allowed
        """
        # Arrange
        request = self.factory.get('/train-model/')

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 405
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert response_data['error_code'] == 'HTTP_405_METHOD_NOT_ALLOWED'

    def test_missing_file_returns_400(self):
        """
        Scenario: Missing CSV file in request
        Given POST request without file
        When sent to train_model endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/train-model/', {
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'archivo CSV' in response_data['error_details']

    def test_missing_data_configuration_returns_400(self):
        """
        Scenario: Missing data configuration
        Given POST request with file but no data configuration
        When sent to train_model endpoint
        Then should return 400 Bad Request
        """
        # Arrange
        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/train-model/', {'file': csv_file})

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'configuración' in response_data['error_details']

    def test_invalid_json_in_data_returns_400(self):
        """
        Scenario: Invalid JSON in data configuration
        Given POST request with malformed JSON in data field
        When sent to train_model endpoint
        Then should return 400 with JSON decode error
        """
        # Arrange
        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': 'invalid json {{'
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'JSON inválido' in response_data['message']

    @patch('os.path.isdir')
    def test_invalid_experiment_directory_returns_404(self, mock_isdir):
        """
        Scenario: Invalid experiment directory
        Given non-existent experiment_dir
        When POST request to train_model
        Then should return 404 Not Found
        """
        # Arrange
        mock_isdir.return_value = False
        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        data = {'experiment_dir': '/invalid/experiment'}
        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 404
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'inválido' in response_data['error_details']

    @patch('api.views.train_model_logic')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.active_run')
    @patch('os.path.isdir')
    def test_training_runtime_error_returns_500(self, mock_isdir, mock_active_run,
                                                mock_set_tracking, mock_train_logic):
        """
        Scenario: Runtime error during training
        Given valid request but training logic raises RuntimeError
        When POST request to train_model
        Then should return 500 Internal Server Error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_active_run.return_value = None
        mock_train_logic.side_effect = RuntimeError('Training failed')

        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 500
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'Error durante el entrenamiento' in response_data['message']

    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.end_run')
    @patch('api.views.train_model_logic')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.active_run')
    @patch('os.path.isdir')
    def test_mlflow_run_cleanup_on_error(self, mock_isdir, mock_active_run,
                                        mock_set_tracking, mock_train_logic,
                                        mock_end_run, mock_get_experiment):
        """
        Scenario: MLflow run cleanup after error
        Given training that fails with active MLflow run
        When exception occurs during training
        Then should call mlflow.end_run() to cleanup
        """
        # Arrange
        mock_isdir.return_value = True
        mock_active_run.return_value = Mock()  # Active run exists
        mock_train_logic.side_effect = Exception('Test error')
        mock_get_experiment.return_value = None

        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 500
        # Verify cleanup was called
        assert mock_end_run.call_count >= 1

    @patch('mlflow.get_experiment_by_name')
    @patch('api.views.train_model_logic')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.end_run')
    @patch('mlflow.active_run')
    @patch('os.path.isdir')
    def test_experiment_not_found_returns_400(self, mock_isdir, mock_active_run,
                                             mock_end_run, mock_set_tracking,
                                             mock_train_logic, mock_get_experiment):
        """
        Scenario: MLflow experiment not found after training
        Given successful training but experiment not found in MLflow
        When retrieving experiment by name
        Then should return 400 with appropriate error
        """
        # Arrange
        mock_isdir.return_value = True
        mock_active_run.return_value = None
        mock_train_logic.return_value = {
            'run_id': 'test-run-id',
            'val_metrics': {'accuracy': 0.95},
            'model_path': '/test/model.pkl'
        }
        mock_get_experiment.return_value = None  # Experiment not found

        csv_content = b'feature1,feature2,target\n1,2,A'
        csv_file = SimpleUploadedFile('train.csv', csv_content, content_type='text/csv')

        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/train-model/', {
            'file': csv_file,
            'data': json.dumps(data)
        })

        # Act
        response = views.train_model(request)

        # Assert
        assert response.status_code == 400
        response_data = json.loads(response.content)
        assert response_data['status'] == 'error'
        assert 'Experimento no encontrado' in response_data['error_details']


@pytest.mark.django_db
class TestErrorHandlingScenarios(TestCase):
    """Test cases for comprehensive error handling across views"""
    
    def setUp(self):
        self.factory = RequestFactory()

    @patch('api.views.create_experiment_logic')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_value_error_create_experiment(self, mock_environ_get, mock_isdir, mock_create_logic):
        # Arrange
        mock_environ_get.return_value = '/app/experimentos'
        mock_isdir.return_value = True
        mock_create_logic.side_effect = ValueError('Invalid parameter')
        
        request = self.factory.post('/create-experiment/')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['type'], 'validation_error')

    @patch('api.views.create_experiment_logic')
    @patch('os.path.isdir')
    @patch('os.environ.get')
    def test_os_error_create_experiment(self, mock_environ_get, mock_isdir, mock_create_logic):
        # Arrange
        mock_environ_get.return_value = '/app/experimentos'
        mock_isdir.return_value = True
        mock_create_logic.side_effect = OSError('Disk full')
        
        request = self.factory.post('/create-experiment/')
        
        # Act
        response = views.create_experiment(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['type'], 'filesystem_error')

    @patch('api.views.init_dvc_logic')
    def test_subprocess_error_init_dvc(self, mock_init_logic):
        # Arrange
        mock_init_logic.side_effect = subprocess.CalledProcessError(
            1, 'dvc init', stderr=b'Command failed'
        )
        
        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/init-dvc/',
                                  data=json.dumps(data),
                                  content_type='application/json')
        
        # Act
        response = views.init_dvc(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertIn('Error al ejecutar comando', response_data['status'])

    @patch('api.views.init_dvc_logic')
    def test_permission_error_init_dvc(self, mock_init_logic):
        # Arrange
        mock_init_logic.side_effect = PermissionError('Access denied')
        
        data = {'experiment_dir': '/test/experiment'}
        request = self.factory.post('/init-dvc/',
                                  data=json.dumps(data),
                                  content_type='application/json')
        
        # Act
        response = views.init_dvc(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Error de permisos en el sistema de archivos')

    @patch('api.views.start_mlflow_logic')
    @patch('os.path.isdir')
    def test_runtime_error_start_mlflow(self, mock_isdir, mock_start_logic):
        # Arrange
        mock_isdir.return_value = True
        mock_start_logic.side_effect = RuntimeError('MLflow server failed to start')
        
        data = {'directory_path': '/test/directory'}
        request = self.factory.post('/start-mlflow/',
                                  data=json.dumps(data),
                                  content_type='application/json')
        
        # Act
        response = views.start_mlflow(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertIn('Error al iniciar MLflow', response_data['status'])

    def test_invalid_json_generar_reporte_eda(self):
        # Arrange
        request = self.factory.post('/generate-eda/',
                                  data='invalid json',
                                  content_type='application/json')
        
        # Act
        response = views.generar_reporte_eda(request)
        
        # Assert
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'JSON inválido.')

    def test_invalid_json_start_mlflow(self):
        # Arrange
        request = self.factory.post('/start-mlflow/',
                                  data='invalid json',
                                  content_type='application/json')
        
        # Act
        response = views.start_mlflow(request)
        
        # Assert
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['status'], 'Formato JSON inválido en la solicitud')

    @patch('api.views.run_pipeline_logic')
    def test_exception_run_pipeline(self, mock_run_logic):
        # Arrange
        mock_run_logic.side_effect = Exception('Pipeline execution failed')
        
        data = {'pipeline_config': {'steps': []}}
        request = self.factory.post('/run-pipeline/',
                                  data=json.dumps(data),
                                  content_type='application/json')
        
        # Act
        response = views.run_pipeline(request)
        
        # Assert
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])

    @patch('api.views.generate_experiment_summary_pdf')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_validation_error_experiment_summary(self, mock_isdir, mock_exists, mock_generate_pdf):
        # Arrange
        mock_isdir.return_value = True
        mock_exists.return_value = True
        mock_generate_pdf.side_effect = ValueError('Invalid config format')
        
        request = self.factory.get('/get-experiment-summary/?directory_path=/test/experiment')
        
        # Act
        response = views.get_experiment_summary(request)
        
        # Assert
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertIn('Error en el contenido de pipeline_config.json', response_data['status'])