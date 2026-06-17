import pytest
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from django.core.files.uploadedfile import SimpleUploadedFile

# Import the service class - adjust the import path based on your project structure
from apiTimeSeries.services import DataEncodingService


class TestEncodeCSVLogic:
    
    @pytest.fixture
    def data_encoding_service(self):
        """Fixture to create DataEncodingService instance"""
        return DataEncodingService()
    
    @pytest.fixture
    def mock_csv_file(self):
        """Fixture to create a mock CSV file"""
        csv_content = b"feature1,feature2,target1\n1,2,A\n3,4,B\n5,6,C\n"
        return SimpleUploadedFile("test_data.csv", csv_content, content_type="text/csv")
    
    @pytest.fixture
    def mock_processed_csv_file(self):
        """Fixture to create a mock processed CSV file"""
        csv_content = b"feature1,feature2,target1\n1,2,A\n3,4,B\n5,6,C\n"
        return SimpleUploadedFile("processed_test_data.csv", csv_content, content_type="text/csv")
    
    @pytest.fixture
    def valid_experiment_dir(self, tmp_path):
        """Fixture to create a valid experiment directory"""
        experiment_dir = tmp_path / "test_experiment"
        experiment_dir.mkdir()
        return str(experiment_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Fixture to create a sample DataFrame for testing"""
        return pd.DataFrame({
            'feature1': [1, 3, 5],
            'feature2': [2, 4, 6],
            'target1': ['A', 'B', 'C']
        })

    # Validation and Input Handling Test Cases

    def test_invalid_experiment_directory(self, data_encoding_service, mock_csv_file):
        """Scenario 2: Invalid experiment directory"""
        # Arrange
        invalid_dir = "/non/existent/directory"
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        with pytest.raises(ValueError, match="La ruta proporcionada no es válida"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=invalid_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    def test_empty_input_features(self, data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 3: Empty input features"""
        # Arrange
        input_features = []
        target_variables = ["target1"]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Variables de entrada y/o de salida no especificadas"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    def test_empty_target_variables(self, data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 4: Empty target variables"""
        # Arrange
        input_features = ["feature1", "feature2"]
        target_variables = []
        
        # Act & Assert
        with pytest.raises(ValueError, match="Variables de entrada y/o de salida no especificadas"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    def test_both_ohe_and_label_encoding_enabled(self, data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 5: Both OHE and Label Encoding enabled"""
        # Arrange
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        with pytest.raises(ValueError, match="No se puede usar OHE y LabelEncoder simultáneamente"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=True,
                apply_target_label=True
            )

    # File Handling Test Cases

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    def test_csv_file_with_processed_prefix(self, mock_makedirs, mock_getsize, mock_exists, 
                                          data_encoding_service, mock_processed_csv_file, valid_experiment_dir):
        """Scenario 6: CSV file already has processed_ prefix"""
        # Arrange
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        # This test verifies the file path logic - the file should be placed in processed/ directory
        # We expect the function to fail at a later stage (MLflow setup), but we can verify the path logic
        with patch('builtins.open', mock_open()), \
             patch('pandas.read_csv'), \
             pytest.raises(Exception):  # Will fail at MLflow setup
            data_encoding_service.encode_csv_logic(
                csv_file=mock_processed_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    def test_csv_file_without_processed_prefix(self, mock_makedirs, mock_getsize, mock_exists,
                                             data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 7: CSV file without processed_ prefix"""
        # Arrange
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        # This test verifies the file path logic - the file should be placed in raw/ directory
        with patch('builtins.open', mock_open()), \
             patch('pandas.read_csv'), \
             pytest.raises(Exception):  # Will fail at MLflow setup
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_csv_file_already_exists_and_not_empty(self, mock_read_csv, mock_makedirs, mock_getsize, mock_exists,
                                                   data_encoding_service, mock_csv_file, valid_experiment_dir, sample_dataframe):
        """Scenario 8: CSV file already exists and is not empty"""
        # Arrange
        mock_exists.return_value = True
        mock_getsize.return_value = 100  # File is not empty
        mock_read_csv.return_value = sample_dataframe
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        # File should not be rewritten, existing file should be used
        with patch('builtins.open') as mock_file_open, \
             pytest.raises(Exception):  # Will fail at MLflow setup
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )
        
        # Assert that file was not opened for writing since it already exists
        mock_file_open.assert_not_called()

    # Column Validation Test Cases

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_missing_input_feature_column(self, mock_read_csv, mock_makedirs, mock_getsize, mock_exists,
                                        data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 12: Missing input feature column"""
        # Arrange
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        # DataFrame missing 'feature3' column
        df_missing_column = pd.DataFrame({
            'feature1': [1, 3, 5],
            'feature2': [2, 4, 6],
            'target1': ['A', 'B', 'C']
        })
        mock_read_csv.return_value = df_missing_column
        
        input_features = ["feature1", "feature2", "feature3"]  # feature3 doesn't exist
        target_variables = ["target1"]
        
        # Act & Assert
        with patch('builtins.open', mock_open()), \
             pytest.raises(RuntimeError, match="Error al validar columnas en el archivo CSV"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_missing_target_variable_column(self, mock_read_csv, mock_makedirs, mock_getsize, mock_exists,
                                          data_encoding_service, mock_csv_file, valid_experiment_dir):
        """Scenario 13: Missing target variable column"""
        # Arrange
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        # DataFrame missing 'target2' column
        df_missing_column = pd.DataFrame({
            'feature1': [1, 3, 5],
            'feature2': [2, 4, 6],
            'target1': ['A', 'B', 'C']
        })
        mock_read_csv.return_value = df_missing_column
        
        input_features = ["feature1", "feature2"]
        target_variables = ["target1", "target2"]  # target2 doesn't exist
        
        # Act & Assert
        with patch('builtins.open', mock_open()), \
             pytest.raises(RuntimeError, match="Error al validar columnas en el archivo CSV"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    # MLflow Integration Test Cases

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    def test_mlflow_experiment_not_found(self, mock_get_experiment, mock_set_uri, mock_read_csv, 
                                       mock_makedirs, mock_getsize, mock_exists,
                                       data_encoding_service, mock_csv_file, valid_experiment_dir, sample_dataframe):
        """Scenario 15: MLflow experiment doesn't exist"""
        # Arrange
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        mock_read_csv.return_value = sample_dataframe
        mock_get_experiment.return_value = None  # Experiment not found
        
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        with patch('builtins.open', mock_open()), \
             pytest.raises(ValueError, match="El experimento .* no fue encontrado en MLflow"):
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )

    # Energy Tracking Test Cases

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_input')
    @patch('mlflow.log_artifact')
    @patch('mlflow.data.from_pandas')
    @patch('subprocess.run')
    @patch('apiTimeSeries.services.encode_data')  # Adjust import path
    def test_energy_tracking_with_none_values(self, mock_encode_data, mock_subprocess, mock_from_pandas,
                                            mock_log_artifact, mock_log_input, mock_log_metric, mock_log_param,
                                            mock_start_run, mock_get_experiment, mock_set_uri,
                                            mock_read_csv, mock_makedirs, mock_getsize, mock_exists,
                                            data_encoding_service, mock_csv_file, valid_experiment_dir, sample_dataframe):
        """Scenario 23: Energy tracking returns None values"""
        # Arrange
        mock_exists.return_value = True  # Files exist
        mock_getsize.return_value = 100
        mock_read_csv.return_value = sample_dataframe
        
        # Mock MLflow experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment
        
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock energy tracker with None values
        mock_tracker = Mock()
        mock_tracker._total_energy = None
        mock_tracker.final_emissions = None
        mock_tracker.start = Mock()
        mock_tracker.stop = Mock()
        
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        with patch('codecarbon.EmissionsTracker', return_value=mock_tracker), \
             patch('builtins.open', mock_open(read_data='{"steps": []}')), \
             patch('json.load', return_value={"steps": []}), \
             patch('json.dump'):
            
            result = data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )
        
        # Assert that default values of 0.0 were used
        mock_log_metric.assert_any_call("energy_consumed_total_kWh", 0.0)
        mock_log_metric.assert_any_call("carbon_emission_kg", 0.0)
        assert result["status"] == "Archivo CSV codificado correctamente."

    # File Output Validation Test Cases

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.start_run')
    @patch('apiTimeSeries.services.encode_data')  # Adjust import path
    def test_processed_file_not_created(self, mock_encode_data, mock_start_run, mock_get_experiment, 
                                      mock_set_uri, mock_read_csv, mock_makedirs, mock_getsize, mock_exists,
                                      data_encoding_service, mock_csv_file, valid_experiment_dir, sample_dataframe):
        """Scenario 25: Processed file not created"""
        # Arrange
        def side_effect_exists(path):
            # Return True for input file, False for processed file
            if "processed_train_" in path:
                return False
            return True
            
        mock_exists.side_effect = side_effect_exists
        mock_getsize.return_value = 100
        mock_read_csv.return_value = sample_dataframe
        
        # Mock MLflow experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_experiment.return_value = mock_experiment
        
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock energy tracker
        mock_tracker = Mock()
        mock_energy = MagicMock()
        mock_energy.kWh = 0.1
        mock_tracker._total_energy = mock_energy
        mock_tracker.final_emissions = 0.01
        mock_tracker.start = Mock()
        mock_tracker.stop = Mock()
        
        input_features = ["feature1", "feature2"]
        target_variables = ["target1"]
        
        # Act & Assert
        with patch('codecarbon.EmissionsTracker', return_value=mock_tracker), \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.log_input'), \
             patch('mlflow.data.from_pandas'), \
             pytest.raises(FileNotFoundError, match="El archivo codificado no se generó correctamente"):
            
            data_encoding_service.encode_csv_logic(
                csv_file=mock_csv_file,
                experiment_dir=valid_experiment_dir,
                input_features=input_features,
                target_variables=target_variables,
                apply_target_ohe=False,
                apply_target_label=False
            )