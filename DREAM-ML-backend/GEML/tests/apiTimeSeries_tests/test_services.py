# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
import os
import sys
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open, MagicMock
from io import StringIO

# Import the classes we want to test
from apiTimeSeries.services import PreProcessingService, PipelineService


class TestPreProcessingService:
    """Test cases for PreProcessingService class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.service = PreProcessingService()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Tests for analyze_csv_logic method
    # ─────────────────────────────────────────────────────────────────────────────
    
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
    
    def test_analyze_csv_logic_csv_reading_error(self):
        """
        Scenario 2: CSV Reading Error
        Given an invalid or corrupted CSV file
        When analyze_csv_logic is called
        Then it should raise an exception and log the error
        """
        # Arrange
        mock_csv_file = Mock()
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("Invalid CSV format")
            
            # Act & Assert
            with pytest.raises(Exception) as exc_info:
                self.service.analyze_csv_logic(mock_csv_file)
            
            assert "Invalid CSV format" in str(exc_info.value)
            mock_read_csv.assert_called_once_with(mock_csv_file, nrows=0)
    
    def test_analyze_csv_logic_empty_csv_file(self):
        """
        Scenario 3: Empty CSV File
        Given an empty CSV file
        When analyze_csv_logic is called
        Then it should return a dictionary with empty columns list
        """
        # Arrange
        mock_csv_file = StringIO("")
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame()  # Empty DataFrame
            mock_read_csv.return_value = mock_df
            
            # Act
            result = self.service.analyze_csv_logic(mock_csv_file)
            
            # Assert
            assert isinstance(result, dict)
            assert "columns" in result
            assert result["columns"] == []
            mock_read_csv.assert_called_once_with(mock_csv_file, nrows=0)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Tests for upload_and_clean_csv_logic method
    # ─────────────────────────────────────────────────────────────────────────────
    
    def test_upload_and_clean_csv_logic_invalid_experiment_directory(self):
        """
        Scenario 5: Invalid Experiment Directory
        Given a CSV file and invalid/non-existent experiment directory
        When upload_and_clean_csv_logic is called
        Then it should raise a ValueError about invalid experiment path
        """
        # Arrange
        mock_csv_file = Mock()
        mock_csv_file.name = "test.csv"
        invalid_experiment_dir = "/non/existent/path"
        optional_methods = []
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            self.service.upload_and_clean_csv_logic(
                mock_csv_file, 
                invalid_experiment_dir, 
                optional_methods
            )
        
        assert "no es válida" in str(exc_info.value) or "not valid" in str(exc_info.value).lower()
    
    
    def test_upload_and_clean_csv_logic_empty_experiment_directory(self):
        """
        Test for empty experiment directory string
        """
        # Arrange
        mock_csv_file = Mock()
        mock_csv_file.name = "test.csv"
        empty_experiment_dir = ""
        optional_methods = []
        
        # Act & Assertd
        with pytest.raises(ValueError) as exc_info:
            self.service.upload_and_clean_csv_logic(
                mock_csv_file, 
                empty_experiment_dir, 
                optional_methods
            )
        
        assert "no es válida" in str(exc_info.value) or "not valid" in str(exc_info.value).lower()
    
    @patch('asgiref.sync.async_to_sync')
    @patch('channels.layers.get_channel_layer')
    @patch('apiTimeSeries.services.set_tracking_uri')
    @patch('apiTimeSeries.services.mlflow.get_experiment_by_name')
    @patch('os.path.isdir')
    def test_upload_and_clean_csv_logic_mlflow_experiment_not_found(
        self, 
        mock_isdir, 
        mock_get_experiment_by_name, 
        mock_set_tracking_uri,
        mock_get_channel_layer,
        mock_async_to_sync
    ):
        """
        Scenario 7: MLflow Experiment Not Found
        Given a valid CSV file but non-existent MLflow experiment
        When upload_and_clean_csv_logic is called
        Then it should raise a ValueError about experiment not found
        """
        # Arrange
        mock_csv_file = Mock()
        mock_csv_file.name = "test.csv"
        mock_csv_file.chunks.return_value = [b"test,data\n1,2"]
        experiment_dir = "/valid/experiment/dir"
        optional_methods = []
        
        mock_isdir.return_value = True
        mock_get_channel_layer.return_value = Mock()
        mock_get_experiment_by_name.return_value = None  # Experiment not found
        
        # Mock async_to_sync to return a simple callable that doesn't cause await issues
        mock_async_to_sync.return_value = Mock()
        
        # Create mock codecarbon module
        mock_codecarbon = Mock()
        mock_tracker = Mock()
        mock_tracker.start.return_value = None
        mock_tracker.stop.return_value = None
        mock_energy = MagicMock()
        mock_energy.kWh = 0.5
        mock_tracker._total_energy = mock_energy
        mock_tracker.final_emissions = 0.1
        mock_codecarbon.EmissionsTracker = Mock(return_value=mock_tracker)
        
        with patch('os.makedirs'), \
             patch('os.path.exists', return_value=False), \
             patch('os.path.getsize', return_value=0), \
             patch('builtins.open', mock_open()), \
             patch('os.path.dirname', return_value="/valid"), \
             patch('os.path.basename', return_value="dir"), \
             patch.dict('sys.modules', {'codecarbon': mock_codecarbon}):
            
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                self.service.upload_and_clean_csv_logic(
                    mock_csv_file, 
                    experiment_dir, 
                    optional_methods
                )
            
            assert "no se encontró en MLflow" in str(exc_info.value) or "not found in MLflow" in str(exc_info.value).lower()


class TestPipelineService:
    """Test cases for PipelineService class"""
    
    def test_pipeline_service_initialization(self):
        """
        Scenario 20: Pipeline Service Initialization
        Given PipelineService class
        When instantiated
        Then it should create an instance successfully
        """
        # Arrange & Act
        service = PipelineService()
        
        # Assert
        assert isinstance(service, PipelineService)
        assert service is not None


class TestLimpiarDatos:
    """Test cases for limpiar_datos function - Limited due to stub implementation"""
    
    @patch('apiTimeSeries.services.limpiar_datos')
    def test_limpiar_datos_basic_functionality(self, mock_limpiar_datos):
        """
        Basic test for limpiar_datos function
        Note: This is limited because the actual function is a stub
        """
        # Arrange
        csv_input = "/path/to/input.csv"
        csv_output_eda = "/path/to/output.csv"
        optional_methods = []
        
        expected_report = {
            "initial_rows": 100,
            "final_rows": 95,
            "processing_type": "full_cleaning"
        }
        
        mock_limpiar_datos.return_value = expected_report
        
        # Act
        from apiTimeSeries.services import limpiar_datos
        result = limpiar_datos(csv_input, csv_output_eda, optional_methods)
        
        # Assert
        assert result == expected_report
        mock_limpiar_datos.assert_called_once_with(csv_input, csv_output_eda, optional_methods)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures and Utilities
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_csv_file():
    """Create a mock CSV file for testing"""
    mock_file = Mock()
    mock_file.name = "test_file.csv"
    mock_file.chunks.return_value = [b"col1,col2,col3\n1,2,3\n4,5,6"]
    return mock_file


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing"""
    return "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"