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

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the function to test
from api.data_cleaning import limpiar_datos


class TestDataCleaning:
    
    def setup_method(self):
        """Setup method to create temporary directories for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup temporary files after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_temp_csv(self, df, filename="input.csv"):
        """Helper method to create temporary CSV files."""
        filepath = os.path.join(self.temp_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def get_output_path(self, filename="output.csv"):
        """Helper method to get output file path."""
        return os.path.join(self.temp_dir, filename)

    # Basic Functionality Tests
    
    def test_basic_data_cleaning_with_default_parameters(self):
        """Test basic data cleaning with default parameters.

        This test verifies how the code currently works:
        - The test data has NO actual duplicates (rows differ in col3)
        - Therefore duplicates_removed should be 0 (not > 0)
        """
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3, np.nan],
            'col2': ['a', 'b', 'b', 'c', ''],
            'col3': [1.1, 2.2, 100.0, 3.3, 4.4]  # Contains outlier
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()

        # Act
        report = limpiar_datos(input_path, output_path)

        # Assert - Test how the code ACTUALLY works
        assert report['initial_rows'] == 5
        assert report['duplicates_removed'] == 0  # No actual duplicates in this data
        assert 'numeric_imputations' in report
        assert 'categorical_missing_filled' in report
        assert os.path.exists(output_path)
    
    def test_load_csv_and_perform_initial_setup(self):
        """Test CSV loading and initial setup recording."""
        # Arrange
        df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert report['initial_rows'] == 3
        assert set(report['initial_columns']) == {'column1', 'column2'}
        assert report['final_rows'] <= report['initial_rows']

    # Column Cleaning Tests
    
    def test_clean_column_names_with_whitespace(self):
        """Test cleaning column names with leading/trailing whitespace."""
        # Arrange
        df = pd.DataFrame({
            '  col1  ': [1, 2, 3],
            'col2   ': ['a', 'b', 'c'],
            '   col3': [4, 5, 6]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        cleaned_df = pd.read_csv(output_path)
        expected_columns = ['col1', 'col2', 'col3']
        assert list(cleaned_df.columns) == expected_columns
    
    def test_clean_cell_values_with_whitespace(self):
        """Test cleaning string cell values with whitespace."""
        # Arrange
        df = pd.DataFrame({
            'col1': ['  value1  ', ' value2 ', 'value3   '],
            'col2': [1, 2, 3]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        limpiar_datos(input_path, output_path)
        
        # Assert
        cleaned_df = pd.read_csv(output_path)
        expected_values = ['value1', 'value2', 'value3']
        assert cleaned_df['col1'].tolist() == expected_values
    
    def test_replace_empty_strings_with_nan(self):
        """Test replacing empty strings and whitespace-only cells with NaN."""
        # Arrange
        df = pd.DataFrame({
            'col1': ['value1', '', '   ', 'value2'],
            'col2': [1, 2, 3, 4]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        limpiar_datos(input_path, output_path)
        
        # Assert
        cleaned_df = pd.read_csv(output_path)
        # Empty strings should be replaced with 'vacio' for categorical columns
        expected_values = ['value1', 'vacio', 'vacio', 'value2']
        assert cleaned_df['col1'].tolist() == expected_values

    # Numeric Column Conversion Tests
    
    def test_convert_string_columns_to_numeric(self):
        """Test converting string columns containing numeric values to numeric type."""
        # Arrange
        df = pd.DataFrame({
            'string_numeric': ['1', '2', '3'],
            'mixed_column': ['a', 'b', 'c'],
            'decimal_strings': ['1.5', '2.5', '3.5']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        cleaned_df = pd.read_csv(output_path)
        assert 'string_numeric' in report['converted_to_numeric']
        assert 'decimal_strings' in report['converted_to_numeric']
        assert 'mixed_column' not in report['converted_to_numeric']
        assert pd.api.types.is_numeric_dtype(cleaned_df['string_numeric'])
    
    def test_handle_columns_that_cannot_be_converted(self):
        """Test handling columns with mixed alphanumeric values that cannot be converted."""
        # Arrange
        df = pd.DataFrame({
            'mixed_alpha': ['1a', '2b', '3c'],
            'pure_alpha': ['abc', 'def', 'ghi'],
            'numeric': [1, 2, 3]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        cleaned_df = pd.read_csv(output_path)
        assert 'mixed_alpha' not in report['converted_to_numeric']
        assert 'pure_alpha' not in report['converted_to_numeric']
        assert not pd.api.types.is_numeric_dtype(cleaned_df['mixed_alpha'])
        assert not pd.api.types.is_numeric_dtype(cleaned_df['pure_alpha'])

    # Duplicate Removal Tests
    
    def test_remove_duplicates_when_enabled(self):
        """Test removing duplicates when eliminar_duplicados=True."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, eliminar_duplicados=True)
        
        # Assert
        assert report['duplicates_removed'] == 1
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) == 3
    
    def test_keep_duplicates_when_disabled(self):
        """Test keeping duplicates when eliminar_duplicados=False."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, eliminar_duplicados=False)
        
        # Assert
        assert report['duplicates_removed'] == 0
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) == 4

    # Missing Values Handling Tests
    
    def test_fill_numeric_missing_values_with_mean(self):
        """Test filling numeric missing values with column means."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1.5, np.nan, 3.5, 4.5]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, relleno_valores_numericos="media")
        
        # Assert
        assert 'numeric_imputations' in report
        assert 'col1' in report['numeric_imputations']
        assert 'col2' in report['numeric_imputations']
        assert report['numeric_imputations']['col1']['filled_with'] == 'mean'
        
        cleaned_df = pd.read_csv(output_path)
        assert not cleaned_df['col1'].isna().any()
        assert not cleaned_df['col2'].isna().any()
    
    def test_drop_rows_with_numeric_missing_values(self):
        """Test dropping rows with numeric missing values."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1.5, 2.5, 3.5, np.nan]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, relleno_valores_numericos="eliminar")
        
        # Assert
        assert 'numeric_rows_dropped_due_to_na' in report
        assert report['numeric_rows_dropped_due_to_na'] == 2
        
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) == 2
        assert not cleaned_df.isna().any().any()
    
    def test_leave_numeric_missing_values_unchanged(self):
        """Test leaving numeric missing values as NaN."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1.5, 2.5, 3.5, 4.5]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, relleno_valores_numericos="dejar")
        
        # Assert
        assert 'numeric_missing_after' in report
        assert report['numeric_missing_after']['col1'] == 1
        
        cleaned_df = pd.read_csv(output_path)
        assert cleaned_df['col1'].isna().sum() == 1
    
    def test_fill_numeric_missing_values_with_specific_value(self):
        """Test filling numeric missing values with a specific value."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1.5, np.nan, 3.5, 4.5]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, 
                             relleno_valores_numericos="valor", valor_imputacion=0)
        
        # Assert
        assert 'numeric_imputations' in report
        assert report['numeric_imputations']['col1']['filled_with'] == 0
        assert report['numeric_imputations']['col2']['filled_with'] == 0
        
        cleaned_df = pd.read_csv(output_path)
        assert cleaned_df.loc[2, 'col1'] == 0
        assert cleaned_df.loc[1, 'col2'] == 0
    
    def test_fill_categorical_missing_values(self):
        """Test filling categorical missing values with 'vacio'."""
        # Arrange
        df = pd.DataFrame({
            'cat_col': ['a', 'b', np.nan, 'c'],
            'num_col': [1, 2, 3, 4]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert 'categorical_missing_filled' in report
        assert 'cat_col' in report['categorical_missing_filled']
        assert report['categorical_missing_filled']['cat_col'] == 1
        
        cleaned_df = pd.read_csv(output_path)
        assert cleaned_df.loc[2, 'cat_col'] == 'vacio'

    # Outlier Filtering Tests
    
    def test_filter_outliers_when_enabled(self):
        """Test filtering outliers using IQR method when enabled."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'col2': [10, 20, 30, 40, 50]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, filtrar_outliers=True)
        
        # Assert
        assert 'outliers_removed' in report
        assert 'col1' in report['outliers_removed']
        assert report['outliers_removed']['col1'] >= 1
        
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) < 5
    
    def test_keep_outliers_when_disabled(self):
        """Test keeping outliers when filtrar_outliers=False."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'col2': [10, 20, 30, 40, 50]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path, filtrar_outliers=False)
        
        # Assert
        assert 'outliers_removed' in report
        assert all(count == 0 for count in report['outliers_removed'].values())
        
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) == 5

    # Column Removal Tests
    
    def test_remove_columns_with_all_null_values(self):
        """Test removing columns that contain only null values."""
        # Arrange
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'all_null_col': [np.nan, np.nan, np.nan],
            'another_good_col': ['a', 'b', 'c']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert 'columns_removed_all_na' in report
        assert 'all_null_col' in report['columns_removed_all_na']
        
        cleaned_df = pd.read_csv(output_path)
        assert 'all_null_col' not in cleaned_df.columns
        assert 'good_col' in cleaned_df.columns

    # Edge Cases
    
    def test_handle_dataset_with_no_numeric_columns(self):
        """Test handling dataset with only categorical columns."""
        # Arrange
        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'],
            'cat2': ['x', 'y', 'z']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert report['converted_to_numeric'] == []
        assert report['numeric_missing_before'] == {}
        assert os.path.exists(output_path)
    
    def test_handle_dataset_with_no_categorical_columns(self):
        """Test handling dataset with only numeric columns."""
        # Arrange
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [1.1, 2.2, 3.3]
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert report['categorical_missing_filled'] == {}
        assert os.path.exists(output_path)
    
    def test_handle_columns_with_no_missing_values(self):
        """Test handling dataset with complete data (no missing values)."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert report['numeric_imputations'] == {}
        assert report['categorical_missing_filled'] == {}
        assert os.path.exists(output_path)

    # File I/O Tests
    
    def test_save_cleaned_dataset_successfully(self):
        """Test successful saving of cleaned dataset."""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        input_path = self.create_temp_csv(df)
        output_path = self.get_output_path()
        
        # Act
        report = limpiar_datos(input_path, output_path)
        
        # Assert
        assert os.path.exists(output_path)
        cleaned_df = pd.read_csv(output_path)
        assert len(cleaned_df) > 0
        assert report['final_rows'] == len(cleaned_df)
        assert set(report['final_columns']) == set(cleaned_df.columns)
    
    def test_handle_invalid_input_file_path(self):
        """Test handling invalid or non-existent input file path."""
        # Arrange
        invalid_input_path = "/nonexistent/path/file.csv"
        output_path = self.get_output_path()
        
        # Act & Assert
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            limpiar_datos(invalid_input_path, output_path)