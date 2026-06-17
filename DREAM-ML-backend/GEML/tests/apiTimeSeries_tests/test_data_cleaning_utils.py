import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the functions to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'apiTimeSeries'))
from apiTimeSeries.data_cleaning_utils import (
    limpiar_datos,
    clean_column_names,
    remove_whitespace_from_df_data,
    replace_empty_values_with_nan,
    convert_to_numeric_columns,
    fill_categorical_missing,
    remove_empty_columns
)


class TestDataCleaningUtils:
    
    # =====================================================================
    # MAIN FUNCTION TESTS
    # =====================================================================
    
    def test_basic_successful_processing(self):
        """Scenario 1: Basic successful processing"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            # Create test CSV with mixed data and whitespace
            test_data = pd.DataFrame({
                ' col1 ': ['  value1  ', '  value2  ', ''],
                'col2': ['123', '456', '789'],
                ' col3 ': [1.5, 2.0, np.nan]
            })
            test_data.to_csv(input_file, index=False)
            
            # Act
            with patch('builtins.print') as mock_print:
                report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            
            # Check that report contains expected keys
            expected_keys = [
                "initial_rows", "initial_columns", "final_rows", "final_columns",
                "processing_type", "duplicates_removed", "numeric_missing_before",
                "numeric_imputations", "categorical_missing_filled", "outliers_removed",
                "columns_removed_all_na", "converted_to_numeric"
            ]
            for key in expected_keys:
                assert key in report
            
            assert report["initial_rows"] == 3
            assert report["final_rows"] == 3
            mock_print.assert_any_call("NO optional methods were passed...")
    
    def test_processing_with_optional_methods(self):
        """Scenario 2: Processing with optional methods"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            test_data = pd.DataFrame({'col1': [1, 2, 3]})
            test_data.to_csv(input_file, index=False)
            
            optional_methods = ['method1', 'method2']
            
            # Act
            with patch('builtins.print') as mock_print:
                report = limpiar_datos(input_file, output_file, optional_methods)
            
            # Assert
            mock_print.assert_any_call("optional methods were passed...")
            mock_print.assert_any_call(optional_methods)
    
    def test_processing_without_optional_methods(self):
        """Scenario 3: Processing without optional methods"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            test_data = pd.DataFrame({'col1': [1, 2, 3]})
            test_data.to_csv(input_file, index=False)
            
            # Act
            with patch('builtins.print') as mock_print:
                report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            mock_print.assert_any_call("NO optional methods were passed...")
    
    # =====================================================================
    # INDIVIDUAL FUNCTION TESTS
    # =====================================================================
    
    def test_clean_column_names_with_whitespace(self):
        """Scenario 4: Clean column names with whitespace"""
        # Arrange
        df = pd.DataFrame({' col1 ': [1, 2], '  col2  ': [3, 4], 'col3': [5, 6]})
        report = {}
        
        # Act
        result_df, result_report = clean_column_names(df, report)
        
        # Assert
        expected_columns = ['col1', 'col2', 'col3']
        assert list(result_df.columns) == expected_columns
        assert result_report == report  # Report should be unchanged
    
    def test_remove_whitespace_from_string_data(self):
        """Scenario 5: Remove whitespace from string data"""
        # Arrange
        df = pd.DataFrame({
            'str_col': ['  value1  ', '  value2  ', ' value3 '],
            'num_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5]
        })
        report = {}
        
        # Act
        result_df, result_report = remove_whitespace_from_df_data(df, report)
        
        # Assert
        assert list(result_df['str_col']) == ['value1', 'value2', 'value3']
        assert list(result_df['num_col']) == [1, 2, 3]  # Numbers unchanged
        assert list(result_df['float_col']) == [1.5, 2.5, 3.5]  # Floats unchanged
        assert result_report == report
    
    def test_replace_empty_values_with_nan(self):
        """Scenario 6: Replace empty strings with NaN"""
        # Arrange
        df = pd.DataFrame({
            'col1': ['value1', '', '   ', 'value2'],
            'col2': ['a', 'b', 'c', 'd']
        })
        report = {}
        
        # Act
        result_df, result_report = replace_empty_values_with_nan(df, report)
        
        # Assert
        assert result_df.loc[0, 'col1'] == 'value1'
        assert pd.isna(result_df.loc[1, 'col1'])  # Empty string -> NaN
        assert pd.isna(result_df.loc[2, 'col1'])  # Whitespace -> NaN
        assert result_df.loc[3, 'col1'] == 'value2'
        assert list(result_df['col2']) == ['a', 'b', 'c', 'd']  # Unchanged
        assert result_report == report
    
    def test_convert_numeric_strings_to_numbers(self):
        """Scenario 7: Convert numeric-looking strings to numbers"""
        # Arrange
        df = pd.DataFrame({
            'numeric_str': ['123', '456', '789'],
            'mixed_col': ['123', 'abc', '456'],
            'regular_col': ['a', 'b', 'c']
        })
        report = {}
        
        # Act
        result_df, result_report = convert_to_numeric_columns(df, report)
        
        # Assert
        assert pd.api.types.is_numeric_dtype(result_df['numeric_str'])
        assert not pd.api.types.is_numeric_dtype(result_df['mixed_col'])
        assert not pd.api.types.is_numeric_dtype(result_df['regular_col'])
        assert 'converted_to_numeric' in result_report
        assert 'numeric_str' in result_report['converted_to_numeric']
    
    def test_convert_decimal_numbers(self):
        """Scenario 8: Convert numeric-looking strings with decimals"""
        # Arrange
        df = pd.DataFrame({
            'decimal_str': ['12.3', '45.6', '78.9']
        })
        report = {}
        
        # Act
        result_df, result_report = convert_to_numeric_columns(df, report)
        
        # Assert
        assert pd.api.types.is_numeric_dtype(result_df['decimal_str'])
        assert 'decimal_str' in result_report['converted_to_numeric']
    
    def test_skip_conversion_for_mixed_data(self):
        """Scenario 9: Skip conversion for mixed data columns"""
        # Arrange
        df = pd.DataFrame({
            'mixed_col': ['123', 'abc', '456', 'def'],
            'another_mixed': ['12.3', 'xyz', '45.6', 'abc']
        })
        report = {}
        
        # Act
        result_df, result_report = convert_to_numeric_columns(df, report)
        
        # Assert
        assert not pd.api.types.is_numeric_dtype(result_df['mixed_col'])
        assert not pd.api.types.is_numeric_dtype(result_df['another_mixed'])
        assert result_report['converted_to_numeric'] == []
    
    def test_fill_categorical_missing_values(self):
        """Scenario 10: Fill categorical missing values"""
        # Arrange
        df = pd.DataFrame({
            'cat_col1': ['a', 'b', np.nan, 'd'],
            'cat_col2': ['x', np.nan, np.nan, 'z'],
            'num_col': [1, 2, 3, 4]
        })
        report = {}
        
        # Act
        result_df, result_report = fill_categorical_missing(df, report)
        
        # Assert
        assert result_df.loc[2, 'cat_col1'] == 'vacio'
        assert result_df.loc[1, 'cat_col2'] == 'vacio'
        assert result_df.loc[2, 'cat_col2'] == 'vacio'
        assert 'categorical_missing_filled' in result_report
        assert result_report['categorical_missing_filled']['cat_col1'] == 1
        assert result_report['categorical_missing_filled']['cat_col2'] == 2
    
    def test_fill_categorical_no_missing_data(self):
        """Scenario 11: Fill categorical missing values with no missing data"""
        # Arrange
        df = pd.DataFrame({
            'cat_col1': ['a', 'b', 'c', 'd'],
            'cat_col2': ['x', 'y', 'z', 'w']
        })
        report = {}
        
        # Act
        result_df, result_report = fill_categorical_missing(df, report)
        
        # Assert
        assert list(result_df['cat_col1']) == ['a', 'b', 'c', 'd']
        assert list(result_df['cat_col2']) == ['x', 'y', 'z', 'w']
        assert result_report['categorical_missing_filled'] == {}
    
    def test_remove_completely_empty_columns(self):
        """Scenario 12: Remove completely empty columns"""
        # Arrange
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'empty_col1': [np.nan, np.nan, np.nan],
            'another_good': ['a', 'b', 'c'],
            'empty_col2': [np.nan, np.nan, np.nan]
        })
        report = {}
        
        # Act
        result_df, result_report = remove_empty_columns(df, report)
        
        # Assert
        assert list(result_df.columns) == ['good_col', 'another_good']
        assert 'columns_removed_all_na' in result_report
        assert set(result_report['columns_removed_all_na']) == {'empty_col1', 'empty_col2'}
    
    def test_keep_columns_with_non_nan_values(self):
        """Scenario 13: Keep columns with at least one non-NaN value"""
        # Arrange
        df = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': [np.nan, 'b', np.nan],
            'col3': ['x', 'y', 'z']
        })
        report = {}
        
        # Act
        result_df, result_report = remove_empty_columns(df, report)
        
        # Assert
        assert list(result_df.columns) == ['col1', 'col2', 'col3']
        assert result_report['columns_removed_all_na'] == []
    
    # =====================================================================
    # EDGE CASES
    # =====================================================================
    
    def test_empty_dataframe_processing(self):
        """Scenario 14: Empty DataFrame processing"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "empty.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            # Create empty CSV with headers only
            empty_df = pd.DataFrame(columns=['col1', 'col2'])
            empty_df.to_csv(input_file, index=False)
            
            # Act
            report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            assert os.path.exists(output_file)
            assert report["initial_rows"] == 0
            assert report["final_rows"] == 0
            assert len(report["initial_columns"]) == 2
    
    def test_single_column_dataframe(self):
        """Scenario 15: Single column DataFrame"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "single.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            single_col_df = pd.DataFrame({'only_col': [1, 2, 3]})
            single_col_df.to_csv(input_file, index=False)
            
            # Act
            report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            assert os.path.exists(output_file)
            assert report["initial_rows"] == 3
            assert len(report["initial_columns"]) == 1
    
    def test_only_numeric_columns(self):
        """Scenario 16: DataFrame with only numeric columns"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "numeric.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            numeric_df = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [1.5, 2.5, 3.5],
                'col3': [10, 20, 30]
            })
            numeric_df.to_csv(input_file, index=False)
            
            # Act
            report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            assert os.path.exists(output_file)
            assert report["categorical_missing_filled"] == {}
    
    def test_only_categorical_columns(self):
        """Scenario 17: DataFrame with only categorical columns"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "categorical.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            cat_df = pd.DataFrame({
                'col1': ['a', 'b', 'c'],
                'col2': ['x', 'y', 'z'],
                'col3': ['p', 'q', 'r']
            })
            cat_df.to_csv(input_file, index=False)
            
            # Act
            report = limpiar_datos(input_file, output_file, [])
            
            # Assert
            assert os.path.exists(output_file)
            assert report["converted_to_numeric"] == []
    
    # =====================================================================
    # ERROR HANDLING TESTS
    # =====================================================================
    
    def test_invalid_input_file_path(self):
        """Scenario 18: Invalid input file path"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "nonexistent.csv")
            output_file = os.path.join(temp_dir, "output.csv")
            
            # Act & Assert
            with pytest.raises(FileNotFoundError):
                limpiar_datos(input_file, output_file, [])
    
    def test_invalid_output_directory(self):
        """Scenario 19: Invalid output directory"""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "input.csv")
            output_file = "/nonexistent/directory/output.csv"
            
            test_df = pd.DataFrame({'col1': [1, 2, 3]})
            test_df.to_csv(input_file, index=False)
            
            # Act & Assert
            with pytest.raises((FileNotFoundError, PermissionError, OSError)):
                limpiar_datos(input_file, output_file, [])