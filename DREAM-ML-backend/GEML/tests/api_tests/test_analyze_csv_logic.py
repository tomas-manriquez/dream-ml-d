import pytest
import pandas as pd
import io
from unittest.mock import Mock, patch
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
import tempfile
import os


# Assuming the function is imported from your module
# from your_module import analyze_csv_logic


def analyze_csv_logic(csv_file) -> dict:
    """
    Lógica interna para analizar columnas de un CSV.
    - Recibe un 'csv_file' (un archivo ya abierto o un objeto InMemoryUploadedFile).
    - Retorna un dict con { "columns": [...] }.
    Lanza excepción si hay error al leer el CSV.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Leer solo las cabeceras del CSV para optimizar
        df = pd.read_csv(csv_file, nrows=0)
        columns = list(df.columns)
        return {"columns": columns}
    except Exception as e:
        logger.error(f"Error al analizar el CSV: {e}", exc_info=True)
        raise


class TestAnalyzeCsvLogic:
    
    def create_csv_file(self, content: str) -> io.StringIO:
        """Helper method to create a CSV file object from string content."""
        return io.StringIO(content)
    
    def create_binary_csv_file(self, content: str, encoding: str = 'utf-8') -> io.BytesIO:
        """Helper method to create a binary CSV file object."""
        return io.BytesIO(content.encode(encoding))
    
    def create_in_memory_uploaded_file(self, content: str, name: str = 'test.csv') -> InMemoryUploadedFile:
        """Helper method to create InMemoryUploadedFile."""
        csv_bytes = content.encode('utf-8')
        return InMemoryUploadedFile(
            file=io.BytesIO(csv_bytes),
            field_name='csv_file',
            name=name,
            content_type='text/csv',
            size=len(csv_bytes),
            charset='utf-8'
        )
    
    from django.core.files.uploadedfile import SimpleUploadedFile

    def create_temporary_uploaded_file(self, content: str, name: str = 'test.csv') -> SimpleUploadedFile:
        """Helper method to create SimpleUploadedFile for testing."""
        from django.core.files.uploadedfile import SimpleUploadedFile
        return SimpleUploadedFile(
            name=name,
            content=content.encode('utf-8'),
            content_type='text/csv'
        )

    
    
    # Happy Path Scenarios
    
    def test_standard_csv_with_multiple_columns(self):
        """Scenario 1: Standard CSV with multiple columns"""
        # Arrange
        csv_content = "name,age,email\nJohn,25,john@email.com"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["name", "age", "email"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_single_column(self):
        """Scenario 2: CSV with single column"""
        # Arrange
        csv_content = "id\n1\n2\n3"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["id"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_many_columns(self):
        """Scenario 3: CSV with many columns (50+)"""
        # Arrange
        column_names = [f"col_{i}" for i in range(1, 51)]  # 50 columns
        csv_header = ",".join(column_names)
        csv_content = f"{csv_header}\n" + ",".join(["data"] * 50)
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": column_names}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
        assert len(result["columns"]) == 50
    
    def test_csv_with_special_characters_in_column_names(self):
        """Scenario 4: CSV with special characters in column names"""
        # Arrange
        csv_content = "User Name,Age (years),Email@Address,#ID\nJohn Doe,25,john@email.com,123"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["User Name", "Age (years)", "Email@Address", "#ID"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_unicode_characters_in_column_names(self):
        """Scenario 5: CSV with Unicode characters in column names"""
        # Arrange
        csv_content = "名前,年齢,メール\n田中,30,tanaka@email.com"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["名前", "年齢", "メール"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    # Edge Cases
    
    def test_csv_with_only_headers(self):
        """Scenario 6: CSV with only headers (no data rows)"""
        # Arrange
        csv_content = "col1,col2,col3"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["col1", "col2", "col3"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_empty_column_names(self):
        """Scenario 7: CSV with empty column names"""
        # Arrange
        csv_content = "name,,age\nJohn,,25"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["name", "Unnamed: 1", "age"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_duplicate_column_names(self):
        """Scenario 8: CSV with duplicate column names"""
        # Arrange
        csv_content = "name,age,name\nJohn,25,Doe"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["name", "age", "name.1"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_whitespace_in_headers(self):
        """Scenario 9: CSV with leading/trailing whitespace in headers"""
        # Arrange
        csv_content = " name , age , email \nJohn,25,john@email.com"
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": [" name ", " age ", " email "]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    # File Object Edge Cases
    
    def test_in_memory_uploaded_file_object(self):
        """Scenario 10: InMemoryUploadedFile object"""
        # Arrange
        csv_content = "name,age,email\nJohn,25,john@email.com"
        csv_file = self.create_in_memory_uploaded_file(csv_content)
        expected_result = {"columns": ["name", "age", "email"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_temporary_uploaded_file_object(self):
        """Scenario 11: SimpleUploadedFile object (renamed for accuracy)"""
        # Arrange
        csv_content = "name,age,email\nJohn,25,john@email.com"
        csv_file = self.create_temporary_uploaded_file(csv_content)
        expected_result = {"columns": ["name", "age", "email"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
        # No cleanup needed - SimpleUploadedFile is in memory
        
        # Error Scenarios
    
    def test_completely_empty_file(self):
        """Scenario 12: Completely empty file (0 bytes)"""
        # Arrange
        csv_file = io.StringIO("")
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
    
    def test_file_with_invalid_csv_format(self):
        """Scenario 13: File with invalid CSV format"""
        # Arrange
        csv_content = 'name,age\n"John,25\nBroken CSV'  # Unmatched quotes
        csv_file = self.create_csv_file(csv_content)
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
    
    def test_none_file_object(self):
        """Scenario 16: None or invalid file object"""
        # Arrange
        csv_file = None
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
    
    def test_invalid_file_object(self):
        """Scenario 16: Invalid file object"""
        # Arrange
        csv_file = "not_a_file_object"
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
    
    def test_closed_file_object(self):
        """Scenario 17: Closed file object"""
        # Arrange
        csv_file = io.StringIO("name,age\nJohn,25")
        csv_file.close()
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
    
    # Pandas-Specific Behavior Tests
    
    def test_csv_with_semicolon_separators(self):
        """Scenario 18: CSV with semicolon separators"""
        # Arrange
        csv_content = "name;age;email\nJohn;25;john@email.com"
        csv_file = self.create_csv_file(csv_content)
        # Since we're using pandas defaults, it should treat this as a single column
        expected_result = {"columns": ["name;age;email"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result
    
    def test_csv_with_quoted_headers_containing_commas(self):
        """Scenario 19: CSV with quoted headers containing commas"""
        # Arrange
        csv_content = '"Last, First",Age,"City, State"\n"Doe, John",25,"New York, NY"'
        csv_file = self.create_csv_file(csv_content)
        expected_result = {"columns": ["Last, First", "Age", "City, State"]}
        
        # Act
        result = analyze_csv_logic(csv_file)
        
        # Assert
        assert result == expected_result


# Additional test with mocked logger to verify logging behavior
class TestAnalyzeCsvLogicWithLogging:
    
    @patch('logging.getLogger')
    def test_error_logging_on_exception(self, mock_get_logger):
        """Test that errors are properly logged when exceptions occur"""
        # Arrange
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        csv_file = None  # This will cause an exception
        
        # Act & Assert
        with pytest.raises(Exception):
            analyze_csv_logic(csv_file)
        
        # Verify that error was logged
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error al analizar el CSV" in args[0]
        assert kwargs.get('exc_info') is True