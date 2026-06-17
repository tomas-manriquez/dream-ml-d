"""
Unit tests for api.data_encoding module.

Tests the codificar_datos() function which handles:
- get_dummies encoding for categorical input features
- OneHotEncoder for multi-class targets
- LabelEncoder for binary/categorical targets
- Boolean to integer conversion
- Filtering of "_vacio" columns
"""
import pytest
import pandas as pd
import numpy as np
import os

from api.data_encoding import codificar_datos


@pytest.mark.unit
class TestCodificarDatos:
    """Test cases for codificar_datos() function."""

    def test_get_dummies_encoding_for_object_input_features(self, temp_experiment_dir):
        """
        Scenario 1: get_dummies encoding for object-type input features
        Given a CSV with categorical input features
        When codificar_datos is called
        Then categorical features should be encoded with get_dummies (drop_first=True)
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'C', 'A', 'B'],
            'num_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        input_features = ['cat_feature', 'num_feature']
        target_variables = ['target']

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=input_features,
            target_variables=target_variables,
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert - DataFrame return value
        assert 'cat_feature' not in result_df.columns, "Original categorical column should be removed"
        assert 'cat_feature_B' in result_df.columns, "Dummy column for 'B' should exist (drop_first=True drops 'A')"
        assert 'cat_feature_C' in result_df.columns, "Dummy column for 'C' should exist"
        assert 'num_feature' in result_df.columns, "Numeric feature should be preserved"
        assert 'target' in result_df.columns, "Target should be preserved"

        # Assert - CSV file created and matches
        assert os.path.exists(output_csv), "Output CSV file should be created"
        saved_df = pd.read_csv(output_csv)
        pd.testing.assert_frame_equal(result_df, saved_df)


    def test_onehot_encoder_for_categorical_target(self, temp_experiment_dir):
        """
        Scenario 2: OneHotEncoder for multi-class target
        Given a CSV with categorical target (A, B, C)
        When apply_ohe_to_target=True
        Then target should be one-hot encoded into separate columns
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'target': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['target'],
            apply_ohe_to_target=True,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'target' not in result_df.columns, "Original target should be removed"
        assert 'target_A' in result_df.columns, "OneHot column for 'A' should exist"
        assert 'target_B' in result_df.columns, "OneHot column for 'B' should exist"
        assert 'target_C' in result_df.columns, "OneHot column for 'C' should exist"

        # Verify CSV file
        assert os.path.exists(output_csv)
        saved_df = pd.read_csv(output_csv)
        pd.testing.assert_frame_equal(result_df, saved_df)


    def test_label_encoder_for_binary_target(self, temp_experiment_dir):
        """
        Scenario 3: LabelEncoder for binary/categorical target
        Given a CSV with categorical target (Yes/No)
        When apply_labelencoder_to_target=True
        Then target should be encoded as integers (0, 1)
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=True
        )

        # Assert
        assert 'target' in result_df.columns, "Target column should still exist"
        assert result_df['target'].dtype in [np.int64, np.int32], "Target should be integer type"
        assert set(result_df['target'].unique()) == {0, 1}, "Binary encoding should produce 0 and 1"

        # Verify CSV file
        assert os.path.exists(output_csv)


    def test_mutual_exclusivity_validation_raises_error(self, temp_experiment_dir):
        """
        Scenario 4: Mutual exclusivity validation
        Given both apply_ohe_to_target=True AND apply_labelencoder_to_target=True
        When codificar_datos is called
        Then should raise ValueError
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': ['A', 'B', 'C']
        })
        df.to_csv(input_csv, index=False)

        # Act & Assert
        with pytest.raises(ValueError, match="No se puede usar OHE y LabelEncoder simultáneamente"):
            codificar_datos(
                csv_input=input_csv,
                csv_output_train=output_csv,
                input_features=['feature1'],
                target_variables=['target'],
                apply_ohe_to_target=True,
                apply_labelencoder_to_target=True
            )


    def test_vacio_column_filtering_for_input_features(self, temp_experiment_dir):
        """
        Scenario 5: "_vacio" column filtering for input features
        Given input_features containing columns ending with "_vacio"
        When codificar_datos is called
        Then "_vacio" columns should be filtered out and not processed
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'real_feature': ['A', 'B', 'A'],
            'placeholder_vacio': ['X', 'Y', 'Z'],  # Should be filtered
            'target': [0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        input_features = ['real_feature', 'placeholder_vacio']  # Include _vacio
        target_variables = ['target']

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=input_features,
            target_variables=target_variables,
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'real_feature_B' in result_df.columns, "Real feature should be encoded"
        # _vacio column should still exist in DataFrame (not encoded, just ignored)
        assert 'placeholder_vacio' in result_df.columns, "_vacio column preserved but not encoded"


    def test_vacio_column_filtering_for_target_variables(self, temp_experiment_dir):
        """
        Scenario 6: "_vacio" column filtering for target variables
        Given target_variables containing columns ending with "_vacio"
        When codificar_datos is called
        Then "_vacio" targets should be filtered out and not encoded
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'real_target': ['A', 'B', 'C'],
            'placeholder_target_vacio': ['X', 'Y', 'Z']  # Should be filtered
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['real_target', 'placeholder_target_vacio'],
            apply_ohe_to_target=True,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'real_target_A' in result_df.columns, "Real target should be OHE encoded"
        # _vacio target should still exist (not encoded, just ignored)
        assert 'placeholder_target_vacio' in result_df.columns, "_vacio target preserved but not encoded"


    def test_boolean_to_integer_conversion(self, temp_experiment_dir):
        """
        Scenario 7: Boolean columns converted to integers
        Given a CSV with boolean columns
        When codificar_datos is called
        Then boolean columns should be converted to 0/1
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'bool_feature': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['bool_feature'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'bool_feature' in result_df.columns, "Boolean feature should exist"
        assert result_df['bool_feature'].dtype in [np.int64, np.int32], "Boolean should be converted to int"
        assert set(result_df['bool_feature'].unique()) == {0, 1}, "Boolean values should be 0 and 1"

        # Verify CSV file
        assert os.path.exists(output_csv)


    def test_mixed_data_types_object_numeric_boolean(self, temp_experiment_dir):
        """
        Scenario 8: Mixed data types (object, numeric, boolean)
        Given a CSV with object, numeric, and boolean features
        When codificar_datos is called
        Then object features encoded, numeric preserved, boolean converted to int
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'A', 'B', 'A'],
            'num_feature': [1.5, 2.3, 3.7, 4.2, 5.9],
            'bool_feature': [True, False, True, False, True],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['cat_feature', 'num_feature', 'bool_feature'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        # Categorical feature encoded
        assert 'cat_feature' not in result_df.columns
        assert 'cat_feature_B' in result_df.columns

        # Numeric feature preserved
        assert 'num_feature' in result_df.columns
        assert result_df['num_feature'].dtype in [np.float64, np.float32]

        # Boolean converted to int
        assert 'bool_feature' in result_df.columns
        assert result_df['bool_feature'].dtype in [np.int64, np.int32]


    def test_int64_target_with_label_encoder(self, temp_experiment_dir):
        """
        Scenario 9: int64 target with LabelEncoder
        Given a CSV with int64 target variable
        When apply_labelencoder_to_target=True
        Then target should be encoded (even though already numeric)
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [10, 20, 30, 10, 20]  # int64 target
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=True
        )

        # Assert
        assert 'target' in result_df.columns
        # LabelEncoder should map the values to 0, 1, 2 (three unique values: 10, 20, 30)
        assert set(result_df['target'].unique()) == {0, 1, 2}


    def test_empty_dataframe_handling(self, temp_experiment_dir):
        """
        Scenario 10: Handling of empty DataFrame
        Given an empty CSV file
        When codificar_datos is called
        Then should handle gracefully and return empty result
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "empty.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame(columns=['feature1', 'target'])
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert len(result_df) == 0, "Result should be empty"
        assert os.path.exists(output_csv), "Output CSV should be created even if empty"
        saved_df = pd.read_csv(output_csv)
        assert len(saved_df) == 0, "Saved CSV should be empty"


    def test_file_not_found_error(self, temp_experiment_dir):
        """
        Scenario 11: Error when input CSV file doesn't exist
        Given an invalid CSV path
        When codificar_datos is called
        Then should raise FileNotFoundError
        """
        # Arrange
        invalid_csv = os.path.join(temp_experiment_dir, "nonexistent.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            codificar_datos(
                csv_input=invalid_csv,
                csv_output_train=output_csv,
                input_features=['feature1'],
                target_variables=['target'],
                apply_ohe_to_target=False,
                apply_labelencoder_to_target=False
            )


    def test_no_encoding_needed_all_numeric(self, temp_experiment_dir):
        """
        Scenario 12: No encoding needed - all numeric features
        Given a CSV with all numeric features
        When codificar_datos is called with no target encoding
        Then data should pass through with minimal changes
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'num_feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num_feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['num_feature1', 'num_feature2'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'num_feature1' in result_df.columns
        assert 'num_feature2' in result_df.columns
        assert 'target' in result_df.columns

        # Values should be unchanged
        pd.testing.assert_series_equal(
            result_df['num_feature1'],
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='num_feature1'),
            check_dtype=False
        )


    def test_preserves_numeric_features_unchanged(self, temp_experiment_dir):
        """
        Scenario 13: Numeric features preserved without modification
        Given a CSV with numeric features alongside categorical
        When codificar_datos is called
        Then numeric features should remain unchanged in values
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        original_numeric_values = [1.5, 2.3, 3.7, 4.2, 5.9]
        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'A', 'B', 'A'],
            'numeric_feature': original_numeric_values,
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['cat_feature', 'numeric_feature'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        pd.testing.assert_series_equal(
            result_df['numeric_feature'],
            pd.Series(original_numeric_values, name='numeric_feature'),
            check_dtype=False
        )


    def test_category_dtype_with_onehot_encoder(self, temp_experiment_dir):
        """
        Scenario 14: Category dtype target with OneHotEncoder
        Given a CSV with pandas 'category' dtype target
        When apply_ohe_to_target=True
        Then target should be one-hot encoded
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': pd.Categorical(['Low', 'Medium', 'High', 'Low', 'Medium'])
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['feature1'],
            target_variables=['target'],
            apply_ohe_to_target=True,
            apply_labelencoder_to_target=False
        )

        # Assert
        assert 'target' not in result_df.columns
        # OneHot columns should exist for the categories
        ohe_columns = [col for col in result_df.columns if col.startswith('target_')]
        assert len(ohe_columns) == 3, "Should have 3 OneHot columns for 3 categories"


    def test_multiple_categorical_features_encoded_correctly(self, temp_experiment_dir):
        """
        Scenario 15: Multiple categorical features encoded correctly
        Given a CSV with multiple categorical features
        When codificar_datos is called
        Then each categorical feature should be independently encoded
        """
        # Arrange
        input_csv = os.path.join(temp_experiment_dir, "input.csv")
        output_csv = os.path.join(temp_experiment_dir, "output.csv")

        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(input_csv, index=False)

        # Act
        result_df = codificar_datos(
            csv_input=input_csv,
            csv_output_train=output_csv,
            input_features=['cat1', 'cat2', 'num1'],
            target_variables=['target'],
            apply_ohe_to_target=False,
            apply_labelencoder_to_target=False
        )

        # Assert
        # cat1 encoded (drop_first=True, so only 'B' column)
        assert 'cat1' not in result_df.columns
        assert 'cat1_B' in result_df.columns

        # cat2 encoded (drop_first=True, so 'Y' and 'Z' columns, 'X' dropped)
        assert 'cat2' not in result_df.columns
        assert 'cat2_Y' in result_df.columns
        assert 'cat2_Z' in result_df.columns

        # Numeric preserved
        assert 'num1' in result_df.columns
