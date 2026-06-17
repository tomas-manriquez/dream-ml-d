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

"""
Test suite for LSTM Phase 1 implementation.

Tests the core LSTM functionality including:
- Sequence creation
- Train/val/test split
- Model building (single and multi-layer)
"""

import pytest
import numpy as np
import pandas as pd
from apiTimeSeries.train import (
    create_sequences_for_lstm,
    lstm_train_val_test_split,
    build_lstm_model
)


class TestCreateSequencesForLSTM:
    """Tests for create_sequences_for_lstm function"""

    def test_create_sequences_basic(self):
        """Test basic sequence creation with univariate data"""
        # Generate synthetic data
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'value': np.sin(np.linspace(0, 10, 100))
        })
        df.set_index('date', inplace=True)

        X, y = create_sequences_for_lstm(df, ['value'], 'value', sequence_length=10)

        # 100 samples - 10 sequence_length - 1 forecast_horizon + 1 = 90 sequences
        assert X.shape == (90, 10, 1), f"Expected X.shape=(90, 10, 1), got {X.shape}"
        assert y.shape == (90,), f"Expected y.shape=(90,), got {y.shape}"
        assert X.dtype == np.float64
        assert y.dtype == np.float64

    def test_create_sequences_multivariate(self):
        """Test sequence creation with multivariate features"""
        # Generate synthetic multivariate data
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=150),
            'feature1': np.sin(np.linspace(0, 10, 150)),
            'feature2': np.cos(np.linspace(0, 10, 150)),
            'target': np.sin(np.linspace(0, 10, 150)) + np.random.randn(150) * 0.1
        })
        df.set_index('date', inplace=True)

        X, y = create_sequences_for_lstm(
            df,
            ['feature1', 'feature2'],
            'target',
            sequence_length=15
        )

        # 150 - 15 - 1 + 1 = 135 sequences
        assert X.shape == (135, 15, 2), f"Expected X.shape=(135, 15, 2), got {X.shape}"
        assert y.shape == (135,), f"Expected y.shape=(135,), got {y.shape}"

    def test_create_sequences_with_forecast_horizon(self):
        """Test sequence creation with custom forecast horizon"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=200),
            'value': np.linspace(0, 100, 200)
        })
        df.set_index('date', inplace=True)

        forecast_horizon = 5
        X, y = create_sequences_for_lstm(
            df,
            ['value'],
            'value',
            sequence_length=10,
            forecast_horizon=forecast_horizon
        )

        # 200 - 10 - 5 + 1 = 186 sequences
        assert X.shape == (186, 10, 1)
        assert y.shape == (186,)

        # Verify forecast horizon: y should be 5 steps ahead
        # X[0] uses data[0:10], y[0] should be data[10 + 5 - 1] = data[14]
        expected_y_0 = df['value'].iloc[14]
        assert np.isclose(y[0], expected_y_0), \
            f"Expected y[0]={expected_y_0}, got {y[0]}"

    def test_create_sequences_insufficient_data(self):
        """Test that error is raised when data is insufficient"""
        # Only 50 samples (less than min_sequences + sequence_length + forecast_horizon)
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50),
            'value': np.random.randn(50)
        })
        df.set_index('date', inplace=True)

        # This should fail because we need at least 50 sequences + 100 sequence_length + 1
        with pytest.raises(ValueError, match="Dataset insuficiente"):
            create_sequences_for_lstm(df, ['value'], 'value', sequence_length=100)


class TestLSTMTrainValTestSplit:
    """Tests for lstm_train_val_test_split function"""

    def test_lstm_train_val_test_split(self):
        """Test temporal split maintains 3D shape"""
        X = np.random.rand(100, 10, 2)
        y = np.random.rand(100)

        X_train, y_train, X_val, y_val, X_test, y_test = lstm_train_val_test_split(
            X, y, {"train": 0.7, "val": 0.15, "test": 0.15}
        )

        # Verify shapes
        assert X_train.shape == (70, 10, 2), f"Expected (70, 10, 2), got {X_train.shape}"
        assert X_val.shape == (15, 10, 2), f"Expected (15, 10, 2), got {X_val.shape}"
        assert X_test.shape == (15, 10, 2), f"Expected (15, 10, 2), got {X_test.shape}"

        assert y_train.shape == (70,)
        assert y_val.shape == (15,)
        assert y_test.shape == (15,)

        # Verify 3D shape is maintained
        assert X_train.ndim == 3
        assert X_val.ndim == 3
        assert X_test.ndim == 3

    def test_temporal_order_maintained(self):
        """Test that temporal order is preserved (no shuffling)"""
        # Create sequential data
        X = np.arange(300).reshape(100, 3, 1).astype(float)
        y = np.arange(100).astype(float)

        X_train, y_train, X_val, y_val, X_test, y_test = lstm_train_val_test_split(
            X, y, {"train": 0.7, "val": 0.15, "test": 0.15}
        )

        # Verify train uses earliest data
        assert np.array_equal(X_train[0], X[0])
        assert y_train[0] == y[0]

        # Verify test uses most recent data
        assert np.array_equal(X_test[-1], X[-1])
        assert y_test[-1] == y[-1]

        # Verify no overlap between sets
        assert y_train[-1] < y_val[0]
        assert y_val[-1] < y_test[0]

    def test_invalid_split_ratios(self):
        """Test that error is raised for invalid split ratios"""
        X = np.random.rand(100, 10, 2)
        y = np.random.rand(100)

        # Ratios don't sum to 1.0
        with pytest.raises(ValueError, match="debe ser 1.0"):
            lstm_train_val_test_split(
                X, y, {"train": 0.6, "val": 0.2, "test": 0.3}  # Sum = 1.1
            )


class TestBuildLSTMModel:
    """Tests for build_lstm_model function"""

    def test_build_lstm_model_single_layer(self):
        """Test single-layer LSTM model"""
        params = {
            "lstm_units": [64],
            "dropout_rate": 0.2,
            "recurrent_dropout_rate": 0.2,
            "learning_rate": 0.001
        }

        model = build_lstm_model(params, input_shape=(10, 2))

        assert model is not None
        # Single LSTM layer + Dense output layer = 2 layers
        assert len(model.layers) == 2, f"Expected 2 layers, got {len(model.layers)}"

        # Verify first layer is LSTM
        assert 'LSTM_Layer' in model.layers[0].name

        # Verify output layer is Dense with 1 unit
        assert model.layers[-1].units == 1

        # Verify model is compiled
        assert model.optimizer is not None
        assert model.loss == 'mse'

    def test_build_lstm_model_multi_layer(self):
        """Test multi-layer LSTM model"""
        params = {
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "recurrent_dropout_rate": 0.2,
            "learning_rate": 0.001
        }

        model = build_lstm_model(params, input_shape=(10, 2))

        # Two LSTM layers + Dense output layer = 3 layers
        assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"

        # Verify layer names
        assert 'LSTM_Layer_1' in model.layers[0].name
        assert 'LSTM_Layer_2' in model.layers[1].name
        assert 'Output_Layer' in model.layers[2].name

    def test_build_lstm_model_three_layers(self):
        """Test three-layer LSTM architecture"""
        params = {
            "lstm_units": [128, 64, 32],
            "dropout_rate": 0.3,
            "recurrent_dropout_rate": 0.3,
            "learning_rate": 0.0005
        }

        model = build_lstm_model(params, input_shape=(15, 3))

        # Three LSTM layers + Dense output layer = 4 layers
        assert len(model.layers) == 4

        # Verify units decrease as expected
        # Note: Can't directly access units from compiled model, but we can check layer count
        assert model.layers[-1].units == 1  # Output layer

    def test_build_lstm_model_with_defaults(self):
        """Test model building with default parameters"""
        params = {}  # Empty params, should use defaults

        model = build_lstm_model(params, input_shape=(10, 1))

        # Should default to single layer [64]
        assert len(model.layers) == 2  # LSTM + Dense
        assert model is not None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
