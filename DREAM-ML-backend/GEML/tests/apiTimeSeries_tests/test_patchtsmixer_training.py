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
Test suite for PatchTSMixer training pipeline - Phase 4

Tests cover:
- Main training function (train_patchtsmixer_model)
- Manual training with Trainer API (train_manual_patchtsmixer)
- Evaluation with metrics and plots (evaluate_patchtsmixer)
- MLflow integration
- Pipeline config generation
- Error handling
"""

import pytest
import numpy as np
import pandas as pd
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Import functions to test
from apiTimeSeries.train import (
    train_patchtsmixer_model,
    train_manual_patchtsmixer,
    evaluate_patchtsmixer,
    TimeSeriesDataset
)


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================


def create_synthetic_dataset(tmp_path, n_rows=500, n_channels=3):
    """
    Helper para crear dataset sintético de prueba.

    Args:
        tmp_path: Path temporal de pytest
        n_rows: Número de filas
        n_channels: Número de canales/variables

    Returns:
        Path al CSV creado
    """
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')

    data = {
        'date': dates
    }

    for i in range(n_channels):
        # Crear serie temporal con tendencia + ruido
        trend = np.linspace(10, 20, n_rows)
        noise = np.random.normal(0, 1, n_rows)
        data[f'channel_{i}'] = trend + noise

    df = pd.DataFrame(data)
    csv_path = os.path.join(tmp_path, 'test_data.csv')
    df.to_csv(csv_path, index=False)

    return csv_path


# ======================================================================================
# TRAINING TESTS
# ======================================================================================


def test_manual_training_completes(tmp_path):
    """
    Test que verifica que train_patchtsmixer_model() completa sin errores
    con configuración mínima.
    """
    # Crear dataset sintético
    csv_path = create_synthetic_dataset(tmp_path, n_rows=500, n_channels=2)

    # Configuración mínima para test rápido
    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,  # Pequeño para test
            "patch_length": 8,
            "d_model": 8,          # Muy pequeño para test rápido
            "num_layers": 2,       # Pocas capas para test rápido
            "dropout": 0.1,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 2,           # Solo 2 épocas para test
            "early_stopping_patience": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")
    os.makedirs(experiment_dir, exist_ok=True)

    # Ejecutar entrenamiento
    result = train_patchtsmixer_model(csv_path, data, experiment_dir)

    # Verificaciones
    assert isinstance(result, dict)
    assert "val_metrics" in result
    assert "test_metrics" in result
    assert "model_path" in result

    # Verificar que métricas tienen las claves esperadas
    for key in ["val_rmse", "val_mae", "val_mape", "val_mse"]:
        assert key in result["val_metrics"]
        # Verificar que es un número válido (puede ser None para MAPE)
        if result["val_metrics"][key] is not None:
            assert isinstance(result["val_metrics"][key], (int, float))
            assert not np.isnan(result["val_metrics"][key])

    # Verificar que modelo se guardó
    assert os.path.exists(result["model_path"])
    # HuggingFace saves as model.safetensors (modern) or pytorch_model.bin (legacy)
    model_weights_exists = (
        os.path.exists(os.path.join(result["model_path"], "model.safetensors")) or
        os.path.exists(os.path.join(result["model_path"], "pytorch_model.bin"))
    )
    assert model_weights_exists, f"Model weights not found in {result['model_path']}"
    assert os.path.exists(os.path.join(result["model_path"], "config.json"))

    # Verificar que pipeline_config.json se creó
    config_path = os.path.join(experiment_dir, "pipeline_config.json")
    assert os.path.exists(config_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["model_type"] == "PatchTSMixer"
    assert "data_params" in config
    assert "model_params" in config
    assert "training_params" in config
    assert "results" in config


@patch('apiTimeSeries.train.mlflow')
def test_mlflow_logging(mock_mlflow, tmp_path):
    """
    Test que verifica que se loguean parámetros y métricas a MLflow.
    """
    # Crear dataset sintético
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 1,
            "early_stopping_patience": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Mock MLflow context manager
    mock_mlflow.start_run.return_value.__enter__ = Mock()
    mock_mlflow.start_run.return_value.__exit__ = Mock()

    # Ejecutar entrenamiento
    result = train_patchtsmixer_model(csv_path, data, experiment_dir)

    # Verificar que se llamó a MLflow
    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metrics.called

    # Verificar que se loguearon métricas correctas
    calls = mock_mlflow.log_metrics.call_args_list
    logged_metrics = {}
    for call in calls:
        logged_metrics.update(call[0][0])  # First positional arg

    # Verificar que se loguearon métricas de val y test
    assert any(key.startswith("val_") for key in logged_metrics.keys())
    assert any(key.startswith("test_") for key in logged_metrics.keys())


def test_trainer_creates_checkpoints(tmp_path):
    """
    Test que verifica que se crean checkpoints durante entrenamiento.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 3,  # Varias épocas para generar checkpoints
            "early_stopping_patience": 5
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Ejecutar entrenamiento
    result = train_patchtsmixer_model(csv_path, data, experiment_dir)

    # Verificar que se creó directorio de checkpoints
    checkpoint_dir = os.path.join(experiment_dir, "patchtsmixer_checkpoints")
    assert os.path.exists(checkpoint_dir)

    # Verificar que hay archivos de checkpoint
    checkpoint_files = os.listdir(checkpoint_dir)
    assert len(checkpoint_files) > 0

    # Verificar que hay subdirectorios de checkpoint (checkpoint-X)
    checkpoint_subdirs = [f for f in checkpoint_files if f.startswith("checkpoint-")]
    assert len(checkpoint_subdirs) > 0


def test_reproducibility(tmp_path):
    """
    Test que verifica que mismo seed produce resultados reproducibles.

    Nota: Puede haber pequeñas diferencias debido a non-determinismo en PyTorch,
    por lo que usamos una tolerancia razonable.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 2,
            "early_stopping_patience": 1
        }
    }

    # Run 1
    experiment_dir_1 = os.path.join(tmp_path, "exp_1")
    result_1 = train_patchtsmixer_model(csv_path, data, experiment_dir_1)

    # Run 2 (same seed via set_global_seeds)
    experiment_dir_2 = os.path.join(tmp_path, "exp_2")
    result_2 = train_patchtsmixer_model(csv_path, data, experiment_dir_2)

    # Verificar que métricas son similares (tolerancia del 5% debido a non-determinismo)
    val_rmse_1 = result_1["val_metrics"]["val_rmse"]
    val_rmse_2 = result_2["val_metrics"]["val_rmse"]

    # Tolerancia del 5% para reproducibilidad (PyTorch en CPU puede tener variaciones)
    assert np.abs(val_rmse_1 - val_rmse_2) / val_rmse_1 < 0.05


def test_pipeline_config_generation(tmp_path):
    """
    Test que verifica que pipeline_config.json se genera correctamente.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 16,
            "num_layers": 4,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 2,
            "early_stopping_patience": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Ejecutar entrenamiento
    result = train_patchtsmixer_model(csv_path, data, experiment_dir)

    # Verificar que pipeline_config.json existe
    config_path = os.path.join(experiment_dir, "pipeline_config.json")
    assert os.path.exists(config_path)

    # Leer y verificar contenido
    with open(config_path) as f:
        config = json.load(f)

    # Verificar estructura
    assert config["model_type"] == "PatchTSMixer"
    assert "data_params" in config
    assert "model_params" in config
    assert "training_params" in config
    assert "reproducibility" in config
    assert "results" in config

    # Verificar contenido de data_params
    assert config["data_params"]["forecast_horizon"] == 24
    assert config["data_params"]["patchtsmixer_channels"] == ["channel_0", "channel_1"]

    # Verificar contenido de model_params
    assert config["model_params"]["d_model"] == 16
    assert config["model_params"]["num_layers"] == 4
    assert config["model_params"]["patch_length"] == 8

    # Verificar contenido de training_params
    assert config["training_params"]["learning_rate"] == 0.001
    assert config["training_params"]["batch_size"] == 32
    assert config["training_params"]["strategy"] == "manual"

    # Verificar que results contiene métricas
    assert "val_metrics" in config["results"]
    assert "test_metrics" in config["results"]
    assert "val_rmse" in config["results"]["val_metrics"]


# ======================================================================================
# ERROR HANDLING TESTS
# ======================================================================================


def test_error_handling_invalid_context_length(tmp_path):
    """
    Test que verifica manejo de errores cuando context_length no es múltiplo de patch_length.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 100,  # NO es múltiplo de 8
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Debe lanzar ValueError
    with pytest.raises(ValueError, match="context_length.*debe ser múltiplo"):
        train_patchtsmixer_model(csv_path, data, experiment_dir)


def test_error_handling_missing_columns(tmp_path):
    """
    Test que verifica manejo de errores cuando columnas no existen en dataset.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_999"],  # channel_999 no existe
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Debe lanzar ValueError
    with pytest.raises(ValueError, match="columnas no existen"):
        train_patchtsmixer_model(csv_path, data, experiment_dir)


def test_error_handling_insufficient_data(tmp_path):
    """
    Test que verifica manejo de errores cuando dataset es muy pequeño.
    """
    # Crear dataset muy pequeño
    csv_path = create_synthetic_dataset(tmp_path, n_rows=50, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 96,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 512,  # Muy largo para dataset pequeño
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Debe lanzar ValueError
    with pytest.raises(ValueError, match="Dataset muy pequeño"):
        train_patchtsmixer_model(csv_path, data, experiment_dir)


# ======================================================================================
# PLOT GENERATION TESTS
# ======================================================================================


def test_plots_are_generated(tmp_path):
    """
    Test que verifica que se generan plots de evaluación.
    """
    csv_path = create_synthetic_dataset(tmp_path, n_rows=300, n_channels=2)

    data = {
        "date_col_name": "date",
        "patchtsmixer_channels": ["channel_0", "channel_1"],
        "forecast_horizon": 24,
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "manual_params": {
            "context_length": 96,
            "patch_length": 8,
            "d_model": 8,
            "num_layers": 2,
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 2,
            "early_stopping_patience": 1
        }
    }

    experiment_dir = os.path.join(tmp_path, "test_exp")

    # Ejecutar entrenamiento
    result = train_patchtsmixer_model(csv_path, data, experiment_dir)

    # Verificar que se generaron plots para val (4 después de Phase 5)
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_val_forecast.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_val_residuals.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_val_residuals_distribution.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_val_horizons.png"))

    # Verificar que se generaron plots para test (4 después de Phase 5)
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_test_forecast.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_test_residuals.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_test_residuals_distribution.png"))
    assert os.path.exists(os.path.join(experiment_dir, "patchtsmixer_test_horizons.png"))


# ======================================================================================
# PHASE 5 TESTS - EVALUATION & MULTI-HORIZON METRICS
# ======================================================================================

"""
Pruebas unitarias para evaluación de PatchTSMixer (Phase 5).

Este módulo prueba:
- Cálculo de métricas agregadas y por horizonte
- Manejo de casos límite (prediction_length < 3)
- Generación de gráficos PNG válidos
- Consistencia del naming pattern de métricas
"""

# Conditional import for PIL with skip marker
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class TestEvaluationMetrics:
    """Pruebas para cálculo de métricas de evaluación."""

    def test_evaluation_metrics_all_keys_present(self, tmp_path):
        """Verifica que todas las claves esperadas estén presentes en métricas."""
        # Arrange: crear datos sintéticos
        n_samples, prediction_length, num_channels = 100, 96, 3
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        # Mock del trainer y dataset
        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        # Act: ejecutar función
        metrics, artifacts = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: verificar claves de métricas agregadas
        assert "val_rmse" in metrics
        assert "val_mae" in metrics
        assert "val_mape" in metrics
        assert "val_mse" in metrics

        # Assert: verificar claves de métricas por horizonte
        assert "val_rmse_h1" in metrics
        assert "val_mae_h1" in metrics
        assert "val_rmse_h48" in metrics  # horizonte medio para prediction_length=96
        assert "val_rmse_h96" in metrics  # último horizonte

    def test_evaluation_metrics_finite_values(self, tmp_path):
        """Verifica que todas las métricas sean valores finitos."""
        n_samples, prediction_length, num_channels = 50, 24, 2
        # Usar valores positivos para evitar problemas con MAPE
        y_true = np.abs(np.random.randn(n_samples, prediction_length, num_channels)) + 0.1
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.05

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "test", str(tmp_path)
        )

        # Assert: todos los valores deben ser finitos
        for key, value in metrics.items():
            if value is not None:
                assert np.isfinite(value), f"Métrica {key} no es finita: {value}"


class TestKeyHorizonMetrics:
    """Pruebas para métricas de horizontes clave."""

    def test_key_horizon_metrics_prediction_length_96(self, tmp_path):
        """Verifica horizontes clave para prediction_length=96."""
        n_samples, prediction_length, num_channels = 32, 96, 3
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: horizontes esperados h1, h48, h96
        assert "val_rmse_h1" in metrics
        assert "val_rmse_h48" in metrics  # 96 // 2 = 48
        assert "val_rmse_h96" in metrics

        # Assert: valores razonables (RMSE debe ser positivo)
        assert metrics["val_rmse_h1"] > 0
        assert metrics["val_rmse_h48"] > 0
        assert metrics["val_rmse_h96"] > 0

    def test_key_horizon_metrics_short_prediction_length_2(self, tmp_path):
        """Verifica manejo de prediction_length=2 (solo h1 y h2)."""
        n_samples, prediction_length, num_channels = 32, 2, 1
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: solo h1 y h2 para prediction_length=2
        assert "val_rmse_h1" in metrics
        assert "val_rmse_h2" in metrics
        # No debe haber horizonte medio (solo existe en prediction_length >= 3)
        assert "val_rmse_h48" not in metrics

    def test_key_horizon_metrics_short_prediction_length_1(self, tmp_path):
        """Verifica manejo de prediction_length=1 (solo h1)."""
        n_samples, prediction_length, num_channels = 32, 1, 1
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: solo h1 para prediction_length=1
        assert "val_rmse_h1" in metrics
        assert "val_rmse_h2" not in metrics

    @pytest.mark.parametrize("prediction_length,expected_middle", [
        (96, 48),   # par: 96 // 2 = 48
        (95, 47),   # impar: 95 // 2 = 47 (floor division)
        (24, 12),   # par pequeño
        (25, 12),   # impar pequeño: 25 // 2 = 12
        (10, 5),    # mínimo para 3 horizontes
    ])
    def test_middle_horizon_rounding(self, tmp_path, prediction_length, expected_middle):
        """Verifica que el horizonte medio use floor division."""
        n_samples, num_channels = 32, 2
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: horizonte medio correcto usando floor division
        assert f"val_rmse_h{expected_middle}" in metrics


class TestPlotting:
    """Pruebas para funciones de generación de gráficos."""

    @pytest.mark.slow
    def test_plotting_files_created(self, tmp_path):
        """Verifica que se creen todos los archivos PNG esperados."""
        n_samples, prediction_length, num_channels = 50, 24, 2
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        _, artifacts = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: archivos creados (4 después de Phase 5)
        assert len(artifacts) == 4  # forecast, residuals, residuals_dist, horizons

        expected_files = [
            "patchtsmixer_val_forecast.png",
            "patchtsmixer_val_residuals.png",
            "patchtsmixer_val_residuals_distribution.png",
            "patchtsmixer_val_horizons.png",
        ]

        for expected_file in expected_files:
            full_path = os.path.join(tmp_path, expected_file)
            assert os.path.exists(full_path), f"Archivo no encontrado: {expected_file}"

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_plotting_valid_png_images(self, tmp_path):
        """Verifica que los archivos generados sean imágenes PNG válidas."""
        n_samples, prediction_length, num_channels = 30, 12, 1
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        _, artifacts = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "test", str(tmp_path)
        )

        # Assert: cada archivo es una imagen PNG válida
        for artifact_path in artifacts:
            assert os.path.exists(artifact_path), f"Archivo no existe: {artifact_path}"
            # Intentar abrir como imagen para validar formato
            img = Image.open(artifact_path)
            assert img.format == "PNG", f"Formato inválido para {artifact_path}"
            img.close()

    def test_horizons_plot_short_prediction_length(self, tmp_path):
        """Verifica que horizons plot maneje prediction_length cortos."""
        n_samples, prediction_length, num_channels = 30, 2, 1
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        _, artifacts = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: horizons plot creado incluso con solo 2 horizontes
        horizons_path = os.path.join(tmp_path, "patchtsmixer_val_horizons.png")
        assert os.path.exists(horizons_path), "Horizons plot no creado para prediction_length=2"


class TestMAPEEdgeCases:
    """Pruebas para casos límite de MAPE."""

    def test_mape_with_zero_values(self, tmp_path):
        """Verifica que MAPE maneje valores cero correctamente."""
        n_samples, prediction_length, num_channels = 32, 24, 2
        # Crear datos con algunos ceros
        y_true = np.random.randn(n_samples, prediction_length, num_channels)
        y_true[0, :, :] = 0  # Primera muestra toda ceros
        y_pred = y_true + np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        # No debe lanzar excepción
        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: MAPE puede ser calculado (o None si todos son cero)
        # En este caso, no todos son cero, así que debe existir
        assert "val_mape" in metrics

    def test_mape_all_zeros_returns_none(self, tmp_path):
        """Verifica que MAPE retorne None cuando todos los valores son cero."""
        n_samples, prediction_length, num_channels = 32, 24, 2
        # Todos ceros para ground truth
        y_true = np.zeros((n_samples, prediction_length, num_channels))
        y_pred = np.random.randn(n_samples, prediction_length, num_channels) * 0.1

        mock_trainer = Mock()
        mock_predictions = Mock()
        mock_predictions.predictions = y_pred
        mock_trainer.predict.return_value = mock_predictions

        mock_dataset = Mock()
        mock_dataset.future_values = Mock()
        mock_dataset.future_values.numpy.return_value = y_true

        # No debe lanzar excepción
        metrics, _ = evaluate_patchtsmixer(
            mock_trainer, mock_dataset, "val", str(tmp_path)
        )

        # Assert: MAPE debe ser None cuando todos son cero
        assert "val_mape" in metrics
        assert metrics["val_mape"] is None
        # Horizon MAPE también debe ser None
        assert "val_mape_h1" in metrics
        assert metrics["val_mape_h1"] is None
