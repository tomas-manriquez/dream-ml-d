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
Test suite for PatchTSMixer model configuration and initialization.

Tests cover:
- Config creation with defaults and custom parameters
- Validation of hyperparameter ranges
- Model initialization and device placement
- Preset configurations
- JSON serialization for reproducibility
"""

import pytest
import numpy as np
from apiTimeSeries.train import (
    create_patchtsmixer_config,
    build_patchtsmixer_model,
    get_patchtsmixer_preset
)


def test_config_creation_with_defaults():
    """Test config creation with default parameters."""
    config = create_patchtsmixer_config({}, 1, 512, 96)

    # Verify defaults
    assert config.patch_length == 8
    assert config.d_model == 32
    assert config.num_layers == 8
    assert config.dropout == 0.2
    assert config.expansion_factor == 2
    assert config.mode == "common_channel"
    assert config.scaling == "std"
    assert config.loss == "mse"
    assert config.patch_stride == config.patch_length

    # Verify passed params
    assert config.num_input_channels == 1
    assert config.context_length == 512
    assert config.prediction_length == 96


def test_config_creation_with_custom_params():
    """Test config creation with custom parameters."""
    params = {
        "patch_length": 16,
        "d_model": 64,
        "num_layers": 12,
        "dropout": 0.3,
        "expansion_factor": 4,
        "mode": "mix_channel",
        "gated_attn": False,
    }

    config = create_patchtsmixer_config(params, 3, 512, 96)

    # Verify custom params
    assert config.patch_length == 16
    assert config.d_model == 64
    assert config.num_layers == 12
    assert config.dropout == 0.3
    assert config.expansion_factor == 4
    assert config.mode == "mix_channel"
    assert config.gated_attn == False


def test_config_validation_context_patch_divisibility():
    """Test that context_length % patch_length == 0 is enforced."""
    params = {"patch_length": 7}

    with pytest.raises(ValueError) as exc_info:
        create_patchtsmixer_config(params, 1, 512, 96)

    # Verify error message contains helpful info
    assert "debe ser divisible" in str(exc_info.value)
    assert "512" in str(exc_info.value)
    assert "7" in str(exc_info.value)


def test_config_validation_invalid_ranges():
    """Test validation of parameter ranges."""
    # Test negative patch_length
    with pytest.raises(ValueError, match="patch_length debe ser >= 1"):
        create_patchtsmixer_config({"patch_length": 0}, 1, 512, 96)

    # Test negative d_model
    with pytest.raises(ValueError, match="d_model debe ser >= 1"):
        create_patchtsmixer_config({"d_model": 0}, 1, 512, 96)

    # Test dropout out of range
    with pytest.raises(ValueError, match="dropout debe estar en"):
        create_patchtsmixer_config({"dropout": 1.5}, 1, 512, 96)


def test_model_initialization():
    """Test model initialization returns correct type."""
    from transformers import PatchTSMixerForPrediction

    config = create_patchtsmixer_config({}, 1, 512, 96)
    model = build_patchtsmixer_model(config)

    # Verify instance type
    assert isinstance(model, PatchTSMixerForPrediction)

    # Verify config matches
    assert model.config.context_length == 512
    assert model.config.prediction_length == 96
    assert model.config.num_input_channels == 1


def test_model_device_placement_cpu():
    """Test that model is placed on CPU device."""
    config = create_patchtsmixer_config({}, 1, 512, 96)
    model = build_patchtsmixer_model(config)

    # Verify device
    device = next(model.parameters()).device
    assert str(device) == 'cpu' or device.type == 'cpu'

    # Verify all parameters are on CPU
    for param in model.parameters():
        assert param.device.type == 'cpu'


def test_model_parameter_count():
    """Test model has reasonable number of parameters."""
    config = create_patchtsmixer_config({}, 3, 512, 96)
    model = build_patchtsmixer_model(config)

    num_params = sum(p.numel() for p in model.parameters())

    # Medium model should have reasonable param count
    # (not too small, not too large)
    assert 10_000 < num_params < 10_000_000


def test_config_serialization_json():
    """Test that config can be serialized to JSON."""
    import json

    config = create_patchtsmixer_config({}, 1, 512, 96)

    # Convert to dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)

    # Verify JSON-serializable
    json_str = json.dumps(config_dict)
    assert len(json_str) > 0

    # Verify can deserialize
    restored = json.loads(json_str)
    assert restored["context_length"] == 512
    assert restored["prediction_length"] == 96


def test_preset_small():
    """Test small preset configuration."""
    params = get_patchtsmixer_preset("small")

    assert params["d_model"] == 16
    assert params["num_layers"] == 6
    assert params["patch_length"] == 16
    assert params["dropout"] == 0.2
    assert params["expansion_factor"] == 2


def test_preset_medium():
    """Test medium preset configuration."""
    params = get_patchtsmixer_preset("medium")

    assert params["d_model"] == 32
    assert params["num_layers"] == 8
    assert params["patch_length"] == 8


def test_preset_large():
    """Test large preset configuration."""
    params = get_patchtsmixer_preset("large")

    assert params["d_model"] == 64
    assert params["num_layers"] == 12
    assert params["patch_length"] == 8


def test_preset_invalid_name():
    """Test invalid preset name raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        get_patchtsmixer_preset("extra_large")

    # Verify error message lists available presets
    assert "Preset inválido" in str(exc_info.value)
    assert "small" in str(exc_info.value)
    assert "medium" in str(exc_info.value)
    assert "large" in str(exc_info.value)


def test_preset_returns_copy():
    """Test that preset returns a copy, not reference."""
    params1 = get_patchtsmixer_preset("medium")
    params2 = get_patchtsmixer_preset("medium")

    # Modify one
    params1["d_model"] = 999

    # Verify other is unchanged
    assert params2["d_model"] == 32
