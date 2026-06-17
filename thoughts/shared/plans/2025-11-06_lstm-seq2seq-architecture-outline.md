# LSTM Sequence-to-Sequence Architecture Outline

**Implementation Plan Reference:** [2025-11-06_lstm-training-implementation.md](./2025-11-06_lstm-training-implementation.md)
**Status:** 📋 Architectural Outline (Implementation Deferred)
**Last Updated:** 2025-11-06
**Phase:** Phase 5 (Future Implementation)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture Overview](#architecture-overview)
- [Encoder-Decoder Design](#encoder-decoder-design)
- [Multi-Step Output Generation](#multi-step-output-generation)
- [Attention Mechanism](#attention-mechanism)
- [New Hyperparameters](#new-hyperparameters)
- [Code Structure](#code-structure)
- [UI Changes](#ui-changes)
- [Backward Compatibility](#backward-compatibility)
- [Migration Path](#migration-path)
- [Implementation Checklist](#implementation-checklist)

---

## Executive Summary

### Objective

Extend the single-step LSTM implementation (Phase 1-4) to support multi-step forecasting using an encoder-decoder (sequence-to-sequence) architecture. This enables predicting multiple future timesteps simultaneously (e.g., next 7 days, next 30 days) instead of only the next single timestep.

### Current State (After Phase 1-4)

**Single-Step LSTM:**
```
Input:  [t-9, t-8, t-7, ..., t-1, t0]  →  LSTM  →  Output: [t+1]
        (sequence_length = 10)                          (forecast_horizon = 1)
```

- Predicts only the immediate next timestep
- Output shape: `(batch_size, 1)`
- Simple Dense output layer
- Works well for short-term forecasting

### Future State (Phase 5)

**Sequence-to-Sequence LSTM:**
```
Input:  [t-9, t-8, ..., t0]  →  Encoder  →  Context Vector
                                      ↓
                                  Decoder  →  Output: [t+1, t+2, ..., t+H]
                                                    (forecast_horizon = H)
```

- Predicts H future timesteps simultaneously
- Output shape: `(batch_size, forecast_horizon)`
- Encoder-decoder architecture
- Optional attention mechanism
- Better for medium/long-term forecasting

### Key Benefits

1. **Multi-Step Forecasting:** Predict next H timesteps in one forward pass
2. **Better Context:** Encoder compresses input sequence into rich context vector
3. **Flexible Horizons:** User-configurable forecast_horizon (1, 7, 30, etc.)
4. **Attention (Optional):** Decoder can "attend" to relevant input timesteps
5. **Backward Compatible:** Single-step models continue to work

---

## Architecture Overview

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SEQUENCE                               │
│  [t-9, t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t0]             │
│  Shape: (batch_size, sequence_length, n_features)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODER LSTM                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ LSTM     │→ │ LSTM     │→ │ LSTM     │                     │
│  │ Layer 1  │  │ Layer 2  │  │ Layer 3  │  (Stacked)          │
│  └──────────┘  └──────────┘  └──────────┘                     │
│                                     │                           │
│                                     ↓                           │
│                        [Hidden State, Cell State]               │
│                        (Context Vector)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DECODER LSTM                                 │
│  Initial State = Encoder's Final State                         │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ LSTM     │→ │ LSTM     │→ │ LSTM     │  (Stacked)          │
│  │ Layer 1  │  │ Layer 2  │  │ Layer 3  │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
│       │              │              │                           │
│       ↓              ↓              ↓                           │
│    [t+1]         [t+2]         [t+3]  ... [t+H]                │
│                                                                 │
│  Optional: Attention Layer (attends to encoder outputs)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT SEQUENCE                              │
│  [y_t+1, y_t+2, y_t+3, ..., y_t+H]                             │
│  Shape: (batch_size, forecast_horizon)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

**1. Encoder**
- Processes input sequence [t-9, ..., t0]
- Stacked LSTM layers (configurable: 1-3 layers)
- Outputs final hidden state and cell state
- These states contain compressed representation of input sequence

**2. Context Vector**
- Encoder's final (hidden_state, cell_state) tuple
- Dimensionality: encoder_units (e.g., 64, 128)
- Passed to decoder as initial state

**3. Decoder**
- Generates output sequence [t+1, ..., t+H]
- Stacked LSTM layers (configurable: 1-3 layers)
- Initialized with encoder's context vector
- Uses teacher forcing during training (optional)
- Can attend to encoder outputs (optional)

**4. Output Layer**
- Dense layer with forecast_horizon units
- Linear activation (regression)
- Or: TimeDistributed Dense for step-by-step output

---

## Encoder-Decoder Design

### Encoder Architecture

**Function:** Compress input sequence into context vector

**Layer Structure:**
```python
def build_encoder(encoder_units: List[int], input_shape: Tuple[int, int]) -> Tuple[keras.Model, keras.layers.Layer]:
    """
    Construye encoder LSTM para seq2seq.

    Args:
        encoder_units: Lista de unidades por capa (e.g., [128, 64])
        input_shape: (sequence_length, n_features)

    Returns:
        encoder_model: Modelo Keras del encoder
        encoder_outputs: Capa que produce outputs para atención (opcional)
    """
    # Input layer
    encoder_inputs = Input(shape=input_shape, name="encoder_input")

    # Stacked LSTM layers
    encoder_lstm_layers = encoder_inputs
    for i, units in enumerate(encoder_units):
        encoder_lstm_layers = LSTM(
            units=units,
            return_sequences=(i < len(encoder_units) - 1),  # True for all but last
            return_state=True if (i == len(encoder_units) - 1) else False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name=f"encoder_lstm_{i+1}"
        )(encoder_lstm_layers)

    # Last layer returns sequences + states
    if len(encoder_units) == 1:
        encoder_outputs, state_h, state_c = LSTM(
            units=encoder_units[0],
            return_sequences=True,  # For attention
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name="encoder_lstm"
        )(encoder_inputs)
    else:
        # Multi-layer: extract from last layer
        # ... (complex stacking logic)
        pass

    encoder_states = [state_h, state_c]

    encoder_model = Model(
        inputs=encoder_inputs,
        outputs=[encoder_outputs, state_h, state_c],
        name="encoder"
    )

    return encoder_model, encoder_outputs
```

**Key Features:**
- `return_sequences=True` on all layers except last (for stacking)
- Last layer returns both sequences (for attention) and states (for context)
- Dropout and recurrent dropout for regularization
- Configurable depth (1-3 layers typically)

---

### Decoder Architecture

**Function:** Generate multi-step output sequence

**Two Decoder Strategies:**

#### Strategy A: Direct Multi-Output (Simpler)

```python
def build_decoder_direct(
    decoder_units: List[int],
    encoder_units: int,
    forecast_horizon: int
) -> keras.Model:
    """
    Decoder directo: genera todos los pasos en una salida.

    Arquitectura:
        Encoder States → LSTM Layers → Dense(forecast_horizon)

    Pros: Más simple, rápido
    Cons: No usa teacher forcing, menos flexible
    """
    # Inputs: encoder states
    decoder_state_input_h = Input(shape=(encoder_units,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(encoder_units,), name="decoder_state_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Decoder LSTM (uses encoder states as initial state)
    decoder_lstm_output = LSTM(
        units=decoder_units[0],
        return_sequences=False,  # Single output per sequence
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        name="decoder_lstm"
    )(
        inputs=RepeatVector(1)(decoder_state_input_h),  # Dummy input
        initial_state=decoder_states_inputs
    )

    # Output layer: forecast_horizon timesteps
    decoder_outputs = Dense(
        units=forecast_horizon,
        activation=None,  # Linear for regression
        name="decoder_output"
    )(decoder_lstm_output)

    decoder_model = Model(
        inputs=decoder_states_inputs,
        outputs=decoder_outputs,
        name="decoder_direct"
    )

    return decoder_model
```

**Output Shape:** `(batch_size, forecast_horizon)`

**Use Case:** When forecast_horizon is small (≤30) and teacher forcing not needed

---

#### Strategy B: Autoregressive Decoder (More Flexible)

```python
def build_decoder_autoregressive(
    decoder_units: List[int],
    encoder_units: int,
    forecast_horizon: int,
    use_attention: bool = False
) -> keras.Model:
    """
    Decoder autoregresivo: genera un paso a la vez.

    Arquitectura:
        Encoder States → LSTM → Dense(1) → Feed back as input → Loop

    Pros: Teacher forcing, atención, más preciso
    Cons: Más complejo, más lento
    """
    # Encoder outputs for attention
    encoder_outputs_input = Input(
        shape=(None, encoder_units),  # (sequence_length, encoder_units)
        name="encoder_outputs"
    )

    # Decoder input (previous prediction, starts with encoder's last output)
    decoder_input = Input(shape=(1,), name="decoder_input")

    # Decoder states
    decoder_state_input_h = Input(shape=(decoder_units[0],))
    decoder_state_input_c = Input(shape=(decoder_units[0],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Attention layer (optional)
    if use_attention:
        attention = Attention(name="attention_layer")
        # Compute attention weights
        context_vector = attention([decoder_input, encoder_outputs_input])
        # Concatenate context with decoder input
        decoder_input_with_context = Concatenate()([decoder_input, context_vector])
    else:
        decoder_input_with_context = decoder_input

    # Decoder LSTM
    decoder_outputs, state_h, state_c = LSTM(
        units=decoder_units[0],
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        name="decoder_lstm"
    )(decoder_input_with_context, initial_state=decoder_states_inputs)

    # Output layer (single timestep)
    decoder_output = TimeDistributed(
        Dense(1, activation=None),
        name="decoder_output_step"
    )(decoder_outputs)

    decoder_model = Model(
        inputs=[decoder_input, encoder_outputs_input] + decoder_states_inputs,
        outputs=[decoder_output, state_h, state_c],
        name="decoder_autoregressive"
    )

    return decoder_model
```

**Output Shape:** `(batch_size, 1)` per step, looped forecast_horizon times

**Use Case:** When forecast_horizon is large (>30) or attention is needed

---

## Multi-Step Output Generation

### Training Phase

**With Teacher Forcing:**
```python
def train_seq2seq_with_teacher_forcing(
    encoder_model,
    decoder_model,
    X_train,
    y_train_sequences,  # Shape: (n_samples, forecast_horizon)
    epochs,
    batch_size
):
    """
    Entrena seq2seq usando teacher forcing.

    Durante entrenamiento:
    - Decoder recibe valores reales (y_t+1) como input para predecir (y_t+2)
    - Esto acelera convergencia y previene error accumulation
    """
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_sequences[i:i+batch_size]

            # Encode
            encoder_outputs, state_h, state_c = encoder_model.predict(X_batch)

            # Decode with teacher forcing
            decoder_states = [state_h, state_c]
            decoder_input = y_batch[:, 0].reshape(-1, 1)  # First target value

            predictions = []
            for t in range(forecast_horizon):
                # Predict next step
                decoder_output, state_h, state_c = decoder_model.predict(
                    [decoder_input, encoder_outputs] + decoder_states
                )
                predictions.append(decoder_output)

                # TEACHER FORCING: Use actual value as next input (not prediction)
                if t < forecast_horizon - 1:
                    decoder_input = y_batch[:, t+1].reshape(-1, 1)

                # Update states
                decoder_states = [state_h, state_c]

            # Compute loss and backpropagate
            # ... (standard training loop)
```

---

### Inference Phase

**Autoregressive Prediction:**
```python
def predict_seq2seq(encoder_model, decoder_model, X_test, forecast_horizon):
    """
    Genera predicciones multi-paso sin teacher forcing.

    Durante inferencia:
    - Decoder usa sus propias predicciones como input para siguiente paso
    - Error puede acumularse a lo largo del horizonte
    """
    # Encode
    encoder_outputs, state_h, state_c = encoder_model.predict(X_test)

    # Initialize decoder
    decoder_states = [state_h, state_c]
    decoder_input = np.zeros((len(X_test), 1))  # Or use last encoder output

    predictions = []
    for t in range(forecast_horizon):
        # Predict next step
        decoder_output, state_h, state_c = decoder_model.predict(
            [decoder_input, encoder_outputs] + decoder_states
        )
        predictions.append(decoder_output)

        # AUTOREGRESSIVE: Use prediction as next input
        decoder_input = decoder_output

        # Update states
        decoder_states = [state_h, state_c]

    # Stack predictions
    predictions = np.concatenate(predictions, axis=1)  # Shape: (n_samples, forecast_horizon)

    return predictions
```

---

## Attention Mechanism

### Why Attention?

**Problem:** Long input sequences cause encoder to "forget" early timesteps when compressing into fixed-size context vector.

**Solution:** Allow decoder to "attend" to all encoder outputs, not just final state.

### Attention Architecture

```
Encoder Outputs: [h1, h2, h3, ..., h_T]  (all timesteps)
                       ↓
              Attention Mechanism
                       ↓
              Context Vector (weighted sum of encoder outputs)
                       ↓
                   Decoder
```

### Implementation Sketch

```python
class AttentionLayer(keras.layers.Layer):
    """
    Capa de atención para seq2seq.

    Calcula pesos de atención sobre encoder outputs y genera context vector.
    """

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W_a = Dense(units, use_bias=False)
        self.U_a = Dense(units, use_bias=False)
        self.V_a = Dense(1, use_bias=False)

    def call(self, decoder_hidden_state, encoder_outputs):
        """
        Args:
            decoder_hidden_state: Shape (batch, decoder_units)
            encoder_outputs: Shape (batch, seq_len, encoder_units)

        Returns:
            context_vector: Shape (batch, encoder_units)
            attention_weights: Shape (batch, seq_len)
        """
        # Expand decoder state to match encoder outputs
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
        # Shape: (batch, 1, decoder_units)

        # Compute attention scores
        score = self.V_a(tf.nn.tanh(
            self.W_a(encoder_outputs) + self.U_a(decoder_hidden_state)
        ))
        # Shape: (batch, seq_len, 1)

        # Compute attention weights (softmax)
        attention_weights = tf.nn.softmax(score, axis=1)
        # Shape: (batch, seq_len, 1)

        # Compute context vector (weighted sum)
        context_vector = tf.reduce_sum(
            attention_weights * encoder_outputs,
            axis=1
        )
        # Shape: (batch, encoder_units)

        return context_vector, tf.squeeze(attention_weights, -1)
```

### When to Use Attention?

- ✅ **Use Attention When:**
  - Input sequence_length > 50
  - Long-term dependencies critical
  - Variable-length sequences
  - Model interpretability needed (visualize attention weights)

- ❌ **Skip Attention When:**
  - Short sequences (< 20 timesteps)
  - Computational constraints (attention adds overhead)
  - Simple forecasting tasks

---

## New Hyperparameters

### Additional Parameters for Seq2Seq

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_units` | List[int] | `[64]` | LSTM units per encoder layer (e.g., [128, 64]) |
| `decoder_units` | List[int] | `[64]` | LSTM units per decoder layer |
| `forecast_horizon` | int | `1` | Number of future timesteps to predict |
| `decoder_strategy` | str | `"direct"` | "direct" or "autoregressive" |
| `use_attention` | bool | `False` | Enable attention mechanism |
| `teacher_forcing_ratio` | float | `1.0` | Ratio of teacher forcing during training (0.0-1.0) |
| `attention_units` | int | `64` | Units in attention layer (if enabled) |

### Relationship to Phase 1 Parameters

**Shared Parameters** (from Phase 1):
- `sequence_length`: Input window size
- `dropout_rate`: Dropout for all LSTM layers
- `recurrent_dropout_rate`: Recurrent dropout
- `learning_rate`: Optimizer learning rate
- `batch_size`: Training batch size
- `epochs`: Training epochs

**Modified Parameters:**
- `lstm_units` (Phase 1) → Split into `encoder_units` + `decoder_units` (Phase 5)

**Backward Compatibility:**
- If `forecast_horizon=1` and `decoder_strategy="direct"`, behaves like Phase 1 single-step model
- Frontend can show "Advanced: Multi-Step Forecasting" section (collapsed by default)

---

## Code Structure

### File Organization

```
DREAM-ML-backend/GEML/apiTimeSeries/
├── train.py
│   ├── create_sequences_for_seq2seq()  # New: creates (X, y_sequences) with shape (n, H)
│   ├── build_encoder_model()           # New
│   ├── build_decoder_model()           # New
│   ├── build_attention_layer()         # New (optional)
│   ├── build_seq2seq_model()           # New: combines encoder + decoder
│   ├── train_seq2seq_loop()            # New: custom training loop with teacher forcing
│   ├── evaluate_seq2seq_model()        # New: multi-step evaluation
│   └── train_lstm_model()              # Modified: add seq2seq path
```

### Sequence Creation Changes

**Current (Phase 1):**
```python
# Single-step: y is 1D
X, y = create_sequences_for_lstm(df, features, target, seq_len, horizon=1)
# X.shape: (n_sequences, seq_len, n_features)
# y.shape: (n_sequences,)
```

**Seq2Seq (Phase 5):**
```python
# Multi-step: y is 2D sequence
X, y_sequences = create_sequences_for_seq2seq(df, features, target, seq_len, horizon=H)
# X.shape: (n_sequences, seq_len, n_features)
# y_sequences.shape: (n_sequences, H)
```

**Implementation:**
```python
def create_sequences_for_seq2seq(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias para seq2seq con output multi-paso.

    Returns:
        X: Input sequences (n, seq_len, n_features)
        y_sequences: Target sequences (n, forecast_horizon)
    """
    features = df[feature_cols].values
    target = df[target_col].values

    X_sequences = []
    y_sequences = []

    # PSEUDOCODE: Sliding window with multi-step target
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        # Input: [t-seq_len, ..., t-1, t0]
        X_seq = features[i:i + sequence_length]

        # Output: [t+1, t+2, ..., t+H]
        y_seq = target[i + sequence_length:i + sequence_length + forecast_horizon]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    X = np.array(X_sequences)
    y = np.array(y_sequences)

    return X, y
```

---

### Model Building Function

```python
def build_seq2seq_model(
    encoder_units: List[int],
    decoder_units: List[int],
    input_shape: Tuple[int, int],
    forecast_horizon: int,
    dropout_rate: float = 0.2,
    recurrent_dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    use_attention: bool = False,
    decoder_strategy: str = "direct"
) -> keras.Model:
    """
    Construye modelo seq2seq completo.

    Args:
        encoder_units: Unidades LSTM del encoder
        decoder_units: Unidades LSTM del decoder
        input_shape: (sequence_length, n_features)
        forecast_horizon: Número de pasos a predecir
        dropout_rate: Dropout rate
        recurrent_dropout_rate: Recurrent dropout rate
        learning_rate: Learning rate para optimizer
        use_attention: Si True, usa atención
        decoder_strategy: "direct" o "autoregressive"

    Returns:
        Modelo Keras compilado
    """
    # Build encoder
    encoder_model, encoder_outputs = build_encoder(
        encoder_units=encoder_units,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate
    )

    # Build decoder
    if decoder_strategy == "direct":
        decoder_model = build_decoder_direct(
            decoder_units=decoder_units,
            encoder_units=encoder_units[-1],
            forecast_horizon=forecast_horizon,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate
        )
    elif decoder_strategy == "autoregressive":
        decoder_model = build_decoder_autoregressive(
            decoder_units=decoder_units,
            encoder_units=encoder_units[-1],
            forecast_horizon=forecast_horizon,
            use_attention=use_attention,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate
        )
    else:
        raise ValueError(f"decoder_strategy inválido: {decoder_strategy}")

    # Combine encoder + decoder into full model
    encoder_input = Input(shape=input_shape, name="seq2seq_input")
    encoder_outputs, state_h, state_c = encoder_model(encoder_input)

    if decoder_strategy == "direct":
        decoder_output = decoder_model([state_h, state_c])
    else:
        # Autoregressive: requires custom training loop
        # Return encoder and decoder separately
        return encoder_model, decoder_model

    # Full model (for direct strategy)
    seq2seq_model = Model(
        inputs=encoder_input,
        outputs=decoder_output,
        name="seq2seq_lstm"
    )

    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    seq2seq_model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"]
    )

    return seq2seq_model
```

---

## UI Changes

### Frontend Parameter Inputs

**New Section in TSTrainCard.jsx:**

```jsx
{/* Multi-Step Forecasting Section (Collapsed by default) */}
{algorithm === "lstm" && (
  <Accordion defaultExpanded={false}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
      <Typography>Advanced: Multi-Step Forecasting (Seq2Seq)</Typography>
    </AccordionSummary>
    <AccordionDetails>

      {/* Forecast Horizon */}
      <TextField
        fullWidth
        type="number"
        label="Forecast Horizon (Pasos Futuros)"
        value={forecastHorizon}
        onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 1)}
        helperText="Número de timesteps futuros a predecir (1 para single-step, 7 para semanal, 30 para mensual)"
        sx={{ mb: 2 }}
        InputProps={{ inputProps: { min: 1, max: 365 } }}
      />

      {/* Enable Seq2Seq */}
      <FormControlLabel
        control={
          <Switch
            checked={enableSeq2Seq}
            onChange={(e) => setEnableSeq2Seq(e.target.checked)}
          />
        }
        label="Enable Sequence-to-Sequence Architecture"
      />

      {enableSeq2Seq && (
        <>
          {/* Encoder Units */}
          <TextField
            fullWidth
            label="Encoder Units (e.g., [128,64])"
            value={encoderUnits}
            onChange={(e) => setEncoderUnits(e.target.value)}
            helperText="Lista de unidades LSTM para encoder"
            sx={{ mb: 2 }}
          />

          {/* Decoder Units */}
          <TextField
            fullWidth
            label="Decoder Units (e.g., [64])"
            value={decoderUnits}
            onChange={(e) => setDecoderUnits(e.target.value)}
            helperText="Lista de unidades LSTM para decoder"
            sx={{ mb: 2 }}
          />

          {/* Decoder Strategy */}
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Decoder Strategy</InputLabel>
            <Select
              value={decoderStrategy}
              onChange={(e) => setDecoderStrategy(e.target.value)}
            >
              <MenuItem value="direct">Direct (más rápido)</MenuItem>
              <MenuItem value="autoregressive">Autoregressive (más preciso)</MenuItem>
            </Select>
            <FormHelperText>
              Direct: genera todos los pasos simultáneamente.
              Autoregressive: genera paso a paso con feedback.
            </FormHelperText>
          </FormControl>

          {/* Use Attention */}
          {decoderStrategy === "autoregressive" && (
            <FormControlLabel
              control={
                <Switch
                  checked={useAttention}
                  onChange={(e) => setUseAttention(e.target.checked)}
                />
              }
              label="Use Attention Mechanism"
            />
          )}

          {/* Teacher Forcing Ratio */}
          {decoderStrategy === "autoregressive" && (
            <TextField
              fullWidth
              type="number"
              label="Teacher Forcing Ratio"
              value={teacherForcingRatio}
              onChange={(e) => setTeacherForcingRatio(parseFloat(e.target.value) || 1.0)}
              helperText="1.0 = siempre usa valores reales, 0.0 = siempre usa predicciones"
              sx={{ mb: 2 }}
              InputProps={{ inputProps: { min: 0.0, max: 1.0, step: 0.1 } }}
            />
          )}
        </>
      )}

    </AccordionDetails>
  </Accordion>
)}
```

### State Variables

```jsx
const [forecastHorizon, setForecastHorizon] = useState(1);
const [enableSeq2Seq, setEnableSeq2Seq] = useState(false);
const [encoderUnits, setEncoderUnits] = useState("[64]");
const [decoderUnits, setDecoderUnits] = useState("[64]");
const [decoderStrategy, setDecoderStrategy] = useState("direct");
const [useAttention, setUseAttention] = useState(false);
const [teacherForcingRatio, setTeacherForcingRatio] = useState(1.0);
```

---

## Backward Compatibility

### Ensuring Phase 1-4 Models Continue Working

**Strategy 1: Automatic Detection**
```python
# In train_lstm_model()
forecast_horizon = data.get("forecast_horizon", 1)
enable_seq2seq = data.get("enable_seq2seq", False)

if forecast_horizon == 1 or not enable_seq2seq:
    # Use Phase 1 single-step LSTM
    model = build_lstm_model(params, input_shape)
    # ... (existing Phase 1 code)
else:
    # Use Phase 5 seq2seq
    model = build_seq2seq_model(
        encoder_units=params.get("encoder_units", params["lstm_units"]),
        decoder_units=params.get("decoder_units", params["lstm_units"]),
        input_shape=input_shape,
        forecast_horizon=forecast_horizon,
        ...
    )
```

**Strategy 2: Model Versioning**
```python
# Add model_version to MLflow metadata
mlflow.log_param("model_version", "lstm_v1")  # Phase 1-4
mlflow.log_param("model_version", "seq2seq_v1")  # Phase 5

# When loading model for inference
model_version = mlflow_model.metadata.get("model_version")
if model_version == "lstm_v1":
    # Use single-step prediction logic
    predictions = model.predict(X_test).flatten()
elif model_version == "seq2seq_v1":
    # Use multi-step prediction logic
    predictions = predict_seq2seq(encoder, decoder, X_test, horizon)
```

---

## Migration Path

### From Phase 1 Single-Step to Phase 5 Multi-Step

**User Workflow:**

1. **Existing Users (Phase 1-4):**
   - No changes required
   - `forecast_horizon=1` by default (backward compatible)
   - UI doesn't show seq2seq section by default (collapsed)

2. **New Users (Want Multi-Step):**
   - Select LSTM algorithm
   - Expand "Advanced: Multi-Step Forecasting" section
   - Set `forecast_horizon` (e.g., 7 for weekly)
   - Enable "Sequence-to-Sequence Architecture" toggle
   - Configure encoder/decoder architecture
   - Train model

3. **Migration:**
   - Existing single-step models remain in MLflow (tagged `lstm_v1`)
   - New multi-step models tagged `seq2seq_v1`
   - Both accessible via MLflow UI
   - Inference code auto-detects version

### Training Data Changes

**Phase 1:**
```python
# Creates (n, seq_len, features) → (n,) mapping
X, y = create_sequences_for_lstm(df, features, target, seq_len, horizon=1)
```

**Phase 5:**
```python
# Creates (n, seq_len, features) → (n, horizon) mapping
X, y_seq = create_sequences_for_seq2seq(df, features, target, seq_len, horizon=H)
```

**Evaluation Metrics:**
- Single-step: RMSE, MAE, MAPE (as before)
- Multi-step: RMSE per step, average RMSE, horizon-weighted RMSE

---

## Implementation Checklist

### Backend Tasks

- [ ] **Sequence Creation**
  - [ ] Implement `create_sequences_for_seq2seq()`
  - [ ] Add unit tests for multi-step sequences
  - [ ] Validate shape: `(n, forecast_horizon)`

- [ ] **Encoder**
  - [ ] Implement `build_encoder_model()`
  - [ ] Support 1-3 stacked LSTM layers
  - [ ] Return encoder outputs + states

- [ ] **Decoder (Direct)**
  - [ ] Implement `build_decoder_direct()`
  - [ ] Dense output layer with `forecast_horizon` units
  - [ ] Test with small horizons (H=7)

- [ ] **Decoder (Autoregressive)**
  - [ ] Implement `build_decoder_autoregressive()`
  - [ ] Custom training loop with teacher forcing
  - [ ] Autoregressive inference loop

- [ ] **Attention (Optional)**
  - [ ] Implement `AttentionLayer` class
  - [ ] Integrate with autoregressive decoder
  - [ ] Visualize attention weights

- [ ] **Training**
  - [ ] Modify `train_lstm_model()` to support seq2seq
  - [ ] Add `forecast_horizon` parameter handling
  - [ ] Implement teacher forcing logic
  - [ ] Track multi-step metrics

- [ ] **Evaluation**
  - [ ] Implement `evaluate_seq2seq_model()`
  - [ ] Per-step RMSE/MAE calculation
  - [ ] Multi-step forecast plots
  - [ ] Horizon error analysis

- [ ] **MLflow**
  - [ ] Log seq2seq-specific parameters
  - [ ] Save encoder and decoder separately
  - [ ] Model versioning (`model_version` tag)

- [ ] **Backward Compatibility**
  - [ ] Detect single-step vs multi-step
  - [ ] Auto-route to appropriate model builder
  - [ ] Test existing Phase 1 models still work

### Frontend Tasks

- [ ] **UI Components**
  - [ ] Add "Forecast Horizon" TextField
  - [ ] Add "Enable Seq2Seq" toggle
  - [ ] Add "Encoder Units" input
  - [ ] Add "Decoder Units" input
  - [ ] Add "Decoder Strategy" selector
  - [ ] Add "Use Attention" toggle
  - [ ] Add "Teacher Forcing Ratio" slider

- [ ] **State Management**
  - [ ] Add state variables for all new params
  - [ ] Validate encoder/decoder units format
  - [ ] Parse lists from strings (e.g., "[128,64]")

- [ ] **Payload Construction**
  - [ ] Include `forecast_horizon` in training payload
  - [ ] Include `enable_seq2seq` flag
  - [ ] Include encoder/decoder params

- [ ] **Results Display**
  - [ ] Show multi-step forecast plot
  - [ ] Display per-step metrics table
  - [ ] Horizon error visualization

### Testing Tasks

- [ ] **Unit Tests**
  - [ ] Test sequence creation (multi-step)
  - [ ] Test encoder model building
  - [ ] Test decoder models (both strategies)
  - [ ] Test attention layer

- [ ] **Integration Tests**
  - [ ] End-to-end seq2seq training
  - [ ] Backward compatibility (Phase 1 still works)
  - [ ] MLflow logging verification

- [ ] **Performance Tests**
  - [ ] Memory usage with long horizons
  - [ ] Training time comparison (direct vs autoregressive)
  - [ ] Inference speed benchmarks

### Documentation Tasks

- [ ] **Code Documentation**
  - [ ] Docstrings for all new functions
  - [ ] Architecture diagram comments
  - [ ] Parameter descriptions

- [ ] **User Guide**
  - [ ] When to use single-step vs multi-step
  - [ ] Encoder/decoder architecture guidelines
  - [ ] Attention mechanism explanation
  - [ ] Example use cases

- [ ] **API Documentation**
  - [ ] New parameters reference
  - [ ] Return format changes
  - [ ] Model versioning guide

---

## Summary

This outline provides a comprehensive roadmap for extending the LSTM implementation to support multi-step forecasting via sequence-to-sequence architecture. The design prioritizes:

1. **Backward Compatibility:** Phase 1-4 models continue working unchanged
2. **Flexibility:** Direct and autoregressive decoder strategies
3. **Optional Complexity:** Attention mechanism available but not required
4. **User Control:** Configurable forecast horizon and architecture
5. **Clear Migration:** Existing users unaffected, new users opt-in

**Key Architectural Decisions:**

- **Encoder-Decoder Split:** Clear separation enables flexibility and attention
- **Two Decoder Strategies:** Direct (fast, simple) vs Autoregressive (accurate, complex)
- **Teacher Forcing:** Improves training stability for autoregressive decoder
- **Attention Optional:** Available for long sequences, not required for short
- **Model Versioning:** `lstm_v1` vs `seq2seq_v1` for backward compatibility

**Implementation Priority:**

1. **Phase 5.1:** Direct decoder (simpler, good for H ≤ 30)
2. **Phase 5.2:** Autoregressive decoder with teacher forcing
3. **Phase 5.3:** Attention mechanism (optional enhancement)

**Estimated Implementation Time:** 12-16 hours (when implemented)

---

**Next Steps:**
1. Review this outline with team
2. Validate architectural decisions
3. Prioritize direct vs autoregressive decoder
4. Begin implementation when Phase 1-4 stable

For questions or clarifications, refer to main implementation plan: [2025-11-06_lstm-training-implementation.md](./2025-11-06_lstm-training-implementation.md)
