from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from typing import Optional

def create_lag_features(df: pd.DataFrame, input_features: list, lag_periods: int, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create lag features for input_features only (not for target variables).

    Parameters:
    - df: Input DataFrame
    - input_features: List of columns to create lag features for (input features only)
    - lag_periods: Number of lag periods to create (1 to lag_periods)
    - date_column: Reference date column (if None, assumes first column)

    Returns:
    - DataFrame with lag features added
    """
    if lag_periods <= 0:
        return df

    df_with_lags = df.copy()

    # Use first column as date reference if not specified
    if date_column is None:
        date_column = df.columns[0]

    # Create lag features only for input_features
    for feature in input_features:
        if feature in df.columns and feature != date_column:
            for lag in range(1, lag_periods + 1):
                lag_column_name = f"{feature}_lag_{lag}"
                df_with_lags[lag_column_name] = df_with_lags[feature].shift(lag)

    return df_with_lags

def handle_lag_nans(df: pd.DataFrame, nan_handling: str) -> pd.DataFrame:
    """
    Handle NaN values created by lag features.

    Parameters:
    - df: DataFrame with potential NaN values from lagging
    - nan_handling: Strategy to handle NaNs
                   "drop" - Remove rows with any NaN values
                   "forward_fill" - Forward fill NaN values
                   "leave_as_is" - Keep NaN values unchanged

    Returns:
    - DataFrame with NaNs handled according to strategy
    """
    if nan_handling == "drop":
        return df.dropna()
    elif nan_handling == "forward_fill":
        return df.ffill()
    elif nan_handling == "leave_as_is":
        return df
    else:
        raise ValueError(f"Invalid nan_handling option: {nan_handling}. Use 'drop', 'forward_fill', or 'leave_as_is'")

# LEGACY/AUXILIARY FUNCTIONS - Kept intact but not used in main flow
def encode_input_features(df:pd.DataFrame, input_features: list) -> pd.DataFrame:
    # 1. Codificación de variables de entrada (features)
    for col in input_features:
        if col in df.select_dtypes(include=[object]).columns:
            # Fix: Deterministic one-hot encoding for reproducibility
            # Sort categories alphabetically to ensure consistent column ordering
            # Without sorting, column order varies across different pandas/Python versions
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            dummies = dummies.reindex(sorted(dummies.columns), axis=1)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])
    return df

def apply_ohe_or_labelencoder_to_target(df:pd.DataFrame, 
                                        target_variables:list,
                                        apply_ohe_to_target:bool,
                                        apply_labelencoder_to_target:bool
) -> pd.DataFrame:
    # 2. Codificación de variables de salida (targets)
    if apply_ohe_to_target or apply_labelencoder_to_target:
        for col in target_variables:
            if col in df.select_dtypes(include=[object, 'category', 'int64', 'int32', 'int16', 'int8']).columns:
                if apply_ohe_to_target:
                    # Codificación OneHot
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=encoder.get_feature_names_out([col]),
                        index=df.index
                    )
                    df = pd.concat([df, encoded_df], axis=1).drop(columns=[col])
                elif apply_labelencoder_to_target:
                    # Codificación LabelEncoder
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
    return df

def encode_boolean_features(df: pd.DataFrame) ->pd.DataFrame:
    # 3. Conversión de booleanos a enteros
    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)
    return df

def encode_data(csv_input: str,
                    csv_output_train: str,
                    input_features: list,
                    target_variables: list,
                    apply_ohe_to_target: bool = False,
                    apply_labelencoder_to_target: bool = False,
                    lag_periods: int = 0,
                    lag_nan_handling: str = "leave_as_is",
                    date_column: Optional[str] = None):
    """
    Create lag features for time series data.

    NEW BEHAVIOR: Creates lag features for input_features only (not target variables).
    Legacy categorical encoding parameters are kept for compatibility but not used.

    Parámetros:
    input_features: List of columns to create lag features for
    target_variables: Target columns (no lag features applied)
    lag_periods: Number of lag periods to create (if 0, no lag features)
    lag_nan_handling: How to handle NaNs - "drop", "forward_fill", or "leave_as_is"
    date_column: Reference date column (if None, uses first column)

    LEGACY PARAMETERS (unused):
    apply_ohe_to_target: Si es True, aplica OneHotEncoder a targets
    apply_labelencoder_to_target: Si es True, aplica LabelEncoder a targets
    """
    print("Cargando el archivo limpio para codificación...")
    df = pd.read_csv(csv_input)

    # Validar parámetros excluyentes
    if apply_ohe_to_target and apply_labelencoder_to_target:
        raise ValueError("No se puede usar OHE y LabelEncoder simultáneamente en targets")

    # Filtrar columnas especiales
    input_features = [col for col in input_features if not col.endswith("_vacio")]
    target_variables = [col for col in target_variables if not col.endswith("_vacio")]

    # NEW BEHAVIOR: Create lag features for input_features only
    if lag_periods > 0:
        print(f"Creating lag features for {lag_periods} periods...")
        df = create_lag_features(df, input_features, lag_periods, date_column)
        print(f"Handling NaN values with strategy: {lag_nan_handling}")
        df = handle_lag_nans(df, lag_nan_handling)
    else:
        print("No lag features requested (lag_periods = 0)")

    # LEGACY ENCODING FUNCTIONS - Not used in current flow but kept intact
    # df = encode_input_features(df, input_features)
    # df = apply_ohe_or_labelencoder_to_target(df, target_variables, apply_ohe_to_target, apply_labelencoder_to_target)
    # df = encode_boolean_features(df)

    # Guardado de resultados
    df.to_csv(csv_output_train, index=False)
    print(f"Dataset codificado guardado en: {csv_output_train}")
    return df