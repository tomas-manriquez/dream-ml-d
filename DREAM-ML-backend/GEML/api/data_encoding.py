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

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

def codificar_datos(csv_input: str,
                    csv_output_train: str,
                    input_features: list,
                    target_variables: list,
                    apply_ohe_to_target: bool = False,
                    apply_labelencoder_to_target: bool = False):
    """
    Codifica variables categóricas usando:
    - get_dummies para features de entrada
    - OneHotEncoder o LabelEncoder para targets (mutuamente excluyentes)
    
    Parámetros:
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

    # 1. Codificación de variables de entrada (features)
    for col in input_features:
        if col in df.select_dtypes(include=[object]).columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])

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

    # 3. Conversión de booleanos a enteros
    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)

    # Guardado de resultados
    df.to_csv(csv_output_train, index=False)
    print(f"Dataset codificado guardado en: {csv_output_train}")
    return df



