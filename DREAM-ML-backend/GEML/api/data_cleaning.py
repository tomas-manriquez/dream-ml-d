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

import pandas as pd
import numpy as np

def limpiar_datos(csv_input: str, csv_output_eda: str, eliminar_duplicados=True, 
                  filtrar_outliers=True, relleno_valores_numericos="media", 
                  valor_imputacion=None) -> dict:
    # Diccionario para almacenar el reporte de limpieza
    report = {}
    
    # Cargar el CSV y guardar información inicial
    df = pd.read_csv(csv_input)
    report["initial_rows"] = int(df.shape[0])
    report["initial_columns"] = list(df.columns)
    
    # Limpieza de nombres de columnas
    df.columns = df.columns.str.strip()
    
    # 1. Eliminar espacios vacíos en todas las celdas
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # 2. Reemplazar valores vacíos por NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # 3. Convertir columnas numéricas mal interpretadas
    converted_cols = []
    for col in df.columns:
        if df[col].dropna().apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted_cols.append(col)
    report["converted_to_numeric"] = converted_cols
    
    # 4. Eliminar duplicados (opcional)
    if eliminar_duplicados:
        before_dup = df.shape[0]
        df = df.drop_duplicates()
        after_dup = df.shape[0]
        report["duplicates_removed"] = int(before_dup - after_dup)
    else:
        report["duplicates_removed"] = 0
    
    # 5. Rellenar valores faltantes en columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_missing_before"] = df[numeric_cols].isna().sum().to_dict()
    
    if relleno_valores_numericos == "eliminar":
        before_drop = df.shape[0]
        df = df.dropna(subset=numeric_cols)
        after_drop = df.shape[0]
        report["numeric_rows_dropped_due_to_na"] = int(before_drop - after_drop)
    elif relleno_valores_numericos == "dejar":
        # No se realiza cambio en los NaN
        report["numeric_missing_after"] = df[numeric_cols].isna().sum().to_dict()
    else:
        imputations = {}
        for col in numeric_cols:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                if relleno_valores_numericos == "media":
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    imputations[col] = {"filled_with": "mean", "missing_count": missing_count, "fill_value": fill_value}
                elif relleno_valores_numericos == "valor" and valor_imputacion is not None:
                    df[col] = df[col].fillna(valor_imputacion)
                    imputations[col] = {"filled_with": valor_imputacion, "missing_count": missing_count}
        report["numeric_imputations"] = imputations
    
    # 6. Rellenar valores faltantes en columnas categóricas con 'vacio'
    categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
    cat_missing = {}
    for col in categorical_cols:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            df[col] = df[col].fillna('vacio')
            cat_missing[col] = missing_count
    report["categorical_missing_filled"] = cat_missing
    
    # 7. Filtrar valores atípicos (opcional)
    outliers_removed = {}
    if filtrar_outliers:
        for col in numeric_cols:
            if df[col].notna().sum() > 1:
                before_filter = df.shape[0]
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_filter = df.shape[0]
                outliers_removed[col] = int(before_filter - after_filter)
    report["outliers_removed"] = outliers_removed
    
    # 8. Eliminar columnas con solo valores nulos
    cols_before = list(df.columns)
    df = df.dropna(axis=1, how='all')
    cols_after = list(df.columns)
    removed_columns = list(set(cols_before) - set(cols_after))
    report["columns_removed_all_na"] = removed_columns
    
    # Guardar dataset limpio
    df.to_csv(csv_output_eda, index=False)
    report["final_rows"] = int(df.shape[0])
    report["final_columns"] = list(df.columns)
    
    print(f"El dataset limpio fue guardado en: {csv_output_eda}")
    return report




