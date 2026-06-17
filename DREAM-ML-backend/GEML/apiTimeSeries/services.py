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

# ─────────────────────────────────────────────────────────────────────────────
# Librerias estandar
# ─────────────────────────────────────────────────────────────────────────────
import json
import logging
import os
import subprocess
import uuid
import time
from datetime import datetime
from typing import Union, Optional
from typing import Dict 

# ─────────────────────────────────────────────────────────────────────────────
# Librerias de terceros
# ─────────────────────────────────────────────────────────────────────────────
import mlflow
from mlflow.tracking import MlflowClient
import psutil
import pandas as pd
from mlflow import (
    get_experiment_by_name, log_artifact, log_metric, log_param, set_experiment,
    set_tracking_uri, start_run
)
from ydata_profiling import ProfileReport
# ─────────────────────────────────────────────────────────────────────────────
# Importaciones locales
# ─────────────────────────────────────────────────────────────────────────────
from .data_cleaning_utils import limpiar_datos
from .data_encoding_utils import encode_data
from apiTimeSeries.train import train_arima_model, train_xgboost_model, train_lstm_model, train_patchtsmixer_model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
from api.utils import (
    configure_dvc_remote_logic,
    generate_experiment_summary_pdf,
    init_dvc_logic,
    send_progress_update)
from api.services import(
    create_experiment_logic
)
# ─────────────────────────────────────────────────────────────────────────────
# Logger y configuración global
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
mlflow_process = None  # Global variable for MLflow process


class PreProcessingService():
    def __init__(self) -> None:
        pass

    def analyze_csv_logic(self, csv_file) -> dict:
        """
        Lógica interna para analizar columnas de un CSV.
        - Recibe un 'csv_file' (un archivo ya abierto o un objeto InMemoryUploadedFile).
        - Retorna un dict con { "columns": [...] }.
            - Cada elemento de la lista tiene el formato: "nombreColumna - dtype"
        Lanza excepción si hay error al leer el CSV.
        """
        try:
            # Leer solo las cabeceras del CSV para optimizar
            df = pd.read_csv(csv_file, nrows=0)
            columns = list(df.columns)
            return {"columns": columns}
        except Exception as e:
            logger.error(f"Error al analizar el CSV: {e}", exc_info=True)
            raise

    def preview_date_transformation(self, csv_file, date_column: str, standardization_type: str) -> dict:
        """
        Preview date transformation without modifying the original data.

        Returns format detection, sample transformations, and validation warnings.
        """
        import io
        from .data_cleaning_utils import preview_date_transformation

        try:
            # Read first 100 rows for preview
            content = csv_file.read()
            df_preview = pd.read_csv(io.BytesIO(content), nrows=100)

            # Call the preview function from data_cleaning_utils
            result = preview_date_transformation(df_preview, date_column, standardization_type)

            return result

        except Exception as e:
            logger.error(f"Error in preview_date_transformation: {e}", exc_info=True)
            return {
                "format_detection": None,
                "preview_samples": [],
                "validation_warnings": [f"Error processing preview: {str(e)}"]
            }

    def upload_and_clean_csv_logic(
        self,
        csv_file,
        experiment_dir: str,
        optional_methods: list,
    ) -> dict:
        """
        Lógica interna para subir y limpiar un archivo CSV, registrándolo en MLflow,
        versionando en DVC y actualizando pipeline_config con rutas relativas.
        
        Retorna un dict con:
        {
        "status": "Archivo CSV limpio para EDA generado correctamente",
        "run_id": <run_id>,
        "raw_file_path": <rel_raw>,
        "processed_eda_path": <rel_processed>
        }
        o lanza excepción si algo falla.
        """
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        from codecarbon import EmissionsTracker
        from mlflow.data.pandas_dataset import PandasDataset  # Importar clase para tracking de datasets

        channel_layer = get_channel_layer()
        print("service - experiment_dir; ", experiment_dir)

        if not experiment_dir or not os.path.isdir(experiment_dir):
            raise ValueError(f"La ruta de experimento '{experiment_dir}' no es válida.")

        # 1. Usar basename para evitar rutas absolutas en csv_file.name
        filename_only = os.path.basename(csv_file.name)
        rel_raw_path = os.path.join("raw", filename_only)
        rel_processed_eda_path = os.path.join("processed", f"processed_eda_{filename_only}")

        raw_file_path = os.path.join(experiment_dir, rel_raw_path)
        processed_eda_path = os.path.join(experiment_dir, rel_processed_eda_path)

        os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_eda_path), exist_ok=True)

        # 2. Guardar el CSV en disco SOLO si no existe o está vacío
        if not os.path.exists(raw_file_path) or os.path.getsize(raw_file_path) == 0:
            with open(raw_file_path, 'wb') as f:
                for chunk in csv_file.chunks():
                    f.write(chunk)
            logger.info(f"Archivo CSV guardado en: {raw_file_path}")
        else:
            logger.info(f"Archivo CSV ya existe en: {raw_file_path}, usándolo sin reescribir")

        # Actualización de progreso: 20%
        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                "progreso",
                {"type": "update_progress", "progress": 20}
            )
        else:
            logger.error("Error: 'channel_layer' no está inicializado correctamente.")

        # 3. Configurar MLflow
        base_dir = os.path.dirname(experiment_dir)
        shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
        set_tracking_uri(f"sqlite:///{shared_db_path}")
        experiment_name = os.path.basename(experiment_dir)
        print("Experiment name: ...", experiment_name)
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        if not mlflow_experiment:
            raise ValueError(f"El experimento '{experiment_name}' no se encontró en MLflow.")

        mlflow_experiment_id = mlflow_experiment.experiment_id
        logger.info(f"Configurando MLflow para el experimento '{experiment_name}' (ID: {mlflow_experiment_id})")

        # 4. Iniciar run principal
        with start_run(experiment_id=mlflow_experiment_id,log_system_metrics=True) as run:
            run_id = run.info.run_id
            logger.info(f"Iniciando run de MLflow (Run ID: {run_id}) para data_cleaning")


            #Registrar el dataset crudo antes de limpiarlo
            # Fix: Explicit UTF-8 encoding for cross-platform reproducibility
            # Ensures consistent text decoding across different OS locales (Windows/Linux/MacOS)
            raw_data = pd.read_csv(raw_file_path, encoding='utf-8')  # Cargar el CSV original
            raw_dataset = mlflow.data.from_pandas(
            raw_data, 
            source=raw_file_path, 
            name="Dataset Crudo"
            )
            mlflow.log_input(raw_dataset, context="raw_data")  # Registrar en MLflow


            # Loguear parámetros en MLflow
            log_param("step", "data_cleaning")
            log_param("raw_file", rel_raw_path)
            log_param("processed_eda_file", rel_processed_eda_path)
            log_param("optional_methods", optional_methods)

            # 5. Versionar archivo crudo en DVC
            try:
                subprocess.run(["dvc", "add", raw_file_path], cwd=experiment_dir, check=True)
                logger.info(f"Archivo crudo versionado con DVC: {raw_file_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al versionar el archivo crudo con DVC: {e}", exc_info=True)
                raise RuntimeError(f"Error al versionar el archivo crudo con DVC: {e}")
            # Actualización de progreso: 40%
            async_to_sync(channel_layer.group_send)(
                "progreso_group",
                {"type": "send_progress", "progress": 40}
            )

            # Commit de cambios en Git (incluyendo archivos .dvc)
            try:
                subprocess.run(["git", "add", raw_file_path + ".dvc"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", ".dvc/config"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", "raw/.gitignore"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "commit", "-m", f"Add raw data {rel_raw_path} with DVC"], cwd=experiment_dir, check=True)
                logger.info(f"Archivo crudo .dvc añadido y comiteado en Git: {raw_file_path}.dvc")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al comitear el archivo crudo .dvc en Git: {e}", exc_info=True)
                raise RuntimeError(f"Error al comitear el archivo crudo .dvc en Git: {e}")

            # Push del archivo crudo a DVC remoto
            try:
                subprocess.run(["dvc", "push", raw_file_path], cwd=experiment_dir, check=True)
                logger.info(f"Archivo crudo subido al remoto de DVC: {raw_file_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al subir el archivo crudo al remoto de DVC: {e}", exc_info=True)
                raise RuntimeError(f"Error al subir el archivo crudo al remoto de DVC: {e}")
            # Actualización de progreso: 50%
            async_to_sync(channel_layer.group_send)(
                "progreso_group",
                {"type": "send_progress", "progress": 50}
            )

            # Loguear artefacto en MLflow
            try:
                log_artifact(raw_file_path, artifact_path="raw_data")
                logger.info(f"Archivo crudo logueado como artefacto en MLflow: {raw_file_path}")
            except Exception as e:
                logger.error(f"Error al loguear el archivo crudo en MLflow: {e}", exc_info=True)
                raise RuntimeError(f"Error al loguear el archivo crudo en MLflow: {e}")

            # 6. Llamar a la función de limpieza y capturar el reporte

            try:
                energy_consumed_total = 0.0
                tracker = EmissionsTracker(output_dir=".", save_to_file=False,allow_multiple_runs=True)
                tracker.start()
                cleaning_report = limpiar_datos(
                    csv_input=raw_file_path,
                    csv_output_eda=processed_eda_path,
                    optional_methods=optional_methods
                )
                # Detener el tracker y obtener métricas de energía y emisiones
                tracker.stop()
                energy_consumed_total = float(tracker._total_energy) # devuelto como float
                carbon_emission_kg = float(tracker.final_emissions)

                if energy_consumed_total is None:
                    energy_consumed_total = 0.0
                if carbon_emission_kg is None:
                    carbon_emission_kg = 0.0

                mlflow.log_metric("energy_consumed_total_kWh", energy_consumed_total)
                mlflow.log_metric("carbon_emission_kg", carbon_emission_kg)

                #  Registrar dataset limpio después de la limpieza
                # Fix: Explicit UTF-8 encoding for cross-platform reproducibility
                cleaned_data = pd.read_csv(processed_eda_path, encoding='utf-8')  # Cargar el dataset ya limpiado
                cleaned_dataset = mlflow.data.from_pandas(
                cleaned_data, 
                source=processed_eda_path, 
                name="Dataset Limpio"
                )
                mlflow.log_input(cleaned_dataset, context="cleaned_data")  # Registrar en MLflow
                logger.info(f"Datos limpios generados en: {processed_eda_path}. Reporte: {cleaning_report}")
            except Exception as e:
                logger.error(f"Error al limpiar los datos: {e}", exc_info=True)
                raise RuntimeError(f"Error al limpiar los datos: {e}")

            # Guardar el reporte de limpieza en un archivo JSON y loguearlo en MLflow
            report_file_path = processed_eda_path.replace('.csv', '_cleaning_report.json')
            with open(report_file_path, 'w') as f:
                json.dump(cleaning_report, f, indent=4)
            mlflow.log_artifact(report_file_path, artifact_path="cleaning_report")
            try:
                log_artifact(report_file_path, artifact_path="cleaning_report")
                logger.info(f"Reporte de limpieza logueado como artefacto en MLflow: {report_file_path}")
            except Exception as e:
                logger.error(f"Error al loguear el reporte de limpieza en MLflow: {e}", exc_info=True)

            # Actualización de progreso: 70%
            async_to_sync(channel_layer.group_send)(
                "progreso_group",
                {"type": "send_progress", "progress": 70}
            )

            if not os.path.exists(processed_eda_path):
                logger.error("El archivo procesado para EDA no se generó correctamente.")
                raise FileNotFoundError("El archivo procesado para EDA no se generó correctamente.")

            # 7. Versionar archivo procesado en DVC
            try:
                subprocess.run(["dvc", "add", processed_eda_path], cwd=experiment_dir, check=True)
                logger.info(f"Archivo procesado versionado con DVC: {processed_eda_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al versionar el archivo procesado con DVC: {e}", exc_info=True)
                raise RuntimeError(f"Error al versionar el archivo procesado con DVC: {e}")
            # Actualización de progreso: 90%
            async_to_sync(channel_layer.group_send)(
                "progreso_group",
                {"type": "send_progress", "progress": 90}
            )

            # Commit de cambios en Git (incluyendo archivos .dvc)
            try:
                subprocess.run(["git", "add", processed_eda_path + ".dvc"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", ".dvc/config"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", "processed/.gitignore"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "commit", "-m", f"Add processed EDA data {rel_processed_eda_path} with DVC"], cwd=experiment_dir, check=True)
                logger.info(f"Archivo procesado .dvc añadido y comiteado en Git: {processed_eda_path}.dvc")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al comitear el archivo procesado .dvc en Git: {e}", exc_info=True)
                raise RuntimeError(f"Error al comitear el archivo procesado .dvc en Git: {e}")

            # Push del archivo procesado a DVC remoto
            try:
                subprocess.run(["dvc", "push", processed_eda_path], cwd=experiment_dir, check=True)
                logger.info(f"Archivo procesado subido al remoto de DVC: {processed_eda_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al subir el archivo procesado al remoto de DVC: {e}", exc_info=True)
                raise RuntimeError(f"Error al subir el archivo procesado al remoto de DVC: {e}")

            # Loguear artefacto en MLflow
            try:
                log_artifact(processed_eda_path, artifact_path="processed_eda_data")
                logger.info(f"Archivo procesado logueado como artefacto en MLflow: {processed_eda_path}")
            except Exception as e:
                logger.error(f"Error al loguear el archivo procesado en MLflow: {e}", exc_info=True)
                raise RuntimeError(f"Error al loguear el archivo procesado en MLflow: {e}")

            # 8. Actualizar pipeline_config
            pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")

            # Agregar la nueva información del reporte de limpieza en la configuración del paso
            step_config = {
                "step": "data_cleaning",
                "run_id": run_id,
                "inputs": {},
                "outputs": {
                    "raw_data": {
                        "path": rel_raw_path,
                        "dvc_file": rel_raw_path + ".dvc"
                    },
                    "processed_eda": {
                        "path": rel_processed_eda_path,
                        "dvc_file": rel_processed_eda_path + ".dvc"
                    }
                },
                "parameters": {
                    "optional_methods": optional_methods,
                },
                "cleaning_report": cleaning_report,
                "energy_metrics": {
        "energy_consumed_total_kWh": energy_consumed_total,
        "carbon_emission__kg": carbon_emission_kg
    }
                
            }

            try:
                if os.path.exists(pipeline_config_path):
                    with open(pipeline_config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {"steps": []}
                config["steps"].append(step_config)
                with open(pipeline_config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                logger.info(f"pipeline_config.json actualizado con el paso 'data_cleaning'.")

                subprocess.run(["dvc", "add", pipeline_config_path], cwd=experiment_dir, check=True)
                logger.info(f"pipeline_config.json versionado con DVC: {pipeline_config_path}")

                subprocess.run(["git", "add", pipeline_config_path + ".dvc"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", ".dvc/config"], cwd=experiment_dir, check=True)
                subprocess.run(["git", "commit", "-m", "Update pipeline_config.json after data_cleaning step with cleaning report"], cwd=experiment_dir, check=True)
                logger.info(f"pipeline_config.json .dvc añadido y comiteado en Git: {pipeline_config_path}.dvc")

                subprocess.run(["dvc", "push", pipeline_config_path], cwd=experiment_dir, check=True)
                logger.info(f"pipeline_config.json subido al remoto de DVC: {pipeline_config_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error al versionar o comitear pipeline_config.json con DVC/Git: {e}", exc_info=True)
                raise RuntimeError(f"Error al versionar o comitear pipeline_config.json con DVC/Git: {e}")
            except Exception as e:
                logger.error(f"Error al actualizar pipeline_config.json: {e}", exc_info=True)
                raise RuntimeError(f"Error al actualizar pipeline_config.json: {e}")

        # Actualización final: 100%
        async_to_sync(channel_layer.group_send)(
            "progreso_group",
            {"type": "send_progress", "progress": 100}
        )

        return {
            "status": "Archivo CSV limpio para EDA generado correctamente.",
            "run_id": run_id,
            "raw_file_path": rel_raw_path,
            "processed_eda_path": rel_processed_eda_path
        }

class EdaService():
    def __init__(self) -> None:
        pass

    def generate_eda_logic(self, 
        dataset_type:str, 
        experiment_dir:str, 
        run_id:str):
        """
        Generate EDA report using ydata_profiling (tsmode=True)

        Parameters:
        - dataset_type: either "eda" or "train" (determines which file to search)
        - experiment_dir: current experiment directory
        - run_id: main run from MLFlow

        Returns: either Json
        {
            "success": True,
            "ydata_report_path": "<relative path to html report>",
            "run_id": <nested_run_id>
        }
        or Exception
        """
        from codecarbon import EmissionsTracker
        # Validaciones
        if dataset_type not in ["eda", "train"]:
            raise ValueError("dataset_type no válido (use 'eda' o 'train').")
        if not experiment_dir or not os.path.isdir(experiment_dir):
            raise FileNotFoundError(f"Directorio del experimento no encontrado: {experiment_dir}")

        
        # Directorios donde se encuentran los datos procesados y se guardarán los reportes
        processed_dir = os.path.join(experiment_dir, "processed")
        eda_reports_dir_abs = os.path.join(experiment_dir, "eda_reports")
        os.makedirs(eda_reports_dir_abs, exist_ok=True)
        logger.info(f"Directorio para reportes EDA creado o existente: {eda_reports_dir_abs}")
        
        # Buscar el archivo procesado correspondiente al dataset_type
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Directorio processed no encontrado: {processed_dir}")
        
        files = [
            f for f in os.listdir(processed_dir)
            if f.startswith(f"processed_{dataset_type}_") and f.endswith(".csv")
        ]
        if not files:
            raise FileNotFoundError(f"No se encontró archivo procesado para dataset_type='{dataset_type}'.")
        
        # Seleccionar el primer archivo encontrado
        rel_input_csv = os.path.join("processed", files[0])
        abs_input_csv = os.path.join(experiment_dir, rel_input_csv)
        logger.info(f"Archivo procesado seleccionado: {abs_input_csv}")
        
        # Leer el CSV procesado
        try:
            df_limpio = pd.read_csv(abs_input_csv, encoding="utf-8-sig")
            logger.info(f"Datos cargados correctamente desde: {abs_input_csv} (Rows: {df_limpio.shape[0]}, Columns: {df_limpio.shape[1]})")
        except Exception as e:
            logger.error(f"Error al leer el archivo CSV: {abs_input_csv}", exc_info=True)
            raise RuntimeError(f"Error al leer el archivo CSV: {e}")
        
        # Definir nombres de los reportes y rutas (relativas y absolutas)
        rel_ydata_report = os.path.join("eda_reports", f"ydata_report_{dataset_type}.html")
        #rel_sweetviz_report = os.path.join("eda_reports", f"sweetviz_report_{dataset_type}.html")
        abs_ydata_report = os.path.join(experiment_dir, rel_ydata_report)
        #abs_sweetviz_report = os.path.join(experiment_dir, rel_sweetviz_report)

        energy_consumed_total = 0.0
        tracker = EmissionsTracker(output_dir=".", save_to_file=False,allow_multiple_runs=True)
        tracker.start()

        # Generar reporte ydata-profiling (solo si no existe)
        if not os.path.exists(abs_ydata_report):
            try:
                report_config = {
                    "title": f"EDA Report ({dataset_type.upper()})",
                    "explorative": True,
                    "tsmode": True,
                    "correlations": {
                        "auto": {"calculate": True},
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": True},
                        "kendall": {"calculate": True},
                        "phi_k": {"calculate": True},
                        "cramers": {"calculate": True},
                    },
                }
                reporte_ydata = ProfileReport(df_limpio, **report_config)
                reporte_ydata.to_file(abs_ydata_report)
                logger.info(f"Reporte ydata-profiling generado en: {abs_ydata_report}")
            except Exception as e:
                logger.error(f"Error al generar reporte ydata-profiling: {e}", exc_info=True)
                raise RuntimeError(f"Error al generar reporte ydata-profiling: {e}")
        else:
            logger.info(f"Reporte ydata-profiling ya existe: {abs_ydata_report}")
        
        # Generar reporte Sweetviz (solo si no existe)
        """
        if not os.path.exists(abs_sweetviz_report):
            try:
                reporte_sweetviz = sv.analyze(df_limpio)
                reporte_sweetviz.show_html(filepath=abs_sweetviz_report, open_browser=False)
                logger.info(f"Reporte Sweetviz generado en: {abs_sweetviz_report}")
            except Exception as e:
                logger.error(f"Error al generar reporte Sweetviz: {e}", exc_info=True)
                raise RuntimeError(f"Error al generar reporte Sweetviz: {e}")
        else:
            logger.info(f"Reporte Sweetviz ya existe: {abs_sweetviz_report}")
        """

        tracker.stop()
        energy_consumed_total = float(tracker._total_energy) # devuelto como float
        carbon_emission_kg = float(tracker.final_emissions)


        if energy_consumed_total is None:
            energy_consumed_total = 0.0
        if carbon_emission_kg is None:
            carbon_emission_kg = 0.0  
        # Configurar MLflow 
        base_dir = os.path.dirname(experiment_dir)
        shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
        set_tracking_uri(f"sqlite:///{shared_db_path}")
        logger.info(f"MLflow Tracking URI configurado a: sqlite:///{shared_db_path}")
        experiment_name = os.path.basename(experiment_dir)
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        if not mlflow_experiment:
            raise ValueError(f"El experimento '{experiment_name}' no se encontró en MLflow.")
        mlflow_experiment_id = mlflow_experiment.experiment_id
        logger.info(f"Experiment ID de MLflow: {mlflow_experiment_id}")
        
        # Iniciar un run en MLflow
        try:
            with start_run(experiment_id=mlflow_experiment_id,log_system_metrics=True) as run:
                run_id = run.info.run_id
                logger.info(f"Run anidado de MLflow iniciado (Run ID: {run_id}) para generate_eda")

                # Registrar el dataset procesado antes de generar el EDA
                processed_dataset = mlflow.data.from_pandas(
                df_limpio, 
                source="",  # 🔹 Evita la inferencia de source
                name=f"Dataset {dataset_type.upper()}"
            )
                mlflow.log_input(processed_dataset, context="processed_data")  # Registrar en MLflow

                log_param("step", "generate_eda")
                log_param("dataset_type", dataset_type)
                log_metric("num_rows", df_limpio.shape[0])
                log_metric("num_columns", df_limpio.shape[1])
                log_metric("missing_values", int(df_limpio.isnull().sum().sum()))
                log_metric("duplicate_rows", int(df_limpio.duplicated().sum()))
                mlflow.log_metric("energy_consumed_total_kWh", energy_consumed_total)
                mlflow.log_metric("carbon_emission_kg", carbon_emission_kg)

                
                try:
                    log_artifact(abs_ydata_report, artifact_path="eda_reports")
                    logger.info(f"Reporte ydata-profiling logueado en MLflow: {abs_ydata_report}")
                except Exception as e:
                    logger.error(f"Error al loguear ydata_report en MLflow: {e}", exc_info=True)
                    raise RuntimeError(f"Error al loguear ydata_report en MLflow: {e}")
                
                #try:
                #    log_artifact(abs_sweetviz_report, artifact_path="eda_reports")
                #    logger.info(f"Reporte Sweetviz logueado en MLflow: {abs_sweetviz_report}")
                #except Exception as e:
                #    logger.error(f"Error al loguear sweetviz_report en MLflow: {e}", exc_info=True)
                #    raise RuntimeError(f"Error al loguear sweetviz_report en MLflow: {e}")
        except Exception as e:
            logger.error(f"Error al iniciar run anidado en MLflow: {e}", exc_info=True)
            raise RuntimeError(f"Error al iniciar run anidado en MLflow: {e}")
        
        # Versionar los reportes con DVC
        try:
            subprocess.run(["dvc", "add", abs_ydata_report], cwd=experiment_dir, check=True)
            logger.info(f"Reporte ydata-profiling versionado con DVC: {abs_ydata_report}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al versionar ydata_report con DVC: {e}", exc_info=True)
            raise RuntimeError(f"Error al versionar ydata_report con DVC: {e}")
        
        #try:
        #    subprocess.run(["dvc", "add", abs_sweetviz_report], cwd=experiment_dir, check=True)
        #    logger.info(f"Reporte Sweetviz versionado con DVC: {abs_sweetviz_report}")
        #except subprocess.CalledProcessError as e:
            logger.error(f"Error al versionar sweetviz_report con DVC: {e}", exc_info=True)
            raise RuntimeError(f"Error al versionar sweetviz_report con DVC: {e}")
        
        # Commit de los archivos .dvc en Git
        try:
            # Convertir rutas a formato Unix (por consistencia)
            rel_ydata_report_unix = rel_ydata_report.replace("\\", "/")
            #rel_sweetviz_report_unix = rel_sweetviz_report.replace("\\", "/")
            dvc_files = [
                rel_ydata_report_unix + ".dvc",
                #rel_sweetviz_report_unix + ".dvc",
                "pipeline_config.json.dvc"
            ]
            for dvc_file in dvc_files:
                subprocess.run(["git", "add", dvc_file], cwd=experiment_dir, check=True)
                logger.info(f"Archivo .dvc añadido a Git: {dvc_file}")
            commit_message = f"Add EDA reports for dataset_type '{dataset_type}'"
            subprocess.run(["git", "commit", "-m", commit_message], cwd=experiment_dir, check=True)
            logger.info(f"Commit realizado en Git: {commit_message}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al comitear archivos .dvc en Git: {e}", exc_info=True)
            raise RuntimeError(f"Error al comitear archivos .dvc en Git: {e}")
        
        # Push de los reportes a DVC remoto
        try:
            subprocess.run(["dvc", "push", abs_ydata_report], cwd=experiment_dir, check=True)
            logger.info(f"Reporte ydata-profiling subido al remoto de DVC: {abs_ydata_report}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al subir ydata_report al remoto de DVC: {e}", exc_info=True)
            raise RuntimeError(f"Error al subir ydata_report al remoto de DVC: {e}")
        
        #try:
        #    subprocess.run(["dvc", "push", abs_sweetviz_report], cwd=experiment_dir, check=True)
        #    logger.info(f"Reporte Sweetviz subido al remoto de DVC: {abs_sweetviz_report}")
        #except subprocess.CalledProcessError as e:
        #    logger.error(f"Error al subir sweetviz_report al remoto de DVC: {e}", exc_info=True)
        #    raise RuntimeError(f"Error al subir sweetviz_report al remoto de DVC: {e}")
        
        # Actualizar pipeline_config.json para incluir el paso EDA
        pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")
        eda_step_config = {
            "step": "generate_eda",
            "dataset_type": dataset_type,
            "input_csv": os.path.join("processed", os.path.basename(abs_input_csv)).replace("\\", "/"),
            "ydata_report_path": rel_ydata_report.replace("\\", "/"),
            #"sweetviz_report_path": rel_sweetviz_report.replace("\\", "/"),
            "energy_metrics": {
        "energy_consumed_total_kWh": energy_consumed_total,
        "carbon_emission__kg": carbon_emission_kg
    }

        }
        
        try:
            if os.path.exists(pipeline_config_path):
                with open(pipeline_config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {"steps": []}
            config["steps"].append(eda_step_config)
            with open(pipeline_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("pipeline_config.json actualizado con el paso 'generate_eda'.")
        
            subprocess.run(["dvc", "add", pipeline_config_path], cwd=experiment_dir, check=True)
            logger.info(f"pipeline_config.json versionado con DVC: {pipeline_config_path}")
        
            subprocess.run(["git", "add", pipeline_config_path + ".dvc"], cwd=experiment_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Update pipeline_config.json after generate_eda step"], cwd=experiment_dir, check=True)
            logger.info("pipeline_config.json .dvc añadido y comiteado en Git.")
        
            subprocess.run(["dvc", "push", pipeline_config_path], cwd=experiment_dir, check=True)
            logger.info("pipeline_config.json subido al remoto de DVC.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al versionar o comitear pipeline_config.json: {e}", exc_info=True)
            raise RuntimeError(f"Error al versionar o comitear pipeline_config.json: {e}")
        except Exception as e:
            logger.error(f"Error al actualizar pipeline_config.json: {e}", exc_info=True)
            raise RuntimeError(f"Error al actualizar pipeline_config.json: {e}")
        
        return {
            "success": True,
            "ydata_report_path": rel_ydata_report.replace("\\", "/"),
            #"sweetviz_report_path": rel_sweetviz_report.replace("\\", "/"),
            "run_id": run_id  # run_id de la run anidada en MLflow
        }


class DataEncodingService():
    def __init__(self) -> None:
        pass

    def encode_csv_logic(self,
        csv_file,
        experiment_dir: str,
        input_features: list[str],
        target_variables: list[str],
        apply_target_ohe: bool,
        apply_target_label: bool,
        lag_periods: int = 0,
        lag_nan_handling: str = "leave_as_is",
        date_column: Optional[str] = None
    ) -> dict:
        from codecarbon import EmissionsTracker

        # 1. Validaciones iniciales
        if not experiment_dir or not os.path.isdir(experiment_dir):
            raise ValueError(f"La ruta proporcionada no es válida: {experiment_dir}")

        # Allow empty input_features for univariate time series models (ARIMA, LSTM)
        # XGBoost validation happens at training layer
        if not target_variables:
            raise ValueError("Variable de salida no especificada.")

        if input_features is None:
            raise ValueError("Variables de entrada no especificadas (debe ser una lista, puede estar vacía para modelos univariados).")

        # Log univariate mode detection
        if len(input_features) == 0:
            logger.info("Modo univariado detectado - input_features vacío. Apropiado para ARIMA/LSTM.")

        if apply_target_ohe and apply_target_label:
            raise ValueError("No se puede usar OHE y LabelEncoder simultáneamente.")

        # 2. Decidir carpeta de entrada según si el archivo ya está "processed_" o no
        filename_only = os.path.basename(csv_file.name)
        if filename_only.startswith("processed_"):
            # Si el archivo ya es "processed_XYZ.csv", lo ponemos en 'processed/'
            rel_raw_file_path = os.path.join("processed", filename_only)
        else:
            # Si no, asumimos que es un CSV "crudo" y lo ponemos en 'raw/'
            rel_raw_file_path = os.path.join("raw", filename_only)

        # Para la salida, conservamos la misma lógica de "processed_train_..."
        rel_processed_train_path = os.path.join("processed", f"processed_train_{filename_only}")

        raw_file_path = os.path.join(experiment_dir, rel_raw_file_path)
        processed_train_path = os.path.join(experiment_dir, rel_processed_train_path)

        os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)

        # 3. Guardar CSV en disco SOLO si no existe o está vacío (se usa cuando se reproduce el pipeline con dvc get)
        if not os.path.exists(raw_file_path) or os.path.getsize(raw_file_path) == 0:
            try:
                with open(raw_file_path, 'wb') as f:
                    for chunk in csv_file.chunks():
                        f.write(chunk)
                logger.info(f"Archivo CSV guardado en: {raw_file_path}")
            except Exception as e:
                logger.error(f"Error al guardar el archivo CSV: {raw_file_path}", exc_info=True)
                raise RuntimeError(f"Error al guardar el archivo CSV: {e}")
        else:
            logger.info(f"Archivo CSV ya existe en: {raw_file_path}, usándolo sin reescribir")

        # 4. Validar columnas
        try:
            df = pd.read_csv(raw_file_path)
            # Validate all specified columns exist (handles empty input_features)
            all_columns_to_validate = list(input_features) + list(target_variables)
            for feature in all_columns_to_validate:
                if feature not in df.columns:
                    raise ValueError(f"Columna no encontrada: {feature}")

            if len(input_features) == 0:
                logger.info("Validación de columnas completada (modo univariado - sin features).")
            else:
                logger.info(f"Validación de columnas completada exitosamente. Features: {len(input_features)}")
        except Exception as e:
            logger.error(f"Error al validar columnas en el archivo CSV: {raw_file_path}", exc_info=True)
            raise RuntimeError(f"Error al validar columnas en el archivo CSV: {e}")

        # 5. Configurar MLflow
        base_dir = os.path.dirname(experiment_dir)
        shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
        set_tracking_uri(f"sqlite:///{shared_db_path}")
        logger.info(f"MLflow Tracking URI configurado a: sqlite:///{shared_db_path}")

        experiment_name = os.path.basename(experiment_dir)
        mlflow_experiment = get_experiment_by_name(experiment_name)
        if not mlflow_experiment:
            raise ValueError(f"El experimento '{experiment_name}' no fue encontrado en MLflow.")

        mlflow_experiment_id = mlflow_experiment.experiment_id
        logger.info(f"Experiment ID de MLflow: {mlflow_experiment_id}")

        # 6. Ejecutar run anidado en MLflow (se ignora run_id en este punto, pero se podría usar para anidar)
        try:
            with start_run(experiment_id=mlflow_experiment_id,log_system_metrics=True) as run:
                run_id = run.info.run_id
                logger.info(f"Run anidado de MLflow iniciado (Run ID: {run_id})")

                # Loguear parámetros en MLflow
                log_param("step", "data_encoding")
                log_param("processed_train_file", rel_processed_train_path)
                log_param("input_features", input_features if len(input_features) > 0 else "[]_univariate_mode")
                log_param("is_univariate", len(input_features) == 0)
                log_param("target_variables", target_variables)
                log_param("encode_target_ohe", apply_target_ohe)
                log_param("encode_target_label", apply_target_label)
                log_param("lag_periods", lag_periods)
                log_param("lag_nan_handling", lag_nan_handling)
                log_param("date_column", date_column)

                # 7. Codificar datos
                try:
                    energy_consumed_total = 0.0
                    tracker = EmissionsTracker(output_dir=".", save_to_file=False,allow_multiple_runs=True)
                    tracker.start()

                    # Registrar el dataset de entrada en MLflow antes de procesarlo
                    df_raw = pd.read_csv(raw_file_path)  # Leer el dataset original
                    raw_dataset = mlflow.data.from_pandas(
                    df_raw, 
                    source=None,  # Se omite el source para evitar warnings
                    name="Dataset Limpio"
                    )
                    mlflow.log_input(raw_dataset, context="raw_data")  # Registrar dataset crudo en MLflow


                    encode_data(
                        csv_input=raw_file_path,
                        csv_output_train=processed_train_path,
                        input_features=input_features,
                        target_variables=target_variables,
                        apply_ohe_to_target=apply_target_ohe,
                        apply_labelencoder_to_target=apply_target_label,
                        lag_periods=lag_periods,
                        lag_nan_handling=lag_nan_handling,
                        date_column=date_column
                    )
                    tracker.stop()
                    energy_consumed_total = float(tracker._total_energy) # devuelto como float
                    carbon_emission_kg = float(tracker.final_emissions)


                    if energy_consumed_total is None:
                        energy_consumed_total = 0.0

                    if carbon_emission_kg is None:
                        carbon_emission_kg = 0.0

                    mlflow.log_metric("energy_consumed_total_kWh", energy_consumed_total)
                    mlflow.log_metric("carbon_emission_kg", carbon_emission_kg)

                    logger.info(f"Datos codificados y guardados en: {processed_train_path}")
                except Exception as e:
                    logger.error(f"Error al codificar los datos: {e}", exc_info=True)
                    raise RuntimeError(f"Error al codificar los datos: {e}")

                if not os.path.exists(processed_train_path):
                    raise FileNotFoundError("El archivo codificado no se generó correctamente.")

                # Registrar información del archivo procesado
                try:
                    df_encoded = pd.read_csv(processed_train_path)

                    # Registrar el dataset codificado en MLflow después del procesamiento
                    encoded_dataset = mlflow.data.from_pandas(
                    df_encoded, 
                    source=None,  # Se omite el source para evitar warnings
                    name="Dataset Codificado"
                    )
                    mlflow.log_input(encoded_dataset, context="encoded_data")  # Registrar dataset codificado en MLflow

                    log_param("encoded_num_rows", df_encoded.shape[0])
                    log_param("encoded_num_columns", df_encoded.shape[1])
                    logger.info("Parámetros de métricas registrados en MLflow.")
                except Exception as e:
                    logger.error(f"Error al registrar métricas de MLflow: {e}", exc_info=True)
                    raise RuntimeError(f"Error al registrar métricas de MLflow: {e}")

                # 8. Versionar con DVC
                try:
                    subprocess.run(["dvc", "add", processed_train_path], cwd=experiment_dir, check=True)
                    logger.info(f"Archivo codificado versionado con DVC: {processed_train_path}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error al versionar processed_train_path con DVC: {e}", exc_info=True)
                    raise RuntimeError(f"Error al versionar processed_train_path con DVC: {e}")

                try:
                    log_artifact(processed_train_path, artifact_path="processed_train_data")
                    logger.info(f"Archivo codificado logueado en MLflow: {processed_train_path}")
                except Exception as e:
                    logger.error(f"Error al loguear processed_train_path en MLflow: {e}", exc_info=True)
                    raise RuntimeError(f"Error al loguear processed_train_path en MLflow: {e}")

                # 9. Actualizar pipeline_config.json
                pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")
                step_config = {
                    "step": "data_encoding",
                    "raw_file_path": rel_raw_file_path.replace("\\", "/"),
                    "processed_train_path": rel_processed_train_path.replace("\\", "/"),
                    "parameters": {
                        "input_features": input_features,
                        "target_variables": target_variables,
                        "encode_target_ohe": apply_target_ohe,
                        "encode_target_label": apply_target_label,
                        "lag_periods": lag_periods,
                        "lag_nan_handling": lag_nan_handling,
                        "date_column": date_column
                    },
                    "energy_metrics": {
                        "energy_consumed_total_kWh": energy_consumed_total,
                        "carbon_emission__kg": carbon_emission_kg
    }
                }

                try:
                    if os.path.exists(pipeline_config_path):
                        with open(pipeline_config_path, 'r') as config_file:
                            config = json.load(config_file)
                    else:
                        config = {"steps": []}

                    config["steps"].append(step_config)
                    with open(pipeline_config_path, 'w') as config_file:
                        json.dump(config, config_file, indent=4)
                    logger.info(f"pipeline_config.json actualizado con el paso 'data_encoding'.")

                    # Versionar pipeline_config.json con DVC
                    subprocess.run(["dvc", "add", pipeline_config_path], cwd=experiment_dir, check=True)
                    logger.info(f"pipeline_config.json versionado con DVC: {pipeline_config_path}")

                    # Añadir .dvc de pipeline_config.json a Git
                    subprocess.run(["git", "add", pipeline_config_path + ".dvc"], cwd=experiment_dir, check=True)
                    subprocess.run(["git", "commit", "-m", "Update pipeline_config.json after data_encoding step"], cwd=experiment_dir, check=True)
                    logger.info("pipeline_config.json .dvc añadido y comiteado en Git.")

                    # Push de pipeline_config.json a DVC remoto
                    subprocess.run(["dvc", "push", pipeline_config_path], cwd=experiment_dir, check=True)
                    logger.info("pipeline_config.json subido al remoto de DVC.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error al versionar o comitear pipeline_config.json: {e}", exc_info=True)
                    raise RuntimeError(f"Error al versionar o comitear pipeline_config.json: {e}")
                except Exception as e:
                    logger.error(f"Error al actualizar pipeline_config.json: {e}", exc_info=True)
                    raise RuntimeError(f"Error al actualizar pipeline_config.json: {e}")

        except Exception as e:
            logger.error(f"Error al iniciar run anidado en MLflow: {e}", exc_info=True)
            raise RuntimeError(f"Error al iniciar run anidado en MLflow: {e}")

        return {
            "status": "Archivo CSV codificado correctamente.",
            "processed_train_path": rel_processed_train_path.replace("\\", "/"),
            "run_id": run_id 
        }

class TrainModelService():
    def __init__(self) -> None:
        pass
    def train_model_logic(self,
                          dataset_file,
                          data:dict) ->dict:
        """
    Lógica interna para entrenar modelos con:
      - Ejecución en un único run en MLflow 
      - Versionamiento con DVC
      - Manejo robusto de parámetros

        Se ejecuta en un único run y retorna
        un diccionario "step_config" con la información necesaria para que el flujo general
        actualice el pipeline_config.json de forma unificada.
    """
    
        # 1. Validaciones iniciales
        experiment_dir = data.get("experiment_dir")
        if not experiment_dir or not os.path.exists(experiment_dir):
            raise FileNotFoundError("Directorio de experimento no encontrado o inválido.")

        algorithm = data.get("algorithm", "logistic").lower()
        supported_algorithms = ["xgboost", "arima", "lstm", "patchtsmixer"]
        if algorithm not in supported_algorithms:
            raise ValueError(f"Algoritmo no soportado. Use: {', '.join(supported_algorithms)}")

        # 2. Configurar MLflow
        base_dir = os.path.dirname(experiment_dir)
        shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")

        
        # Obtener el experimento de MLflow
        experiment_name = os.path.basename(experiment_dir)  # Nombre del experimento = nombre del directorio
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not mlflow_experiment:
            raise ValueError(f"No se encontró el experimento: {experiment_name}")
        
        mlflow_experiment_id = mlflow_experiment.experiment_id

        # 3. Iniciar un run único en MLflow
        with start_run(experiment_id=mlflow_experiment_id, description=f"Entrenamiento {algorithm}",log_system_metrics=True) as run:
            run_id = run.info.run_id
            logger.info(f"Run de MLflow iniciado - ID: {run_id}")

            try:
                # 4. Guardar dataset y versionar con DVC
                trained_dir = os.path.join(experiment_dir, "trained")
                os.makedirs(trained_dir, exist_ok=True)

                dataset_filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                dataset_path = os.path.join(trained_dir, dataset_filename)

                if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
                    with open(dataset_path, 'wb') as f:
                        for chunk in dataset_file.chunks():
                            f.write(chunk)
                    logger.info(f"Archivo dataset guardado en: {dataset_path}")
                else:
                    logger.info(f"Archivo dataset ya existe en: {dataset_path}, usándolo sin reescribir")

                subprocess.run(["dvc", "add", dataset_path], cwd=experiment_dir, check=True)
                subprocess.run(["git", "add", f"{dataset_path}.dvc"], cwd=experiment_dir, check=True)
                subprocess.run(
                    ["git", "commit", "-m", f"[DVC] Add training dataset {dataset_filename}"],
                    cwd=experiment_dir, check=True
                )
                subprocess.run(["dvc", "push", dataset_path], cwd=experiment_dir, check=True)

                # 5. Ejecutar el entrenamiento (llamando a la función correspondiente)
                #Registrar el dataset de entrada en MLflow antes de entrenar
                # Fix: Explicit UTF-8 encoding for cross-platform reproducibility
                df_train = pd.read_csv(dataset_path, encoding='utf-8')  # Leer dataset de entrenamiento
                train_dataset = mlflow.data.from_pandas(
                df_train, 
                source=None,  # Se omite el source para evitar warnings
                name="Dataset de Entrenamiento"
                )
                mlflow.log_input(train_dataset, context="train_data")  # Registrar dataset en MLflow
                
                mlflow.log_param("step", f"{algorithm}_training")
                if algorithm == "arima":
                    result = train_arima_model(
                        dataset_path=dataset_path,
                        data=data,
                        experiment_dir=experiment_dir
                    )
                elif algorithm == "xgboost":
                    result = train_xgboost_model(
                        dataset_path=dataset_path,
                        data=data,
                        experiment_dir=experiment_dir
                    )
                elif algorithm == "lstm":
                    result = train_lstm_model(
                        dataset_path=dataset_path,
                        data=data,
                        experiment_dir=experiment_dir
                    )
                elif algorithm == "patchtsmixer":
                    result = train_patchtsmixer_model(
                        dataset_path=dataset_path,
                        data=data,
                        experiment_dir=experiment_dir
                    )

                # 6. Versionar el modelo con DVC
                model_path = result.get("model_path")
                if model_path and os.path.exists(model_path):
                    subprocess.run(["dvc", "add", model_path], cwd=experiment_dir, check=True)
                    subprocess.run(["git", "add", f"{model_path}.dvc"], cwd=experiment_dir, check=True)
                    subprocess.run(
                        ["git", "commit", "-m", f"[DVC] Add model {os.path.basename(model_path)}"],
                        cwd=experiment_dir, check=True
                    )
                    subprocess.run(["dvc", "push", model_path], cwd=experiment_dir, check=True)

                # 7. Consolidar y registrar métricas en MLflow
                combined_metrics = {}
                if "val_metrics" in result:
                    filtered_val_metrics = {k: v for k, v in result["val_metrics"].items() if v is not None}
                    combined_metrics["val"] = filtered_val_metrics
                    mlflow.log_metrics(filtered_val_metrics)
                if "test_metrics" in result:
                    filtered_test_metrics = {k: v for k, v in result["test_metrics"].items() if v is not None}
                    combined_metrics["test"] = filtered_test_metrics
                    mlflow.log_metrics(filtered_test_metrics)

                mlflow.set_tag("training_phase", "completed")

                # 8. Preparar el diccionario step_config para actualizar el pipeline_config.json
                step_config = {
                    "step": f"train_{algorithm}",
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "dataset_path": os.path.relpath(dataset_path, experiment_dir),
                    "model_path": os.path.relpath(model_path, experiment_dir) if model_path else None,
                    "metrics": combined_metrics,
                    "timestamp": datetime.now().isoformat()
                }

                # Retornar la información del entrenamiento y el step_config
                return {
                    "status": "Modelo registrado correctamente en MLflow.",
                    "val_metrics": result.get("val_metrics", {}),
                    "test_metrics": result.get("test_metrics", {}),
                    "model_path": model_path,
                    "step_config": step_config,
                    "run_id": run_id
                }

            except Exception as e:
                mlflow.end_run(status="FAILED")
                mlflow.set_tag("training_status", "failed")
                logger.error(f"Error durante el entrenamiento: {str(e)}", exc_info=True)
                raise RuntimeError(f"Error en el proceso de entrenamiento: {str(e)}")

class PipelineService():
    def __init__(self)->None:
        pass

    def run_pipeline_logic(self, data):
        """
    Orquesta la ejecución secuencial del pipeline utilizando DVC y MLflow.

    Se espera que 'data' contenga:
      - base_dir: ruta base donde se almacenan los experimentos.
      - pipeline_config: dict con la configuración del pipeline (con pasos para limpieza, EDA, encoding y entrenamiento).
      - run_eda: (booleano, opcional) indica si se debe ejecutar el paso EDA después de la limpieza.

    Retorna un diccionario con la información de los pasos ejecutados, incluyendo la ruta del resumen en PDF.
    """
        trainModelService = TrainModelService()
        dataEncodingService = DataEncodingService()
        preProcessingService = PreProcessingService()
        edaService = EdaService()

        try:
            logger.info("Iniciando run_pipeline_logic...")
            base_dir = os.environ.get('EXPERIMENTS_DIR', '/app/experimentos')
            pipeline_config = data.get("pipeline_config")
            run_eda_flag = data.get("run_eda", False)

            if not base_dir or not os.path.isdir(base_dir):
                raise ValueError("La ruta base no es válida.")
            if not pipeline_config or "steps" not in pipeline_config:
                raise ValueError("No se recibió pipeline_config o no tiene 'steps'.")

            # 1. Crear NUEVO experimento.
            logger.info("Creando nuevo experimento...")
            create_exp_result = create_experiment_logic(base_dir)
            new_experiment_dir = create_exp_result["experiment_dir"]
            logger.info(f"Nuevo experimento creado: {new_experiment_dir}")
            data["experiment_dir"] = new_experiment_dir

            # 2. Inicializar DVC y configurar remoto.
            logger.info("Inicializando DVC en el nuevo experimento...")
            init_result = init_dvc_logic(new_experiment_dir)
            logger.info(f"Inicialización DVC: {init_result['status']}")
            logger.info("Configurando remoto compartido en el nuevo experimento...")
            remote_result = configure_dvc_remote_logic(new_experiment_dir)
            logger.info(f"Configuración remoto: {remote_result['status']}")

            # 3. Recuperar el archivo raw del experimento anterior.
            logger.info("Recuperando el archivo raw del experimento anterior...")
            old_experiment_name = pipeline_config.get("experiment_name")
            if old_experiment_name:
                old_experiment_dir = os.path.join(base_dir, old_experiment_name)

            # Asumimos que el primer step (data_cleaning) contiene la salida del raw_data.
            raw_rel_path = pipeline_config["steps"][0]["outputs"]["raw_data"]["path"]
            logger.info(f"Ruta relativa del archivo raw: {raw_rel_path}")
            new_raw_dir = os.path.join(new_experiment_dir, "raw")
            os.makedirs(new_raw_dir, exist_ok=True)
            csv_filename = os.path.basename(raw_rel_path)
            new_raw_path = os.path.join(new_raw_dir, csv_filename)

            logger.info("Ejecutando 'dvc get' para obtener el archivo raw...")
            try:
                subprocess.run(
                    ["dvc", "get", old_experiment_dir, raw_rel_path, "-o", new_raw_path],
                    cwd=new_experiment_dir,
                    check=True
                )
                logger.info(f"Archivo obtenido con dvc get: {new_raw_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error en dvc get: {e}", exc_info=True)
                raise RuntimeError(f"Error al obtener el archivo CSV con dvc get: {e}")

            try:
                with open(new_raw_path, "r", encoding="utf-8") as f:
                    snippet = "".join([f.readline() for _ in range(5)])
                logger.info(f"Snippet del archivo raw obtenido:\n{snippet}")
            except Exception as e:
                logger.warning(f"No se pudo leer un snippet del archivo raw: {e}")

            file_size = os.path.getsize(new_raw_path)
            if file_size == 0:
                raise RuntimeError(f"El archivo obtenido está vacío: {new_raw_path}")
            else:
                logger.info(f"Tamaño del archivo obtenido: {file_size} bytes")

            # 4. Preparar objeto PseudoFile para simular el archivo (como si fuera un InMemoryUploadedFile de Django).
            logger.info("Preparando objeto PseudoFile para el archivo raw...")
            class PseudoFile:
                def __init__(self, path):
                    self.path = path
                    self.name = os.path.basename(path)
                def chunks(self):
                    with open(self.path, "rb") as f:
                        while True:
                            chunk = f.read(8192)
                            if not chunk:
                                break
                            yield chunk
            csv_file_obj = PseudoFile(new_raw_path)
            try:
                pseudo_content = b"".join(list(csv_file_obj.chunks()))
                logger.info(f"Snippet desde PseudoFile (primeros 100 bytes): {pseudo_content[:100]!r}")
            except Exception as e:
                logger.warning(f"No se pudo leer contenido a través de PseudoFile: {e}")

            # 5. Extraer parámetros de limpieza (se asume que el primer step es de limpieza).
            logger.info("Extrayendo parámetros de limpieza...")
            cleaning_params = pipeline_config["steps"][0].get("parameters", {})
            optional_methods = cleaning_params.get("optional_methods")
            logger.info(f"Parámetros de limpieza - optional_methods: {optional_methods}")

            # 6. Ejecutar el paso de limpieza.
            logger.info("Ejecutando el paso de limpieza...")
            try:
                cleaning_result = preProcessingService.upload_and_clean_csv_logic(
                    csv_file=csv_file_obj,
                    experiment_dir=new_experiment_dir,
                    optional_methods=optional_methods
                )
                logger.info(f"Paso de limpieza completado exitosamente: {cleaning_result}")
                send_progress_update("data_cleaning", "OK")
            except Exception as e:
                logger.error(f"Error en el paso de limpieza: {e}", exc_info=True)
                send_progress_update("data_cleaning", "ERROR")
                raise RuntimeError(f"Error al limpiar los datos: {e}")

            # 7. Ejecutar el paso de EDA (si se requiere).
            eda_result = None
            has_eda_step = any(step.get("step") == "generate_eda" for step in pipeline_config.get("steps", []))
            if run_eda_flag or has_eda_step:
                logger.info("Ejecutando el paso de EDA automático...")
                try:
                    eda_result = edaService.generate_eda_logic(
                        dataset_type="eda",
                        experiment_dir=new_experiment_dir,
                        run_id=cleaning_result.get("run_id", "")
                    )
                    logger.info(f"Paso de EDA completado exitosamente: {eda_result}")
                    send_progress_update("eda", "OK")
                except Exception as e:
                    logger.error(f"Error en el paso de EDA: {e}", exc_info=True)
                    send_progress_update("eda", "ERROR")
                    raise RuntimeError(f"Error al generar el reporte EDA: {e}")
            else:
                logger.info("Paso de EDA no configurado; se omite.")

            # 8. Ejecutar el paso de Encoding (si se encuentra configurado).
            encoding_result = None
            encoding_params = None
            for step in pipeline_config.get("steps", []):
                if step.get("step") == "data_encoding":
                    encoding_params = step.get("parameters")
                    break

            if encoding_params:
                input_features = encoding_params.get("input_features")
                target_variables = encoding_params.get("target_variables")
                apply_target_ohe = encoding_params.get("encode_target_ohe", False)
                apply_target_label = encoding_params.get("encode_target_label", False)
                date_column = encoding_params.get("date_column", None)
                lag_periods = encoding_params.get("lag_periods", 0)
                lag_nan_handling = encoding_params.get("lag_nan_handling", "leave_as_is")
                if not input_features or not target_variables:
                    raise ValueError("Los parámetros 'input_features' y 'target_variables' para encoding son obligatorios.")
                logger.info(f"Parámetros de encoding: input_features={input_features}, target_variables={target_variables}, "
                            f"apply_target_ohe={apply_target_ohe}, apply_target_label={apply_target_label}, "
                            f"lag_periods={lag_periods}, lag_nan_handling={lag_nan_handling}")
                cleaned_rel_path = cleaning_result.get("processed_eda_path")
                if not cleaned_rel_path:
                    raise RuntimeError("El resultado de limpieza no contiene 'processed_eda_path'.")
                cleaned_abs_path = os.path.join(new_experiment_dir, cleaned_rel_path)
                if not os.path.exists(cleaned_abs_path):
                    raise FileNotFoundError(f"El archivo de limpieza no existe: {cleaned_abs_path}")

                class PseudoFileEncoding:
                    def __init__(self, path):
                        self.path = path
                        self.name = os.path.basename(path)
                    def chunks(self):
                        with open(self.path, "rb") as f:
                            while True:
                                chunk = f.read(8192)
                                if not chunk:
                                    break
                                yield chunk
                encoding_file_obj = PseudoFileEncoding(cleaned_abs_path)
                logger.info("Ejecutando el paso de encoding...")
                try:
                    encoding_result = dataEncodingService.encode_csv_logic(
                        csv_file=encoding_file_obj,
                        experiment_dir=new_experiment_dir,
                        input_features=input_features,
                        target_variables=target_variables,
                        apply_target_ohe=apply_target_ohe,
                        apply_target_label=apply_target_label,
                        lag_periods=lag_periods,
                        lag_nan_handling=lag_nan_handling,
                        date_column=date_column
                    )
                    logger.info(f"Paso de encoding completado exitosamente: {encoding_result}")
                    send_progress_update("data_encoding", "OK")
                except Exception as e:
                    logger.error(f"Error en el paso de encoding: {e}", exc_info=True)
                    send_progress_update("data_encoding", "ERROR")
                    raise RuntimeError(f"Error al codificar los datos: {e}")
            else:
                logger.info("Paso de encoding no configurado; se omite.")

            # 9. Ejecutar el paso de Entrenamiento (si se encuentra configurado).
            training_result = None
            training_params = None
            # Se busca el primer step cuyo nombre comience con "train" (por ejemplo, "train_mlp_model", "train_xgboost", etc.)
            for step in pipeline_config.get("steps", []):
                if str(step.get("step", "")).lower().startswith("train"):
                    training_params = step
                    break

            if training_params:
                # Actualizar datos de entrada si es necesario
                if "target_variable" in training_params and "target_variables" not in training_params:
                    training_params["target_variables"] = [training_params["target_variable"]]
                data["input_features"] = training_params.get("input_features")
                data["target_variable"] = training_params.get("target_variable")
                data["date_col_name"] = training_params.get("date_col_name")
                data["model_name"] = training_params.get("model_name", None)
                data["forecast_horizon"] = training_params.get("forecast_horizon", 1)
                data["problem_type"] = training_params.get("problem_type")
                data["manual_params"] = training_params.get("hyperparameters", {})
                if not data.get("manual_params"):
                    data["manual_params"] = training_params.get("params", {})
                data["use_grid_search"] = (training_params.get("grid_search") or {}).get("use_grid_search", False)
                data["grid_search"] = training_params.get("grid_search", {})
                
                # Extract random_search config from hyperparameter_search section (schema v1.1+)
                # with backward compatibility for legacy top-level random_search
                hyperparameter_search_for_random = training_params.get("hyperparameter_search", {})

                # Try v1.1 schema first (nested in hyperparameter_search)
                n_random_iterations = hyperparameter_search_for_random.get("n_random_iterations")
                random_search_params = hyperparameter_search_for_random.get("random_search_params")

                # Fallback to legacy schema if needed
                if n_random_iterations is None or random_search_params is None:
                    random_search_config = training_params.get("random_search", {})
                    if n_random_iterations is None:
                        n_random_iterations = random_search_config.get("n_random_iterations", 100)
                    if random_search_params is None:
                        random_search_params = random_search_config.get("random_search_params", {})

                data["n_random_iterations"] = n_random_iterations
                data["random_search_params"] = random_search_params
                data["hyperparameter_search_strategy"] = training_params.get("hyperparameter_search_strategy", "none")
                data["optimization_metric"] = training_params.get("optimization_metric", "val_rmse")
                # Extract grid_search_params from hyperparameter_search section (schema v1.1+)
                hyperparameter_search = training_params.get("hyperparameter_search", {})
                data["grid_search_params"] = hyperparameter_search.get("grid_search_params", {})
                # Extract bayesian_config (stored at step level, same as random_search)
                bayesian_config = training_params.get("bayesian_config", {})
                data["bayesian_config"] = bayesian_config
                if "run_id" not in data or not data["run_id"]:
                    data["run_id"] = cleaning_result.get("run_id")
                if encoding_result is not None and encoding_result.get("processed_train_path"):
                    training_file_rel = encoding_result.get("processed_train_path")
                else:
                    training_file_rel = cleaning_result.get("processed_eda_path")
                if not training_file_rel:
                    raise RuntimeError("No se encontró archivo de dataset para entrenamiento.")
                training_file_abs = os.path.join(new_experiment_dir, training_file_rel)
                if not os.path.exists(training_file_abs):
                    raise FileNotFoundError(f"El archivo de entrenamiento no existe: {training_file_abs}")
                class PseudoFileTraining:
                    def __init__(self, path):
                        self.path = path
                        self.name = os.path.basename(path)
                    def chunks(self):
                        with open(self.path, "rb") as f:
                            while True:
                                chunk = f.read(8192)
                                if not chunk:
                                    break
                                yield chunk
                training_file_obj = PseudoFileTraining(training_file_abs)
                logger.info("Ejecutando el paso de entrenamiento...")
                try:
                    if "algorithm" not in data:
                        # First try to get algorithm from training_params
                        if "algorithm" in training_params:
                            data["algorithm"] = training_params.get("algorithm")
                        else:
                            # Fallback: extract from step name for backward compatibility
                            alg = str(training_params.get("step", "")).lower().replace("train_", "")
                            if alg == "logistic_regression":
                                alg = "logistic"
                            elif alg.endswith("_model"):
                                alg = alg.replace("_model", "")
                            data["algorithm"] = alg
                    data["training_params"] = training_params
                    # Llamada a la función unificada de entrenamiento (run único) que retorna "step_config"
                    training_result = trainModelService.train_model_logic(
                        dataset_file=training_file_obj,
                        data=data
                    )
                    logger.info(f"Paso de entrenamiento completado exitosamente: {training_result}")
                    send_progress_update(training_params.get("step", "training"), "OK")
                except Exception as e:
                    logger.error(f"Error en el paso de entrenamiento: {e}", exc_info=True)
                    send_progress_update(training_params.get("step", "training"), "ERROR")
                    raise RuntimeError(f"Error durante el entrenamiento: {e}")
            else:
                logger.info("Paso de entrenamiento no configurado; se omite.")

            # 10. Consolidar resultados de los pasos ejecutados.
            result = {
                "status": "Pipeline ejecutado correctamente.",
                "experiment_dir": new_experiment_dir,
                "data_cleaning": cleaning_result
            }
            if eda_result is not None:
                result["eda"] = eda_result
            if encoding_result is not None:
                result["data_encoding"] = encoding_result
            if training_result is not None:
                result["training"] = training_result

            # 11. Actualizar el archivo pipeline_config.json (simplemente se agrega el step recibido)
            pipeline_config_path = os.path.join(new_experiment_dir, "pipeline_config.json")
            

            # 12. Generar el resumen automático en PDF.
            summary_pdf_path = os.path.join(new_experiment_dir, "experiment_summary.pdf")
            try:
                generate_experiment_summary_pdf(pipeline_config_path, summary_pdf_path)
                logger.info("Resumen del experimento generado en: " + summary_pdf_path)
                result["summary_pdf_path"] = summary_pdf_path
            except Exception as pdf_error:
                logger.error("Error generando el resumen del experimento: " + str(pdf_error))
                result["summary_pdf_path"] = f"Error generando resumen: {str(pdf_error)}"

            return result

        except Exception as ex:
            logger.error(f"Error en run_pipeline_logic: {ex}", exc_info=True)
            raise