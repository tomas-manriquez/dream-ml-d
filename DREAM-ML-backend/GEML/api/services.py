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
from typing import Union
from typing import Dict 

# ─────────────────────────────────────────────────────────────────────────────
# Librerias de terceros
# ─────────────────────────────────────────────────────────────────────────────
import mlflow
import pandas as pd
import psutil
import sweetviz as sv
from mlflow import (
    get_experiment_by_name, log_artifact, log_metric, log_param, set_experiment,
    set_tracking_uri, start_run
)
from mlflow.tracking import MlflowClient
from ydata_profiling import ProfileReport

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones locales
# ─────────────────────────────────────────────────────────────────────────────
from .data_cleaning import limpiar_datos
from .data_encoding import codificar_datos
from .train import (
    train_logistic_regression_model,
    train_mlp_model,
    train_xgboost_model
)
from .utils import (
    configure_dvc_remote_logic,
    generate_experiment_summary_pdf,
    init_dvc_logic,
    send_progress_update
)

# ─────────────────────────────────────────────────────────────────────────────
# Logger y configuración global
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
mlflow_process = None  # Global variable for MLflow process








def create_experiment_logic(base_dir: str) -> Dict[str, str]:
    """
    Crea un nuevo experimento en el directorio base especificado.
    
    Detiene cualquier instancia activa de MLflow, crea una nueva estructura de directorios,
    inicializa MLflow con una base de datos SQLite compartida y configura el pipeline inicial.

    Parámetros:
        base_dir (str): Ruta base donde se almacenarán los experimentos

    Retorna:
        dict: Diccionario con metadatos del experimento creado:
            - experiment_id: UUID único del experimento
            - experiment_dir: Ruta completa del directorio del experimento
            - artifact_uri: URI para artefactos de MLflow
            - mlflow_tracking_uri: URI de seguimiento de MLflow
            - mlflow_experiment_id: ID del experimento en MLflow
            - experiment_name: Nombre del experimento

    Excepciones:
        ValueError: Si la ruta base no es válida o si se detecta un experimento duplicado
        OSError: Si hay errores al crear directorios o archivos
    """
    global mlflow_process

    # Validación de la ruta base
    if not base_dir or not os.path.isdir(base_dir):
        raise ValueError("La ruta base proporcionada no existe o no es un directorio válido")

    # Detener servidor MLflow si está activo
    if mlflow_process and mlflow_process.poll() is None:
        try:
            logger.info("Deteniendo servidor MLflow activo...")
            parent = psutil.Process(mlflow_process.pid)
            
            # Terminar procesos hijos recursivamente
            for child in parent.children(recursive=True):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    logger.debug(f"Proceso hijo {child.pid} ya terminado")
            
            # Terminar proceso principal
            mlflow_process.terminate()
            mlflow_process.wait(timeout=5)
            logger.info("Servidor MLflow detenido exitosamente")
        except (psutil.NoSuchProcess, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error al detener MLflow: {str(e)}")
        finally:
            mlflow_process = None

    # Configurar tracking URI para MLflow
    shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
    shared_tracking_uri = f"sqlite:///{shared_db_path}"
    
    try:
        mlflow.set_tracking_uri(shared_tracking_uri)
        logger.info(f"Tracking URI configurado: {shared_tracking_uri}")
    except Exception as e:
        logger.error(f"Error configurando MLflow: {str(e)}")
        raise RuntimeError("No se pudo configurar MLflow") from e

    # Crear estructura de directorios usando hora local del sistema
    now = datetime.now()  # Hora local del sistema
    timestamp = now.strftime('%Y%m%d_%H%M%S')  # Formato compacto
    experiment_id = uuid.uuid4().hex  # Hex para nombres más cortos
    experiment_name = f"Exp_{timestamp}_{experiment_id[:8]}"  # Nombre conciso
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    try:
        os.makedirs(experiment_dir, exist_ok=True)
        logger.info(f"Directorio de experimento creado: {experiment_dir}")
    except OSError as e:
        logger.error(f"Error creando directorio: {str(e)}")
        raise OSError(f"No se pudo crear el directorio del experimento") from e

    # Directorio para artefactos
    artifact_dir = os.path.join(experiment_dir, "artifacts")
    artifact_uri = f"file:///{os.path.abspath(artifact_dir)}"
    
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        logger.info(f"Directorio de artefactos creado: {artifact_dir}")
    except OSError as e:
        logger.error(f"Error creando directorio de artefactos: {str(e)}")
        raise OSError(f"No se pudo crear el directorio de artefactos") from e

    # Crear/recuperar experimento en MLflow
    client = MlflowClient()
    existing_experiment = client.get_experiment_by_name(experiment_name)
    
    if existing_experiment:
        # Verificar consistencia en la ubicación de artefactos
        if existing_experiment.artifact_location != artifact_uri:
            error_msg = (f"Conflicto: Experimento '{experiment_name}' ya existe "
                         f"con diferente ubicación de artefactos")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        mlflow_experiment_id = existing_experiment.experiment_id
        logger.info(f"Usando experimento existente: {experiment_name}")
    else:
        try:
            mlflow_experiment_id = client.create_experiment(
                name=experiment_name,
                artifact_location=artifact_uri
            )
            logger.info(f"Nuevo experimento MLflow creado: {experiment_name} (ID: {mlflow_experiment_id})")
        except Exception as e:
            logger.error(f"Error creando experimento en MLflow: {str(e)}")
            raise RuntimeError("Fallo al crear experimento en MLflow") from e

    # Configurar experimento activo
    try:
        mlflow.set_experiment(experiment_name)
        logger.debug(f"Experimento configurado como activo: {experiment_name}")
    except Exception as e:
        logger.warning(f"Error configurando experimento activo: {str(e)}")

    # Crear configuración inicial del pipeline
    pipeline_config = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "created_at": now.isoformat(),
        "server_timezone": time.tzname,  # Registrar la zona horaria del sistema
        "steps": []
    }
    
    pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")
    
    try:
        with open(pipeline_config_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuración de pipeline creada: {pipeline_config_path}")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error creando pipeline_config: {str(e)}")
        raise RuntimeError("Fallo al crear configuración de pipeline") from e

    return {
        "experiment_id": str(experiment_id),
        "experiment_dir": experiment_dir,
        "artifact_uri": artifact_uri,
        "mlflow_tracking_uri": shared_tracking_uri,
        "mlflow_experiment_id": mlflow_experiment_id,
        "experiment_name": experiment_name,
        "server_time": now.isoformat(),  # Hora local del servidor
        "server_timezone": time.tzname   # Zona horaria del sistema
    }





def upload_and_clean_csv_logic(
    csv_file,
    experiment_dir: str,
    eliminar_duplicados: bool,
    filtrar_outliers: bool,
    relleno_valores_numericos: str,
    valor_imputacion: Union[float, None]
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
        raw_data = pd.read_csv(raw_file_path)  # Cargar el CSV original
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
        log_param("eliminar_duplicados", eliminar_duplicados)
        log_param("filtrar_outliers", filtrar_outliers)
        log_param("relleno_valores_numericos", relleno_valores_numericos)
        if relleno_valores_numericos == "valor":
            log_param("valor_imputacion", valor_imputacion)

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
                eliminar_duplicados=eliminar_duplicados,
                filtrar_outliers=filtrar_outliers,
                relleno_valores_numericos=relleno_valores_numericos,
                valor_imputacion=valor_imputacion
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
            cleaned_data = pd.read_csv(processed_eda_path)  # Cargar el dataset ya limpiado
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
                "eliminar_duplicados": eliminar_duplicados,
                "filtrar_outliers": filtrar_outliers,
                "relleno_valores_numericos": relleno_valores_numericos,
                "valor_imputacion": valor_imputacion,
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







def generate_eda_logic(
    dataset_type: str,
    experiment_dir: str,
    run_id
) -> dict:
    """
    Lógica interna para generar reportes de EDA (usando ydata_profiling y Sweetviz).
    
    Parámetros:
      - dataset_type: "eda" o "train" (determina qué archivo procesado buscar)
      - experiment_dir: directorio del experimento actual
      - run_id: run principal de MLflow (para anidar la ejecución)
    
    Retorna:
      {
        "success": True,
        "ydata_report_path": "<ruta relativa ydata>",
        "sweetviz_report_path": "<ruta relativa sweetviz>",
        "run_id": <nested_run_id>
      }
    o lanza excepción si algo falla.
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
    rel_sweetviz_report = os.path.join("eda_reports", f"sweetviz_report_{dataset_type}.html")
    abs_ydata_report = os.path.join(experiment_dir, rel_ydata_report)
    abs_sweetviz_report = os.path.join(experiment_dir, rel_sweetviz_report)

    energy_consumed_total = 0.0
    tracker = EmissionsTracker(output_dir=".", save_to_file=False,allow_multiple_runs=True)
    tracker.start()

    # Generar reporte ydata-profiling (solo si no existe)
    if not os.path.exists(abs_ydata_report):
        try:
            report_config = {
                "title": f"EDA Report ({dataset_type.upper()})",
                "explorative": True,
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
            
            try:
                log_artifact(abs_sweetviz_report, artifact_path="eda_reports")
                logger.info(f"Reporte Sweetviz logueado en MLflow: {abs_sweetviz_report}")
            except Exception as e:
                logger.error(f"Error al loguear sweetviz_report en MLflow: {e}", exc_info=True)
                raise RuntimeError(f"Error al loguear sweetviz_report en MLflow: {e}")
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
    
    try:
        subprocess.run(["dvc", "add", abs_sweetviz_report], cwd=experiment_dir, check=True)
        logger.info(f"Reporte Sweetviz versionado con DVC: {abs_sweetviz_report}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al versionar sweetviz_report con DVC: {e}", exc_info=True)
        raise RuntimeError(f"Error al versionar sweetviz_report con DVC: {e}")
    
    # Commit de los archivos .dvc en Git
    try:
        # Convertir rutas a formato Unix (por consistencia)
        rel_ydata_report_unix = rel_ydata_report.replace("\\", "/")
        rel_sweetviz_report_unix = rel_sweetviz_report.replace("\\", "/")
        dvc_files = [
            rel_ydata_report_unix + ".dvc",
            rel_sweetviz_report_unix + ".dvc",
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
    
    try:
        subprocess.run(["dvc", "push", abs_sweetviz_report], cwd=experiment_dir, check=True)
        logger.info(f"Reporte Sweetviz subido al remoto de DVC: {abs_sweetviz_report}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al subir sweetviz_report al remoto de DVC: {e}", exc_info=True)
        raise RuntimeError(f"Error al subir sweetviz_report al remoto de DVC: {e}")
    
    # Actualizar pipeline_config.json para incluir el paso EDA
    pipeline_config_path = os.path.join(experiment_dir, "pipeline_config.json")
    eda_step_config = {
        "step": "generate_eda",
        "dataset_type": dataset_type,
        "input_csv": os.path.join("processed", os.path.basename(abs_input_csv)).replace("\\", "/"),
        "ydata_report_path": rel_ydata_report.replace("\\", "/"),
        "sweetviz_report_path": rel_sweetviz_report.replace("\\", "/"),
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
        "sweetviz_report_path": rel_sweetviz_report.replace("\\", "/"),
        "run_id": run_id  # run_id de la run anidada en MLflow
    }




def encode_csv_logic(
    csv_file,
    experiment_dir: str,
    input_features: list[str],
    target_variables: list[str],
    apply_target_ohe: bool,
    apply_target_label: bool
) -> dict:
    """
    Lógica interna para codificar un archivo CSV (data_encoding).
    - csv_file: el archivo CSV (un InMemoryUploadedFile o un file-like object).
    - experiment_dir: ruta al directorio del experimento.
    - input_features: lista con nombres de columnas de entrada.
    - target_variables: lista con nombre(s) de columna(s) target.
    - run_id: ID del run padre en MLflow, donde anidamos este step.
    - apply_target_ohe: si True, aplica One-Hot Encoding al target.
    - apply_target_label: si True, aplica LabelEncoder al target.
    
    Retorna un dict con:
      {
        "status": "Archivo CSV codificado correctamente.",
        "processed_train_path": "<ruta RELATIVA>",
        "run_id": <nested_run_id>
      }
    o lanza excepción si ocurre un error.
    """
    from codecarbon import EmissionsTracker

    # 1. Validaciones iniciales
    if not experiment_dir or not os.path.isdir(experiment_dir):
        raise ValueError(f"La ruta proporcionada no es válida: {experiment_dir}")

    if not input_features or not target_variables:
        raise ValueError("Variables de entrada y/o de salida no especificadas.")

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
        for feature in (input_features + target_variables):
            if feature not in df.columns:
                raise ValueError(f"Columna no encontrada: {feature}")
        logger.info("Validación de columnas completada exitosamente.")
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
            log_param("input_features", input_features)
            log_param("target_variables", target_variables)
            log_param("encode_target_ohe", apply_target_ohe)
            log_param("encode_target_label", apply_target_label)

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


                codificar_datos(
                    csv_input=raw_file_path,
                    csv_output_train=processed_train_path,
                    input_features=input_features,
                    target_variables=target_variables,
                    apply_ohe_to_target=apply_target_ohe,
                    apply_labelencoder_to_target=apply_target_label
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
                    "encode_target_label": apply_target_label
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













def train_model_logic(dataset_file, data: dict) -> dict:
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
    supported_algorithms = ["logistic", "mlp", "xgboost"]
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
            df_train = pd.read_csv(dataset_path)  # Leer dataset de entrenamiento
            train_dataset = mlflow.data.from_pandas(
            df_train, 
            source=None,  # Se omite el source para evitar warnings
            name="Dataset de Entrenamiento"
            )
            mlflow.log_input(train_dataset, context="train_data")  # Registrar dataset en MLflow
            
            mlflow.log_param("step", f"{algorithm}_training")
            if algorithm == "logistic":
                result = train_logistic_regression_model(
                    dataset_path=dataset_path,
                    data=data,
                    experiment_dir=experiment_dir
                )
            elif algorithm == "mlp":
                result = train_mlp_model(
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









def run_pipeline_logic(data):
    """
    Orquesta la ejecución secuencial del pipeline utilizando DVC y MLflow.

    Se espera que 'data' contenga:
      - base_dir: ruta base donde se almacenan los experimentos.
      - pipeline_config: dict con la configuración del pipeline (con pasos para limpieza, EDA, encoding y entrenamiento).
      - run_eda: (booleano, opcional) indica si se debe ejecutar el paso EDA después de la limpieza.

    Retorna un diccionario con la información de los pasos ejecutados, incluyendo la ruta del resumen en PDF.
    """
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
        eliminar_duplicados = cleaning_params.get("eliminar_duplicados", True)
        filtrar_outliers = cleaning_params.get("filtrar_outliers", True)
        relleno_valores_numericos = cleaning_params.get("relleno_valores_numericos", "media")
        valor_imputacion = cleaning_params.get("valor_imputacion", None)
        logger.info(f"Parámetros de limpieza: eliminar_duplicados={eliminar_duplicados}, "
                    f"filtrar_outliers={filtrar_outliers}, relleno_valores_numericos={relleno_valores_numericos}, "
                    f"valor_imputacion={valor_imputacion}")

        # 6. Ejecutar el paso de limpieza.
        logger.info("Ejecutando el paso de limpieza...")
        try:
            cleaning_result = upload_and_clean_csv_logic(
                csv_file=csv_file_obj,
                experiment_dir=new_experiment_dir,
                eliminar_duplicados=eliminar_duplicados,
                filtrar_outliers=filtrar_outliers,
                relleno_valores_numericos=relleno_valores_numericos,
                valor_imputacion=valor_imputacion
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
                eda_result = generate_eda_logic(
                    dataset_type="eda",
                    experiment_dir=new_experiment_dir,
                    run_id=cleaning_result.get("run_id")
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
            if not input_features or not target_variables:
                raise ValueError("Los parámetros 'input_features' y 'target_variables' para encoding son obligatorios.")
            logger.info(f"Parámetros de encoding: input_features={input_features}, target_variables={target_variables}, "
                        f"apply_target_ohe={apply_target_ohe}, apply_target_label={apply_target_label}")
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
                encoding_result = encode_csv_logic(
                    csv_file=encoding_file_obj,
                    experiment_dir=new_experiment_dir,
                    input_features=input_features,
                    target_variables=target_variables,
                    apply_target_ohe=apply_target_ohe,
                    apply_target_label=apply_target_label
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
            data["problem_type"] = training_params.get("problem_type")
            data["params"] = training_params.get("hyperparameters", {})
            data["use_grid_search"] = (training_params.get("grid_search") or {}).get("use_grid_search", False)

            # Extract hyperparameter search strategy
            data["hyperparameter_search_strategy"] = training_params.get("hyperparameter_search_strategy", "none")

            # Extract Bayesian search configuration if present
            bayesian_search_config = training_params.get("bayesian_search", {})
            if bayesian_search_config.get("use_bayesian_search", False):
                data["bayesian_config"] = {
                    "n_trials": bayesian_search_config.get("n_trials"),
                    "n_initial_points": bayesian_search_config.get("n_initial_points"),
                    "timeout_seconds": bayesian_search_config.get("timeout_seconds")
                }
                data["bayesian_search_params"] = bayesian_search_config.get("bayesian_search_params", {})

            # Extract Random search configuration if present
            random_search_config = training_params.get("random_search", {})
            if random_search_config.get("use_random_search", False):
                data["n_random_iterations"] = random_search_config.get("n_random_iterations")
                data["random_search_params"] = random_search_config.get("random_search_params", {})

            # Extract split ratios if present
            if "split_ratios" in training_params:
                data["split_ratios"] = training_params.get("split_ratios")

            # Extract model name if present
            if "model_name" in training_params:
                data["model_name"] = training_params.get("model_name")

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
                    alg = str(training_params.get("step", "")).lower().replace("train_", "")
                    if alg == "logistic_regression":
                        alg = "logistic"
                    elif alg.endswith("_model"):
                        alg = alg.replace("_model", "")
                    data["algorithm"] = alg
                data["training_params"] = training_params
                # Llamada a la función unificada de entrenamiento (run único) que retorna "step_config"
                training_result = train_model_logic(
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


