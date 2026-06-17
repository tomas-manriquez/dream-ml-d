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
import os
import json
import subprocess
import logging
import mlflow

# ─────────────────────────────────────────────────────────────────────────────
# Librerias de terceros
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
# ─────────────────────────────────────────────────────────────────────────────
# Django
# ─────────────────────────────────────────────────────────────────────────────
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
# ─────────────────────────────────────────────────────────────────────────────
# Modulos locales
# ─────────────────────────────────────────────────────────────────────────────
from .services import (
    PipelineService, 
    PreProcessingService,
    EdaService,
    DataEncodingService,
    TrainModelService,
)
# ─────────────────────────────────────────────────────────────────────────────
# Logger & Global State
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

mlflow_process = None  # Estado global del proceso MLflow

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

# ─────────────────────────────────────────────────────────────────────────────
# Service Classes
# ─────────────────────────────────────────────────────────────────────────────
pipelineService = PipelineService()
preProcessingService = PreProcessingService()
edaService = EdaService()
dataEncodingService = DataEncodingService()
trainModelService = TrainModelService()


# ─────────────────────────────────────────────────────────────────────────────
# TS.0 Experiment Creation
# ─────────────────────────────────────────────────────────────────────────────
    
# ─────────────────────────────────────────────────────────────────────────────
# TS.1 Load File + Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


# Se va a usar 'analyze_csv' desde api.views porque la logica es la misma.
# Asi, evitamos codigo duplicado

@csrf_exempt
def preview_date_standardization(request):
    """
    Preview date standardization without processing the full file.

    Receives:
    - file: CSV file
    - date_column: string
    - standardization_type: string ("utc" or "retain_timezone")

    Returns:
    - format_detection: dict with detected format info
    - preview_samples: list of sample transformations
    - validation_warnings: list of warnings
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    if not request.FILES.get('file'):
        return JsonResponse({"error": "No file provided"}, status=400)

    try:
        csv_file = request.FILES['file']
        date_column = request.POST.get('date_column')
        standardization_type = request.POST.get('standardization_type')

        if not date_column or not standardization_type:
            return JsonResponse({
                "error": "Missing required parameters: date_column and standardization_type"
            }, status=400)

        # Call the preview function from data cleaning utils
        result = preProcessingService.preview_date_transformation(
            csv_file, date_column, standardization_type
        )

        return JsonResponse(result, status=200)

    except Exception as e:
        logger.error(f"Error in preview_date_standardization: {e}", exc_info=True)
        return JsonResponse({
            "error": f"Error previewing date standardization: {str(e)}"
        }, status=500)


@csrf_exempt
def upload_and_clean_csv(request):
    """
    Vista Django que recibe:
      - file: CSV
      - experiment_dir
      - eliminar_duplicados, filtrar_outliers, relleno_valores_numericos, valor_imputacion
    Llama a upload_and_clean_csv_logic(...) y devuelve JsonResponse con los resultados.
    """
    print("upload_and_clean_csv.view...\nRequest recieved: ", request)
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            csv_file = request.FILES['file']
            experiment_dir = request.POST.get('experiment_dir')
            print("view - experiment_dir: ",experiment_dir)

            # Validar experiment_dir
            if not experiment_dir:
                return JsonResponse({"status": "La ruta del experimento no se proporcionó."}, status=400)
            if not os.path.isdir(experiment_dir):
                return JsonResponse({"status": f"La ruta '{experiment_dir}' no existe o no es un directorio válido."}, status=400)

            # Validar archivo CSV
            if not csv_file.name.endswith('.csv'):
                return JsonResponse({"status": "El archivo debe ser un CSV."}, status=400)
            
            # Opcional: Limitar el tamaño del archivo (ejemplo: 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if csv_file.size > max_size:
                return JsonResponse({"status": "El archivo excede el tamaño máximo permitido de 10MB."}, status=400)

            #############
            eliminar_duplicados = request.POST.get('eliminar_duplicados', 'false').lower() == 'true'
            filtrar_outliers = request.POST.get('filtrar_outliers', 'false').lower() == 'true'
            relleno_valores_numericos = request.POST.get('relleno_valores_numericos', 'dejar')
            valor_imputacion = request.POST.get('valor_imputacion')
            if relleno_valores_numericos == 'valor':
                if valor_imputacion is None:
                    return JsonResponse({"status": "Se requiere 'valor_imputacion' cuando 'relleno_valores_numericos' es 'valor'."}, status=400)
                try:
                    valor_imputacion = float(valor_imputacion)
                except ValueError:
                    return JsonResponse({"status": "El valor de imputación debe ser un número."}, status=400)
            else:
                valor_imputacion = None  # Asegurar que sea None si no se usa
            #############
            # Leer parámetros de limpieza
            optional_methods= request.POST.get('optional_methods')

            # Llamar a la lógica pura
            result = preProcessingService.upload_and_clean_csv_logic(
                csv_file=csv_file,
                experiment_dir=experiment_dir,
                optional_methods=optional_methods
            )

            return JsonResponse(result, status=200)  # { "status", "run_id", "raw_file_path", ... }

        except subprocess.CalledProcessError as e:
            logger.error(f"Error en DVC: {e}", exc_info=True)
            return JsonResponse({"status": f"Error en DVC: {str(e)}"}, status=500)
        except RuntimeError as re:
            logger.error(f"Error de ejecución: {re}", exc_info=True)
            return JsonResponse({"status": f"Error de ejecución: {str(re)}"}, status=500)
        except ValueError as ve:
            logger.error(f"Valor inválido: {ve}", exc_info=True)
            return JsonResponse({"status": f"Valor inválido: {str(ve)}"}, status=400)
        except FileNotFoundError as fnfe:
            logger.error(f"Archivo no encontrado: {fnfe}", exc_info=True)
            return JsonResponse({"status": f"Archivo no encontrado: {str(fnfe)}"}, status=500)
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
            return JsonResponse({"status": f"Error inesperado: {str(e)}"}, status=500)

    return JsonResponse({"status": "Método no permitido"}, status=405)

# ─────────────────────────────────────────────────────────────────────────────
# TS.2 Auto EDA reports
# ────────────────────────────────────────────────────────────────────────────
@csrf_exempt
def generate_eda_report(request):
    """
    Django view that receives POSR requests with:
    - "dataset_type" (either "eda" or "train")
    - "experiment_dir"
    - "run_id"
    """
    if request.method != "POST":
        logger.warning("Método no permitido para generar_reporte_eda.")
        return JsonResponse({"success": False, "error": "Método no permitido."}, status=405)

    try:
        data = json.loads(request.body)
        dataset_type = data.get("dataset_type")
        experiment_dir = data.get("experiment_dir")
        run_id = data.get("run_id")

        # Validar parámetros
        if not dataset_type or dataset_type not in ["eda", "train"]:
            return JsonResponse({"success": False, "error": "Parámetro 'dataset_type' inválido o faltante. Use 'eda' o 'train'."}, status=400)

        if not experiment_dir or not os.path.isdir(experiment_dir):
            return JsonResponse({"success": False, "error": f"La ruta del experimento '{experiment_dir}' no es válida o no existe."}, status=400)

        if not run_id:
            return JsonResponse({"success": False, "error": "Parámetro 'run_id' faltante o inválido."}, status=400)

        # Llamar a la lógica pura
        result = edaService.generate_eda_logic(dataset_type, experiment_dir, run_id)
        return JsonResponse(result, status=200)

    except json.JSONDecodeError:
        logger.error("Error al decodificar el JSON en generar_reporte_eda.", exc_info=True)
        return JsonResponse({"success": False, "error": "JSON inválido."}, status=400)
    except ValueError as ve:
        logger.error(f"Error de valor en generar_reporte_eda: {ve}", exc_info=True)
        return JsonResponse({"success": False, "error": str(ve)}, status=400)
    except FileNotFoundError as fnfe:
        logger.error(f"Archivo no encontrado en generar_reporte_eda: {fnfe}", exc_info=True)
        return JsonResponse({"success": False, "error": str(fnfe)}, status=500)
    except RuntimeError as re:
        logger.error(f"Error de ejecución en generar_reporte_eda: {re}", exc_info=True)
        return JsonResponse({"success": False, "error": str(re)}, status=500)
    except Exception as e:
        logger.error(f"Error inesperado en generar_reporte_eda: {e}", exc_info=True)
        return JsonResponse({"success": False, "error": "Error interno del servidor."}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# TS.3 Dataset encoding for model training
# ────────────────────────────────────────────────────────────────────────────-
@csrf_exempt
def encode_csv(request):
    """
    Input:
    - 'file' (CSV file path)
    - 'experiment_dir' (string)
    - 'input_features' (list, passed as string)
    - 'target_variables' (list, passed as string)
    - 'run_id'
    - 'methods_to_apply' (list of dicts)
    Output:
    - JsonResponse()
    """
    if request.method != 'POST':
        logger.warning("Método no permitido para encode_csv.")
        return JsonResponse({"status": "Método no permitido."}, status=405)

    try:
        logger.info("Inicio del proceso encode_csv")

        # Obtener el archivo CSV
        csv_file = request.FILES.get('file')
        if not csv_file:
            logger.error("Archivo CSV no recibido.")
            return JsonResponse({"status": "Archivo CSV no recibido."}, status=400)

        # Obtener otros parámetros del POST
        experiment_dir = request.POST.get('experiment_dir')
        input_features_str = request.POST.get('input_features')
        target_variables_str = request.POST.get('target_variables')
        run_id = request.POST.get('run_id')

        encode_target_ohe = request.POST.get('encode_target_ohe', 'False')
        encode_target_label = request.POST.get('encode_target_label', 'False')
        apply_target_ohe = (encode_target_ohe.lower() == 'true')
        apply_target_label = (encode_target_label.lower() == 'true')

        # New lag feature parameters
        try:
            lag_periods = int(request.POST.get('lag_periods', '0'))
            if lag_periods < 0:
                raise ValueError("lag_periods must be non-negative")
        except ValueError:
            logger.error("Invalid lag_periods parameter")
            return JsonResponse({
                "status": "lag_periods must be a non-negative integer"
            }, status=400)

        lag_nan_handling = request.POST.get('lag_nan_handling', 'leave_as_is')
        date_column = request.POST.get('date_column', None)
        # Convert empty string to None
        if date_column == '':
            date_column = None

        # Validate lag_nan_handling parameter
        valid_nan_handling = ['drop', 'forward_fill', 'leave_as_is']
        if lag_nan_handling not in valid_nan_handling:
            logger.error(f"Invalid lag_nan_handling: {lag_nan_handling}")
            return JsonResponse({
                "status": f"Invalid lag_nan_handling. Use one of: {', '.join(valid_nan_handling)}"
            }, status=400)

        if not input_features_str or not target_variables_str:
            logger.error("Variables de entrada/salida no especificadas.")
            return JsonResponse({"status": "Variables de entrada/salida no especificadas."}, status=400)

        # Convertir a listas y limpiar espacios
        input_features = [feature.strip() for feature in input_features_str.split(",")]
        target_variables = [target.strip() for target in target_variables_str.split(",")]

        logger.debug(f"Input Features: {input_features}")
        logger.debug(f"Target Variables: {target_variables}")
        logger.debug(f"Apply OHE: {apply_target_ohe}")
        logger.debug(f"Apply Label Encoder: {apply_target_label}")
        logger.debug(f"Lag Periods: {lag_periods}")
        logger.debug(f"Lag NaN Handling: {lag_nan_handling}")
        logger.debug(f"Date Column: {date_column}")

        # Verificar que el run_id esté activo
        if not mlflow.get_run(run_id):
            logger.error(f"Run padre no encontrado o no activo: {run_id}")
            return JsonResponse({"status": f"Run padre no encontrado o no activo: {run_id}"}, status=400)

        # Llamar a la lógica interna de codificación
        result = dataEncodingService.encode_csv_logic(
            csv_file=csv_file,
            experiment_dir=experiment_dir,
            input_features=input_features,
            target_variables=target_variables,
            apply_target_ohe=apply_target_ohe,
            apply_target_label=apply_target_label,
            lag_periods=lag_periods,
            lag_nan_handling=lag_nan_handling,
            date_column=date_column
        )

        logger.info("Proceso encode_csv completado exitosamente.")
        return JsonResponse(result, status=200)

    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {str(e)}", exc_info=True)
        return JsonResponse({"status": f"Archivo no encontrado: {str(e)}"}, status=500)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error en DVC: {str(e)}", exc_info=True)
        return JsonResponse({"status": f"Error en DVC: {str(e)}"}, status=500)
    except ValueError as ve:
        logger.error(f"Error de valor: {ve}", exc_info=True)
        return JsonResponse({"status": f"Error de valor: {str(ve)}"}, status=400)
    except RuntimeError as re:
        logger.error(f"Error de ejecución: {re}", exc_info=True)
        return JsonResponse({"status": f"Error de ejecución: {str(re)}"}, status=500)
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        return JsonResponse({"status": f"Error inesperado: {str(e)}"}, status=500)

# ─────────────────────────────────────────────────────────────────────────────
# TS.4 Model Training
# ────────────────────────────────────────────────────────────────────────────-
@csrf_exempt
def train_model(request):
    """
    Vista Django para entrenamiento de modelos con gestión de MLflow.
    """
    if request.method != 'POST':
        return JsonResponse({
            "status": "error",
            "message": "Método no permitido. Use POST",
            "error_code": "HTTP_405_METHOD_NOT_ALLOWED"
        }, status=405)
    
    try:
        # 1. Validación inicial de la solicitud
        if 'file' not in request.FILES:
            raise ValueError("No se encontró archivo CSV en la solicitud")
        
        if 'data' not in request.POST:
            raise ValueError("Datos de configuración faltantes en el cuerpo de la solicitud")

        # 2. Parsear y validar parámetros
        data = json.loads(request.POST['data'])
        experiment_dir = data.get('experiment_dir')
        
        if not experiment_dir or not os.path.isdir(experiment_dir):
            raise FileNotFoundError(f"Directorio de experimento inválido: {experiment_dir}")

        # 3. Configuración inicial de MLflow
        base_dir = os.path.dirname(experiment_dir)
        shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{shared_db_path}")
        
        # 4. Limpieza preventiva de ejecuciones
        if mlflow.active_run():
            mlflow.end_run()
            logger.warning("Run MLflow activa detectada y finalizada")

        # 5. Ejecutar la lógica principal de entrenamiento
        result = trainModelService.train_model_logic(
            dataset_file=request.FILES['file'],
            data=data
        )

        # 6. Limpieza final asegurando estado limpio
        if mlflow.active_run():
            mlflow.end_run()

        experiment_name = os.path.basename(experiment_dir)
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not mlflow_experiment:
            raise ValueError(f"Experimento no encontrado: {experiment_name}")
        
        mlflow_experiment_id = mlflow_experiment.experiment_id
        

        return JsonResponse({
            "status": "success",
            "run_id": result.get("run_id", ""),
            "metrics": result.get("val_metrics", {}),  
            "model_path": result.get("model_path", ""),
            "mlflow_ui": f"http://{os.environ.get('MLFLOW_UI_URL', 'http://localhost:5000')}/#/experiments/{mlflow_experiment_id}/runs/{result.get('run_id', '')}"
        }, status=200)

    except json.JSONDecodeError as e:
        logger.error(f"Error parseando JSON: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": "Formato JSON inválido en datos de configuración",
            "error_details": str(e)
        }, status=400)
        
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": "Error en parámetros de entrada",
            "error_details": str(ve)
        }, status=400)
        
    except FileNotFoundError as fnfe:
        logger.error(f"Error de recursos: {str(fnfe)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": "Recurso no encontrado",
            "error_details": str(fnfe)
        }, status=404)
        
    except RuntimeError as re:
        logger.error(f"Error de ejecución: {str(re)}", exc_info=True)
        return JsonResponse({
            "status": "error",
            "message": "Error durante el entrenamiento",
            "error_details": str(re)
        }, status=500)
        
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        if mlflow.active_run():
            mlflow.end_run()
        return JsonResponse({
            "status": "error",
            "message": "Error interno del servidor",
            "error_details": str(e)
        }, status=500)

# ─────────────────────────────────────────────────────────────────────────────
# TS.4 Pipeline Config
# ────────────────────────────────────────────────────────────────────────────-
@csrf_exempt
def run_pipeline(request):
    """
    Vista Django que orquesta la ejecución del pipeline basado en un pipeline_config.json
    proporcionado por el usuario, generando un NUEVO experimento en la ruta base dada.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            result = pipelineService.run_pipeline_logic(data)
            return JsonResponse(result, status=200)
        except Exception as e:
            logger.error(f"Error general en run_pipeline: {e}", exc_info=True)
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    else:
        return JsonResponse({"success": False, "error": "Método no permitido."}, status=405)
