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
# ─────────────────────────────────────────────────────────────────────────────
# Librerias de terceros
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import seaborn as sns
import mlflow 

# ─────────────────────────────────────────────────────────────────────────────
# Django
# ─────────────────────────────────────────────────────────────────────────────
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


# ─────────────────────────────────────────────────────────────────────────────
# Modulos locales
# ─────────────────────────────────────────────────────────────────────────────

from .services import create_experiment_logic
from .utils import init_dvc_logic
from .utils import configure_dvc_remote_logic
from .utils import start_mlflow_logic
from .utils import analyze_csv_logic
from .services import upload_and_clean_csv_logic
from .services import generate_eda_logic
from .services import encode_csv_logic
from .utils import start_jupyter_logic
from .services import train_model_logic
from .services import run_pipeline_logic
from .utils import generate_experiment_summary_pdf

# ─────────────────────────────────────────────────────────────────────────────
# Logger & Global State
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

mlflow_process = None  # Estado global del proceso MLflow




@csrf_exempt
def create_experiment(request):
    """
    Vista que maneja la creación de nuevos experimentos.
    
    Recibe un POST con parámetros opcionales y crea un nuevo experimento
    en el directorio base configurado.
    
    Parámetros esperados:
        Ninguno en el cuerpo (se usa configuración de entorno)
        Opcionalmente se puede enviar JSON con:
            - directory_path: Ruta base alternativa
    
    Retorna:
        JsonResponse con:
            - status: Mensaje de estado
            - experiment_id: UUID del experimento
            - experiment_dir: Ruta completa del directorio
            - artifact_uri: URI para artefactos
            - mlflow_tracking_uri: URI de seguimiento MLflow
            - mlflow_experiment_id: ID de experimento en MLflow
            - experiment_name: Nombre del experimento
    
    Códigos de estado:
        - 201: Experimento creado exitosamente
        - 400: Error en la solicitud (ruta inválida)
        - 405: Método no permitido
        - 500: Error interno del servidor
    """
    if request.method != 'POST':
        return JsonResponse({
            "status": "Método no permitido",
            "allowed_methods": ["POST"]
        }, status=405)
    
    try:
        # Obtener directorio base de la configuración
        base_dir = os.environ.get('EXPERIMENTS_DIR', '/app/experimentos')
        
        # Intentar obtener ruta alternativa del cuerpo (si se envía)
        try:
            if request.body:
                data = json.loads(request.body)
                custom_dir = data.get('directory_path')
                if custom_dir and os.path.isdir(custom_dir):
                    base_dir = custom_dir
        except json.JSONDecodeError:
            logger.warning("Cuerpo de solicitud no es JSON válido, usando directorio por defecto")
        
        # Validar existencia del directorio base
        if not os.path.isdir(base_dir):
            error_msg = f"Directorio base no encontrado: {base_dir}"
            logger.error(error_msg)
            return JsonResponse({
                "status": error_msg,
                "suggestions": [
                    "Verifique la variable de entorno EXPERIMENTS_DIR",
                    "Cree el directorio manualmente si es necesario",
                    "Envíe un 'directory_path' válido en el cuerpo JSON"
                ]
            }, status=400)
        
        # Crear nuevo experimento
        result = create_experiment_logic(base_dir)
        
        logger.info(f"Experimento creado exitosamente: {result['experiment_name']}")
        return JsonResponse({
            "status": "Experimento creado exitosamente",
            "details": result
        }, status=201)
    
    except ValueError as ve:
        error_msg = f"Error de validación: {str(ve)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse({
            "status": error_msg,
            "type": "validation_error"
        }, status=400)
        
    except OSError as ose:
        error_msg = f"Error del sistema: {str(ose)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse({
            "status": error_msg,
            "type": "filesystem_error",
            "suggestion": "Verifique los permisos y espacio en disco"
        }, status=500)
        
    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse({
            "status": "Error interno al crear experimento",
            "details": error_msg,
            "type": "server_error"
        }, status=500)




@csrf_exempt
def init_dvc(request):
    """
    Vista Django para inicializar DVC en un directorio de experimentos.
    
    Requiere:
        Método: POST
        Content-Type: application/json
        Cuerpo: {"experiment_dir": "/ruta/al/directorio"}
    
    Respuestas:
        200: Éxito - {"status": "...", "experiment_dir": "..."}
        400: Solicitud inválida (JSON malformado, directorio faltante)
        405: Método no permitido
        500: Error del servidor (problemas con DVC/Git)
    """
    # Verificar método HTTP
    if request.method != 'POST':
        return JsonResponse({
            "status": "Método no permitido",
            "allowed_methods": ["POST"]
        }, status=405)
    
    try:
        # Parsear y validar JSON
        try:
            data = json.loads(request.body)
            experiment_dir = data.get('experiment_dir')
        except json.JSONDecodeError:
            return JsonResponse({
                "status": "Cuerpo de solicitud inválido: se espera JSON"
            }, status=400)
        
        # Validar parámetro obligatorio
        if not experiment_dir:
            return JsonResponse({
                "status": "Parámetro 'experiment_dir' requerido en el JSON"
            }, status=400)
        
        # Normalizar ruta del directorio
        experiment_dir = os.path.abspath(os.path.normpath(experiment_dir))
        
        # Ejecutar lógica de inicialización
        result = init_dvc_logic(experiment_dir)
        return JsonResponse(result, status=200)
    
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}", exc_info=True)
        return JsonResponse({
            "status": f"Error de validación: {str(ve)}",
            "directory": experiment_dir
        }, status=400)
    
    except subprocess.CalledProcessError as cpe:
        logger.error(f"Error en comando: {cpe.cmd}\nSalida: {cpe.output}\nError: {cpe.stderr}", 
                     exc_info=True)
        return JsonResponse({
            "status": f"Error al ejecutar comando: {cpe.cmd}",
            "error_details": cpe.stderr.decode('utf-8') if cpe.stderr else str(cpe)
        }, status=500)
    
    except PermissionError as pe:
        logger.error(f"Error de permisos: {str(pe)}", exc_info=True)
        return JsonResponse({
            "status": "Error de permisos en el sistema de archivos",
            "directory": experiment_dir
        }, status=500)
    
    except Exception as e:
        logger.critical(f"Error inesperado: {str(e)}", exc_info=True)
        return JsonResponse({
            "status": "Error interno del servidor",
            "error_type": type(e).__name__
        }, status=500)





@csrf_exempt
def configure_dvc_remote(request):
    """
    Vista Django para configurar un remoto DVC local compartido.

    Requiere una solicitud POST con JSON: {"experiment_dir": "/ruta/experimento"}
    
    Pasos:
    1. Valida el método HTTP (solo POST permitido)
    2. Parsea y valida los datos de entrada
    3. Verifica que el directorio del experimento exista
    4. Ejecuta la lógica de configuración del remoto DVC
    5. Maneja errores específicos y retorna respuestas adecuadas

    Respuestas posibles:
    - 200 OK: Remoto configurado exitosamente
    - 400 Bad Request: Datos inválidos o faltantes
    - 405 Method Not Allowed: Método HTTP no permitido
    - 500 Internal Server Error: Error en la configuración

    Args:
        request: HttpRequest de Django con datos en cuerpo JSON

    Returns:
        JsonResponse: Resultado de la operación o mensaje de error
    """
    # Validar método HTTP
    if request.method != 'POST':
        return JsonResponse(
            {"status": "Método no permitido", "allowed_methods": ["POST"]},
            status=405
        )

    try:
        # Parsear y validar datos de entrada
        try:
            data = json.loads(request.body)
            experiment_dir = data.get('experiment_dir', '').strip()
        except json.JSONDecodeError:
            return JsonResponse(
                {"status": "Formato JSON inválido en el cuerpo de la solicitud"},
                status=400
            )
        
        # Validar presencia de ruta
        if not experiment_dir:
            return JsonResponse(
                {"status": "Se requiere el campo 'experiment_dir'"},
                status=400
            )
        
        # Validar existencia del directorio
        if not os.path.isdir(experiment_dir):
            return JsonResponse(
                {
                    "status": "Directorio no encontrado",
                    "details": f"La ruta proporcionada no existe: {experiment_dir}"
                },
                status=400
            )
        
        logger.info(f"Iniciando configuración DVC remoto para: {experiment_dir}")
        
        # Ejecutar lógica principal
        result = configure_dvc_remote_logic(experiment_dir)
        
        logger.info(f"Configuración DVC remoto completada: {result['remote_path']}")
        return JsonResponse(result, status=200)

    except subprocess.CalledProcessError as e:
        error_msg = f"Error en comando DVC: {e.stderr.strip() if e.stderr else str(e)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse(
            {
                "status": "Error en configuración DVC",
                "details": error_msg,
                "command": e.cmd
            },
            status=500
        )
        
    except OSError as e:
        error_msg = f"Error del sistema: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse(
            {
                "status": "Error de operación del sistema",
                "details": error_msg
            },
            status=500
        )
        
    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JsonResponse(
            {
                "status": "Error interno del servidor",
                "details": error_msg
            },
            status=500
        )





@csrf_exempt
def start_mlflow(request) -> JsonResponse:
    """
    Vista para iniciar un servidor MLflow en el directorio especificado.

    Requiere una solicitud POST con JSON: {"directory_path": "/ruta/al/directorio"}
    
    Parámetros:
        request (HttpRequest): Objeto de solicitud Django
    
    Retorna:
        JsonResponse: 
            - 200 OK con información del servidor en caso de éxito
            - 400 Bad Request para solicitudes inválidas
            - 405 Method Not Allowed para métodos no permitidos
            - 500 Internal Server Error para errores del servidor
    
    Ejemplo de éxito:
        {
            "status": "Servidor MLflow iniciado exitosamente",
            "backend_store_uri": "sqlite:////ruta/shared_mlflow.db",
            "artifact_store": "/ruta/artifacts",
            "log_stdout": "/ruta/mlflow_logs/mlflow_stdout.log",
            "log_stderr": "/ruta/mlflow_logs/mlflow_stderr.log"
        }
    """
    if request.method != 'POST':
        logger.warning(f"Método no permitido: {request.method}")
        return JsonResponse(
            {"status": "Método no permitido. Use POST."}, 
            status=405
        )

    try:
        # Parsear y validar datos de entrada
        data = json.loads(request.body)
        base_dir = data.get('directory_path', '').strip()
        
        if not base_dir:
            logger.error("Falta 'directory_path' en la solicitud")
            return JsonResponse(
                {"status": "Se requiere el campo 'directory_path'."}, 
                status=400
            )
        
        if not os.path.isdir(base_dir):
            logger.error(f"Directorio inválido: '{base_dir}'")
            return JsonResponse(
                {"status": f"La ruta '{base_dir}' no es un directorio válido."}, 
                status=400
            )

        # Ejecutar lógica principal
        result = start_mlflow_logic(base_dir)
        logger.info(f"MLflow iniciado exitosamente en: {base_dir}")
        return JsonResponse(result, status=200)

    except json.JSONDecodeError:
        logger.error("Error decodificando JSON", exc_info=True)
        return JsonResponse(
            {"status": "Formato JSON inválido en la solicitud"}, 
            status=400
        )
    except ValueError as ve:
        logger.error(f"Error en los datos de entrada: {ve}", exc_info=True)
        return JsonResponse(
            {"status": f"Error en los datos: {str(ve)}"}, 
            status=400
        )
    except RuntimeError as re:
        logger.error(f"Error al iniciar MLflow: {re}", exc_info=True)
        return JsonResponse(
            {"status": f"Error al iniciar MLflow: {str(re)}"}, 
            status=500
        )
    except Exception as e:
        logger.critical(f"Error inesperado: {e}", exc_info=True)
        return JsonResponse(
            {"status": f"Error interno del servidor: {str(e)}"}, 
            status=500
        )







@csrf_exempt
def analyze_csv(request):
    """
    Vista Django que recibe un archivo CSV (via request.FILES['file'])
    y retorna las columnas.
    """
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            csv_file = request.FILES['file']

            # Validar el tipo de archivo (opcional)
            if not csv_file.name.endswith('.csv'):
                return JsonResponse({"error": "El archivo debe ser un CSV."}, status=400)

            # Opcional: Limitar el tamaño del archivo (ejemplo: 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if csv_file.size > max_size:
                return JsonResponse({"error": "El archivo excede el tamaño máximo permitido de 10MB."}, status=400)

            result = analyze_csv_logic(csv_file)  # Llamar a la lógica pura
            return JsonResponse(result, status=200)
        except pd.errors.EmptyDataError:
            logger.error("Archivo CSV está vacío.")
            return JsonResponse({"error": "El archivo CSV está vacío."}, status=400)
        except pd.errors.ParserError:
            logger.error("Error al parsear el archivo CSV.")
            return JsonResponse({"error": "Error al parsear el archivo CSV. Asegúrate de que está bien formateado."}, status=400)
        except Exception as e:
            logger.error(f"Error al analizar el archivo CSV: {e}", exc_info=True)
            return JsonResponse({"error": f"Error al analizar el archivo: {str(e)}"}, status=500)

    return JsonResponse({"error": "Método no permitido"}, status=405)








@csrf_exempt
def upload_and_clean_csv(request):
    """
    Vista Django que recibe:
      - file: CSV
      - experiment_dir
      - eliminar_duplicados, filtrar_outliers, relleno_valores_numericos, valor_imputacion
    Llama a upload_and_clean_csv_logic(...) y devuelve JsonResponse con los resultados.
    """
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            csv_file = request.FILES['file']
            experiment_dir = request.POST.get('experiment_dir')

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

            # Leer parámetros de limpieza
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

            # Llamar a la lógica pura
            result = upload_and_clean_csv_logic(
                csv_file=csv_file,
                experiment_dir=experiment_dir,
                eliminar_duplicados=eliminar_duplicados,
                filtrar_outliers=filtrar_outliers,
                relleno_valores_numericos=relleno_valores_numericos,
                valor_imputacion=valor_imputacion
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








@csrf_exempt
def generar_reporte_eda(request):
    """
    Vista Django que recibe POST con:
      - "dataset_type" ("eda" o "train")
      - "experiment_dir"
      - "run_id"
    Llama a generate_eda_logic(...) y retorna un JsonResponse con la info.
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
        result = generate_eda_logic(dataset_type, experiment_dir, run_id)
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







@csrf_exempt
def encode_csv(request):
    """
    Vista que recibe:
      - 'file' (CSV),
      - 'experiment_dir',
      - 'input_features',
      - 'target_variables',
      - 'run_id',
      - 'encode_target_ohe',
      - 'encode_target_label'.
    Llama a encode_csv_logic(...) y retorna JsonResponse.
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

        # Verificar que el run_id esté activo
        if not mlflow.get_run(run_id):
            logger.error(f"Run padre no encontrado o no activo: {run_id}")
            return JsonResponse({"status": f"Run padre no encontrado o no activo: {run_id}"}, status=400)

        # Llamar a la lógica interna de codificación
        result = encode_csv_logic(
            csv_file=csv_file,
            experiment_dir=experiment_dir,
            input_features=input_features,
            target_variables=target_variables,
            apply_target_ohe=apply_target_ohe,
            apply_target_label=apply_target_label
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







@csrf_exempt
def start_jupyter(request):
    """
    Vista que maneja POST con JSON:
      - "experiment_dir": <ruta>
      - "run_id": <MLflow parent run>
      - "port": <opcional, default=8888>
    Llama a start_jupyter_logic(...) y retorna JsonResponse.
    """
    if request.method != "POST":
        logger.warning("Método no permitido para start_jupyter.")
        return JsonResponse({"success": False, "error": "Método no permitido."}, status=405)

    try:
        data = json.loads(request.body)
        experiment_dir = data.get("experiment_dir")
        run_id = data.get("run_id")
        port = data.get("port", 8888)

        # Validaciones de parámetros
        if not experiment_dir:
            logger.error("Parámetro 'experiment_dir' faltante.")
            return JsonResponse({"success": False, "error": "Parámetro 'experiment_dir' faltante."}, status=400)

        if not run_id:
            logger.error("Parámetro 'run_id' faltante o inválido.")
            return JsonResponse({"success": False, "error": "Parámetro 'run_id' faltante o inválido."}, status=400)

        # Verificar que la run padre esté activa
        try:
            mlflow.get_run(run_id)
        except mlflow.exceptions.RestException as e:
            logger.error(f"Run padre no encontrada o no activa: {run_id}", exc_info=True)
            return JsonResponse({"success": False, "error": f"Run padre no encontrada o no activa: {run_id}"}, status=400)
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error al obtener run padre: {e}", exc_info=True)
            return JsonResponse({"success": False, "error": f"Error al obtener run padre: {e}"}, status=500)

        # Llamar a la lógica interna para iniciar Jupyter
        result = start_jupyter_logic(
            experiment_dir=experiment_dir,
            run_id=run_id,
            port=port
        )

        logger.info("Proceso start_jupyter completado exitosamente.")
        return JsonResponse(result, status=200)

    except json.JSONDecodeError:
        logger.error("Error al decodificar el JSON en start_jupyter.", exc_info=True)
        return JsonResponse({"success": False, "error": "JSON inválido."}, status=400)
    except FileNotFoundError as e:
        logger.error(f"Archivo no encontrado: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": f"Archivo no encontrado: {str(e)}"}, status=500)
    except OSError as e:
        logger.error(f"Error de sistema operativo: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": f"Error de sistema operativo: {str(e)}"}, status=500)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error en DVC o Git: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": f"Error en DVC o Git: {str(e)}"}, status=500)
    except RuntimeError as e:
        logger.error(f"Error de ejecución: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": f"Error de ejecución: {str(e)}"}, status=500)
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": "Error interno del servidor."}, status=500)








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
        result = train_model_logic(
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







@csrf_exempt
def get_pipeline_config(request):
    directory_path = request.GET.get("directory_path")
    pipeline_config_path = os.path.join(directory_path, "pipeline_config.json")

    if not os.path.exists(pipeline_config_path):
        return JsonResponse({"status": "No hay configuraciones registradas aún."}, status=404)

    try:
        with open(pipeline_config_path, 'r') as config_file:
            config = json.load(config_file)
        return JsonResponse(config)
    except Exception as e:
        return JsonResponse({"status": f"Error al leer las configuraciones: {str(e)}"}, status=500)







@csrf_exempt
def run_pipeline(request):
    """
    Vista Django que orquesta la ejecución del pipeline basado en un pipeline_config.json
    proporcionado por el usuario, generando un NUEVO experimento en la ruta base dada.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            result = run_pipeline_logic(data)
            return JsonResponse(result, status=200)
        except Exception as e:
            logger.error(f"Error general en run_pipeline: {e}", exc_info=True)
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    else:
        return JsonResponse({"success": False, "error": "Método no permitido."}, status=405)







@csrf_exempt
def get_experiment_summary(request):
    """
    Vista Django que genera y devuelve el resumen del experimento en PDF
    a partir de pipeline_config.json.
    Espera "?directory_path=<ruta>" en GET.
    """
    directory_path = request.GET.get("directory_path")
    if not directory_path or not os.path.isdir(directory_path):
        return JsonResponse({"status": "La ruta del experimento no es válida."}, status=400)

    pipeline_config_path = os.path.join(directory_path, "pipeline_config.json")
    if not os.path.exists(pipeline_config_path):
        return JsonResponse({"status": "No se encontró el pipeline_config.json."}, status=404)

    output_pdf_path = os.path.join(directory_path, "experiment_summary.pdf")

    try:
        generate_experiment_summary_pdf(pipeline_config_path, output_pdf_path)

        if not os.path.isfile(output_pdf_path):
            return JsonResponse({"status": "No se generó el PDF de resumen."}, status=500)

        with open(output_pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()

        response = HttpResponse(pdf_content, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="experiment_summary.pdf"'
        return response

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Error de validación al generar el resumen: {e}", exc_info=True)
        return JsonResponse({"status": f"Error en el contenido de pipeline_config.json: {str(e)}"}, status=400)
    except Exception as e:
        logger.error(f"Error general al generar el resumen: {e}", exc_info=True)
        return JsonResponse({"status": f"Error interno al generar el resumen: {str(e)}"}, status=500)
