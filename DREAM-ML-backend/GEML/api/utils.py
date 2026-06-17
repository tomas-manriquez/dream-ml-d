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

import os
import subprocess
import logging
import requests
import time
import socket
import shutil
import pandas as pd
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
mlflow_process = None # Variable global para el proceso de MLflow
from mlflow import (
    set_tracking_uri,
    set_experiment,
    start_run,
    log_param,
    log_metric,
    log_artifact,
    get_experiment_by_name,
)
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync



from pathlib import Path
import os
import subprocess
import logging

# Configuración básica de logging (asumiendo que ya existe)
logger = logging.getLogger(__name__)

def init_dvc_logic(experiment_dir: str) -> dict:
    """
    Inicializa un entorno DVC en el directorio especificado, configurando Git y DVC si es necesario.

    Realiza:
    1. Validación del directorio de experimentos
    2. Inicialización de repositorio Git (si no existe)
    3. Inicialización de DVC (si no está configurado)
    4. Configuración de directorio de caché (.dvc_cache)
    5. Creación/actualización de .gitignore con exclusiones básicas
    6. Commit inicial si se crearon nuevos elementos

    Parámetros:
        experiment_dir (str): Ruta absoluta al directorio de experimentos

    Retorna:
        dict: {"status": str, "experiment_dir": str}

    Excepciones:
        ValueError: Directorio no existe o es inválido
        subprocess.CalledProcessError: Fallo en comandos externos
        OSError: Errores de sistema de archivos

    Ejemplo de retorno:
        {"status": "DVC inicializado correctamente", "experiment_dir": "/ruta/experimento"}
    """
    # Convertir a Path y validar
    exp_path = Path(experiment_dir).resolve()
    if not exp_path.is_dir():
        raise ValueError(f"Directorio inválido: {experiment_dir}")

    # Definir rutas importantes
    cache_dir = exp_path / ".dvc_cache"
    git_dir = exp_path / ".git"
    dvc_dir = exp_path / ".dvc"
    gitignore = exp_path / ".gitignore"
    
    # Flags para operaciones realizadas
    needs_commit = False

    try:
        # 1. Inicialización de Git
        if not git_dir.exists():
            subprocess.run(["git", "init"], cwd=exp_path, check=True)
            logger.info("Repositorio Git inicializado")
            needs_commit = True

        # 2. Inicialización de DVC
        if not dvc_dir.exists():
            subprocess.run(["dvc", "init"], cwd=exp_path, check=True)
            logger.info("Repositorio DVC inicializado")
            needs_commit = True

        # 3. Configuración de caché
        cache_dir.mkdir(exist_ok=True)
        subprocess.run(["dvc", "config", "cache.dir", str(cache_dir)], 
                      cwd=exp_path, check=True)
        logger.info(f"Caché DVC configurada en: {cache_dir}")

        # 4. Gestión de .gitignore
        gitignore_content = "\n".join([
            "# DVC config",
            ".dvc/tmp/",
            ".dvc/cache/",
            ".dvc/state",
            ".dvc/config.local",
            ".dvc_cache/"
        ])

        needs_update = False
        gitignore_path = exp_path / ".gitignore"

        if not gitignore_path.exists():
            gitignore_path.write_text(gitignore_content, encoding="utf-8")
            logger.info(".gitignore creado")
            needs_update = True
        else:
            # Verificar si necesita actualización
            current_content = gitignore_path.read_text(encoding="utf-8")
            for line in gitignore_content.split("\n"):
                if line.strip() and line not in current_content:
                    with gitignore_path.open("a", encoding="utf-8") as f:
                        f.write("\n" + line)
                    logger.info(f"Se agregó {line} a .gitignore")
                    needs_update = True

        # 5. Commit inicial si hubo cambios
        if needs_commit or needs_update:
            # Configuración local de usuario
            subprocess.run(["git", "config", "user.email", "geml@user.com"], 
                          cwd=exp_path, check=True)
            subprocess.run(["git", "config", "user.name", "geml user"], 
                          cwd=exp_path, check=True)
            
            # Agregar cambios relevantes
            subprocess.run(["git", "add", ".gitignore"], cwd=exp_path, check=True)
            if dvc_dir.exists():
                subprocess.run(["git", "add", ".dvc"], cwd=exp_path, check=True)
                
            subprocess.run(["git", "commit", "-m", "Configuración inicial DVC"], 
                          cwd=exp_path, check=True)
            logger.info("Commit inicial realizado")

        return {
            "status": "DVC inicializado correctamente",
            "experiment_dir": str(exp_path)
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Error en comando: {e.cmd} - {e.stderr}")
        raise
    except OSError as e:
        logger.error(f"Error de sistema: {str(e)}")
        raise





import os
import subprocess
import logging

# Configurar logger (asumiendo que está configurado en otro lugar)
logger = logging.getLogger(__name__)

def configure_dvc_remote_logic(experiment_dir: str) -> dict:
    """
    Configura un remoto DVC compartido en el directorio base del experimento.
    
    Realiza las siguientes acciones:
    1. Construye la ruta del remoto compartido en el directorio base
    2. Crea el directorio remoto si no existe
    3. Verifica si el remoto ya está configurado en DVC
    4. Añade el remoto si no existe
    5. Establece el remoto como predeterminado
    
    Args:
        experiment_dir: Ruta del directorio del experimento
    
    Returns:
        dict: Resultado de la operación con formato:
              {
                  "status": "mensaje descriptivo",
                  "remote_path": "ruta/del/remoto"
              }
    
    Raises:
        subprocess.CalledProcessError: Si falla algún comando DVC
        OSError: Si hay problemas al crear el directorio remoto
    """
    # Construir ruta del remoto compartido
    base_dir = os.path.dirname(experiment_dir)
    shared_remote = os.path.join(base_dir, "dvc_remote")
    remote_name = "shared_remote"
    
    try:
        # Crear directorio remoto con manejo seguro
        os.makedirs(shared_remote, exist_ok=True)
        logger.debug(f"Directorio remoto creado/verificado: {shared_remote}")
        
        # Verificar remotos existentes
        existing_remotes = _get_existing_dvc_remotes(experiment_dir)
        
        # Añadir remoto si no existe
        if remote_name not in existing_remotes:
            _add_dvc_remote(experiment_dir, remote_name, shared_remote)
            logger.info(f"Remoto DVC añadido: {remote_name} -> {shared_remote}")
        else:
            logger.info(f"Remoto '{remote_name}' ya existe, omitiendo creación")
        
        # Establecer como predeterminado
        _set_default_dvc_remote(experiment_dir, remote_name)
        logger.info(f"Remoto predeterminado establecido: {remote_name}")
        
        return {
            "status": "Remoto DVC configurado exitosamente en ubicación compartida",
            "remote_path": shared_remote
        }
        
    except (subprocess.CalledProcessError, OSError) as error:
        logger.exception(f"Error configurando remoto DVC: {str(error)}")
        raise

def _get_existing_dvc_remotes(experiment_dir: str) -> list:
    """Obtiene la lista de remotos DVC configurados"""
    result = subprocess.run(
        ["dvc", "remote", "list"],
        cwd=experiment_dir,
        check=True,
        capture_output=True,
        text=True
    )
    # Extraer solo los nombres de los remotos
    return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]

def _add_dvc_remote(experiment_dir: str, name: str, path: str):
    """Añade un nuevo remoto DVC"""
    subprocess.run(
        ["dvc", "remote", "add", name, path],
        cwd=experiment_dir,
        check=True
    )

def _set_default_dvc_remote(experiment_dir: str, name: str):
    """Establece un remoto como predeterminado"""
    subprocess.run(
        ["dvc", "remote", "default", name],
        cwd=experiment_dir,
        check=True
    )




def is_mlflow_running(url: str, timeout: int = 5) -> bool:
    """
    Verifica si el servidor de MLflow está disponible en la URL especificada.

    Parámetros:
        url (str): URL del servidor MLflow a verificar
        timeout (int, opcional): Tiempo máximo de espera en segundos. Valor por defecto: 5

    Retorna:
        bool: True si el servidor responde con código 200 (OK) o 405 (Método no permitido),
              False en caso de error de conexión u otros errores.
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in [200, 405]
    except requests.ConnectionError:
        return False
    except requests.RequestException as e:
        logger.error(f"Error al verificar MLflow: {e}", exc_info=True)
        return False
    

def start_mlflow_logic(base_dir: str) -> dict:
    """
    Inicia un servidor MLflow local con almacenamiento SQLite y directorio de artefactos.

    Configuración:
        - Backend store: {base_dir}/shared_mlflow.db (SQLite)
        - Artifact store: {base_dir}/artifacts/
        - Logs: {base_dir}/mlflow_logs/
        - Host: 0.0.0.0
        - Puerto: 5000

    Parámetros:
        base_dir (str): Directorio base para almacenamiento y logs

    Retorna:
        dict: Diccionario con información del servidor:
            - status: mensaje de estado
            - backend_store_uri: URI de la base de datos
            - artifact_store: ruta de artefactos
            - log_stdout: ruta del log stdout
            - log_stderr: ruta del log stderr

    Excepciones:
        ValueError: Si el directorio base no es válido
        RuntimeError: Si el servidor no inicia después de varios intentos
        Exception: Para errores inesperados durante el inicio
    """
    global mlflow_process

    # Validación del directorio base
    if not base_dir or not os.path.isdir(base_dir):
        raise ValueError("La ruta proporcionada no es válida o no es un directorio")

    # Configuración de rutas
    shared_db_path = os.path.join(base_dir, "shared_mlflow.db")
    artifact_store = os.path.join(base_dir, "artifacts")
    log_dir = os.path.join(base_dir, "mlflow_logs")
    
    # Crear directorios necesarios
    os.makedirs(artifact_store, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Directorios configurados: Artefactos: {artifact_store}, Logs: {log_dir}")

    # Detener servidor previo si está activo
    if mlflow_process and mlflow_process.poll() is None:
        logger.info("Deteniendo instancia previa de MLflow...")
        mlflow_process.terminate()
        mlflow_process.wait(timeout=10)
        mlflow_process = None

    # Configuración del comando
    command = [
        "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{shared_db_path}",
        "--default-artifact-root", artifact_store,
        "--host", "0.0.0.0",
        "--port", "5000"
    ]

    # Archivos de log
    stdout_path = os.path.join(log_dir, "mlflow_stdout.log")
    stderr_path = os.path.join(log_dir, "mlflow_stderr.log")

    try:
        # Abrir archivos de log en modo append
        with open(stdout_path, 'a') as stdout_log, \
             open(stderr_path, 'a') as stderr_log:

            # Iniciar proceso
            mlflow_process = subprocess.Popen(
                command,
                stdout=stdout_log,
                stderr=stderr_log,
                text=True,
                encoding='utf-8'
            )
            
            logger.info(f"Servidor MLflow iniciado con PID: {mlflow_process.pid}")
            logger.info(f"Backend store: sqlite:///{shared_db_path}")
            logger.info(f"Artifact store: {artifact_store}")

            # Verificar inicio del servidor
            mlflow_port = os.environ.get('MLFLOW_PORT', '5000')
            mlflow_url = os.environ.get('MLFLOW_UI_URL', f'http://localhost:{mlflow_port}')
            max_retries = 10
            for attempt in range(1, max_retries + 1):
                if is_mlflow_running(mlflow_url):
                    logger.info("Servidor MLflow disponible")
                    break
                    
                logger.info(f"Verificando disponibilidad ({attempt}/{max_retries})...")
                time.sleep(2)
            else:
                raise RuntimeError("El servidor no respondió después de múltiples intentos")

    except Exception as e:
        logger.exception("Error crítico al iniciar MLflow")
        # Limpieza en caso de error
        if 'mlflow_process' in globals() and mlflow_process.poll() is None:
            mlflow_process.terminate()
        raise RuntimeError(f"Error al iniciar MLflow: {str(e)}") from e

    return {
        "status": "Servidor MLflow iniciado exitosamente",
        "backend_store_uri": f"sqlite:///{shared_db_path}",
        "artifact_store": artifact_store,
        "log_stdout": stdout_path,
        "log_stderr": stderr_path,
    }




def analyze_csv_logic(csv_file) -> dict:
    """
    Lógica interna para analizar columnas de un CSV.
    - Recibe un 'csv_file' (un archivo ya abierto o un objeto InMemoryUploadedFile).
    - Retorna un dict con { "columns": [...] }.
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





def is_port_available(port: int) -> bool:
    """Verifica si un puerto está disponible en el host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def start_jupyter_logic(experiment_dir: str, run_id: str, port: int = 8888) -> dict:
    """
    Inicia un servidor de Jupyter Notebook para el EDA manual:
      - Copia 'EDA_manual.ipynb' desde notebooks/ al experiment_dir.
      - Arranca 'jupyter notebook' sin browser/token en un puerto dado.
      - Registra el notebook en MLflow.

    Retorna un diccionario con:
      {
        "success": True,
        "notebook_url": "...",
        "notebook_path": "EDA_manual.ipynb"
      }
    """

    # . Validaciones
    if not experiment_dir or not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Directorio del experimento no válido: {experiment_dir}")

    if not run_id:
        raise ValueError("El run_id no fue proporcionado o es inválido.")

    if not is_port_available(port):
        raise OSError(f"El puerto {port} no está disponible.")

    # 2. Ubicación de la plantilla del Notebook
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(project_root, "notebooks", "EDA_manual.ipynb")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError("Plantilla EDA no encontrada en notebooks/EDA_manual.ipynb")

    # 3. Copiar la plantilla al directorio del experimento
    rel_notebook_path = "EDA_manual.ipynb"
    notebook_abs_path = os.path.join(experiment_dir, rel_notebook_path)

    try:
        shutil.copy(template_path, notebook_abs_path)
        logger.info(f"Plantilla EDA copiada a: {notebook_abs_path}")
    except Exception as e:
        logger.error(f"Error al copiar la plantilla EDA: {e}", exc_info=True)
        raise RuntimeError(f"Error al copiar la plantilla EDA: {e}")

    # 4. Configurar MLflow
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

    # 5. Ejecutar un run anidado en MLflow
    try:
        with start_run(nested=True) as run:
            run_id = run.info.run_id
            logger.info(f"Run anidado de MLflow iniciado (Run ID: {run_id})")

            log_param("step", "start_jupyter")
            log_param("notebook_path", rel_notebook_path)
            log_param("port", port)
            log_param("notebook_url", f"http://localhost:{port}/tree/{rel_notebook_path}")

            log_artifact(notebook_abs_path, artifact_path="manual_eda")
            logger.info(f"Notebook registrado en MLflow: {notebook_abs_path}")

    except Exception as e:
        logger.error(f"Error al iniciar run anidado en MLflow: {e}", exc_info=True)
        raise RuntimeError(f"Error al iniciar run anidado en MLflow: {e}")

    # 6. Iniciar Jupyter Notebook
    try:
        jupyter_config_dir = os.path.join(experiment_dir, ".jupyter")
        os.makedirs(jupyter_config_dir, exist_ok=True)

        command = [
            "python", "-m", "notebook",
            "--no-browser",
            f"--notebook-dir={experiment_dir}",
            "--NotebookApp.token=",
            "--NotebookApp.disable_check_xsrf=True",
            f"--port={port}",
        ]

        process = subprocess.Popen(
            command,
            cwd=experiment_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Removed shell=True  # Añadido para Windows
            # if shell=True then command should be a string, not list
        )

        logger.info(f"Jupyter Notebook iniciado en puerto {port}.")

    except FileNotFoundError as e:
        logger.error(f"Comando no encontrado: {e}", exc_info=True)
        raise RuntimeError(f"Comando no encontrado: {e}")
    except Exception as e:
        logger.error(f"Error al iniciar Jupyter Notebook: {e}", exc_info=True)
        raise RuntimeError(f"Error al iniciar Jupyter Notebook: {e}")

    # 🚀 Retorno SIN pipeline_config.json
    return {
        "success": True,
        "notebook_url": f"http://localhost:{port}/tree/{rel_notebook_path}",
        "notebook_path": rel_notebook_path
    }















#channel_layer = get_channel_layer()  # Para enviar mensajes vía WebSocket

def send_progress_update(step, status):
    """
    Envía un mensaje al grupo de progreso vía channel layer.
    Se espera que el mensaje tenga la forma: { "step": <nombre_step>, "status": <estado> }
    """
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "progreso_group",
        {
            "type": "send_progress",
            "step": step,
            "status": status,
        }
    )





from svglib.svglib import svg2rlg
from reportlab.lib.units import cm
from reportlab.graphics import renderPDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import json
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Frame,
    PageTemplate,
    Preformatted,
    Image,
    LongTable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "media", "GEML.svg")


def header_footer(canvas, doc):
    canvas.saveState()
    # Agregar el logo si existe
    header_x = doc.leftMargin
    if os.path.exists(LOGO_PATH):
        ext = os.path.splitext(LOGO_PATH)[1].lower()
        if ext == ".svg":
            drawing = svg2rlg(LOGO_PATH)
            # Escalar el dibujo para que tenga un ancho de 2 cm
            logo_width = 2 * cm
            scale = logo_width / drawing.width
            logo_height = drawing.height * scale
            drawing.width = logo_width
            drawing.height = logo_height
            renderPDF.draw(drawing, canvas, doc.leftMargin, A4[1] - logo_height - 0.5*cm)
            header_x = doc.leftMargin + logo_width + 0.5*cm
        else:
            logo_width = 2 * cm
            logo_height = 2 * cm
            canvas.drawImage(
                LOGO_PATH,
                doc.leftMargin,
                A4[1] - logo_height - 0.5*cm,
                width=logo_width,
                height=logo_height,
                preserveAspectRatio=True
            )
            header_x = doc.leftMargin + logo_width + 0.5*cm

    # Encabezado
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(header_x, A4[1] - 1 * cm, "Resumen del Experimento")

    # Línea divisoria debajo del encabezado
    canvas.setStrokeColor(colors.grey)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 1.2 * cm, A4[0] - doc.rightMargin, A4[1] - 1.2 * cm)

    # Pie de página: número de página centrado
    canvas.setFont("Helvetica", 9)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(A4[0] / 2.0, 1 * cm, f"Página {page_num}")
    canvas.restoreState()


def chunk_list(lst, chunk_size=10):
    """
    Función auxiliar para trocear una lista 'lst' en sublistas
    de longitud 'chunk_size'.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


def generate_experiment_summary_pdf(pipeline_config_path, output_pdf_path):
    """
    Genera un resumen en PDF a partir del pipeline_config.json con diseño mejorado,
    incluyendo logo, encabezado, pie de página y estilos personalizados.
    Además trocea manualmente listas grandes (por ejemplo 'input_features')
    para evitar celdas muy altas que provoquen LayoutError.
    """
    # 1. Validar que el archivo exista y no esté vacío
    if not os.path.isfile(pipeline_config_path) or os.path.getsize(pipeline_config_path) == 0:
        raise FileNotFoundError(f"El archivo '{pipeline_config_path}' no existe o está vacío.")

    # 2. Intentar cargar el JSON
    with open(pipeline_config_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en '{pipeline_config_path}': {str(e)}")

    # 3. Validar la presencia mínima de las claves esperadas
    if "experiment_id" not in config:
        raise KeyError("El pipeline_config.json no contiene la clave 'experiment_id'.")
    if "steps" not in config or not isinstance(config["steps"], list):
        raise KeyError("El pipeline_config.json no contiene una lista 'steps' válida.")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=3 * cm,
        bottomMargin=2 * cm
    )

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CustomTitle",
                              fontName="Helvetica-Bold",
                              fontSize=20,
                              leading=24,
                              alignment=1,
                              spaceAfter=12))
    styles.add(ParagraphStyle(name="CustomHeading",
                              fontName="Helvetica-Bold",
                              fontSize=14,
                              leading=18,
                              textColor=colors.darkgreen,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomNormal",
                              fontName="Helvetica",
                              fontSize=9,
                              leading=10,
                              spaceAfter=6))
    styles.add(ParagraphStyle(name="PreformattedSmall",
                              fontName="Courier",
                              fontSize=8,
                              leading=9,
                              spaceAfter=6))

    elements = []

    # Título principal
    title = Paragraph("Resumen del Experimento", styles["CustomTitle"])
    elements.append(title)

    # Mostrar Experiment ID
    exp_id = config.get("experiment_id", "N/A")
    p_exp = Paragraph(f"<b>Experiment ID:</b> {exp_id}", styles["CustomNormal"])
    elements.append(p_exp)
    elements.append(Spacer(1, 12))

    # Recorrer cada paso del pipeline
    steps = config["steps"]
    for idx, step in enumerate(steps, 1):
        step_title = Paragraph(f"Paso {idx}: {step.get('step', 'N/A')}", styles["CustomHeading"])
        elements.append(step_title)
        elements.append(Spacer(1, 6))

        data_table = [
            [
                Paragraph("<b>Parámetro</b>", styles["CustomNormal"]),
                Paragraph("<b>Valor</b>", styles["CustomNormal"])
            ]
        ]

        for key, value in step.items():
            # Si es una lista muy grande, la troceamos
            if isinstance(value, list) and len(value) > 15:
                # Dividir en sublistas de 10 para no saturar
                part_num = 1
                for chunk in chunk_list(value, 10):
                    # Convertimos chunk en JSON
                    chunk_str = json.dumps(chunk, indent=2, ensure_ascii=False)
                    chunk_str = chunk_str.replace("\n", "<br/>")
                    value_element = Paragraph(chunk_str, styles["PreformattedSmall"])

                    label_text = f"{key} (parte {part_num})"
                    data_table.append([
                        Paragraph(f"<b>{label_text}</b>", styles["CustomNormal"]),
                        value_element
                    ])
                    part_num += 1

            elif isinstance(value, (dict, list)):
                # Si es lista pero no tan grande, o un dict
                val_str = json.dumps(value, indent=2, ensure_ascii=False)
                val_str = val_str.replace("\n", "<br/>")
                value_element = Paragraph(val_str, styles["PreformattedSmall"])

                data_table.append([
                    Paragraph(f"<b>{key}</b>", styles["CustomNormal"]),
                    value_element
                ])
            else:
                # Es un string o número
                data_table.append([
                    Paragraph(f"<b>{key}</b>", styles["CustomNormal"]),
                    Paragraph(str(value), styles["CustomNormal"])
                ])

        # LongTable para permitir splits en varias páginas
        table = LongTable(
            data_table,
            colWidths=[5 * cm, 10 * cm],
            repeatRows=1,
            splitInRow=True
        )
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('WORDWRAP', (0, 0), (-1, -1), True),
            ('SPLITTABLE', (0, 0), (-1, -1), True),
        ])
        table.setStyle(table_style)

        elements.append(table)
        elements.append(Spacer(1, 12))

    # PageTemplate con encabezado y pie
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='test', frames=frame, onPage=header_footer)
    doc.addPageTemplates([template])

    doc.build(elements)